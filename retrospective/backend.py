"""JSON-lines client for the pinned llama.cpp worker; greedy decoding only."""
import json
import subprocess
from pathlib import Path


class NativeEngine:
    def __init__(self, worker, model, log_path, context=2048, gpu_layers=99):
        self.log = Path(log_path).open('w')
        self.process = subprocess.Popen([str(worker), str(model), str(context), str(gpu_layers)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, encoding='utf-8')
        self.tokens = []
        self.snapshots = {}
        try:
            response = self._read()
            if response.get('ready') is not True:
                raise RuntimeError(f'Worker did not initialize: {response}')
            self.info = response['info']
        except BaseException:
            self.close()
            raise

    def _read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f'Worker exited; inspect {self.log.name}')
        value = json.loads(line)
        if 'error' in value:
            raise RuntimeError(value['error'])
        return value

    def request(self, op, **fields):
        self.process.stdin.write(json.dumps(dict(op=op, **fields), allow_nan=False)+'\n')
        self.process.stdin.flush()
        return self._read()

    def tokenize(self, text):
        return self.request('tokenize', text=text)['tokens']

    def text(self, tokens):
        return self.request('detokenize', tokens=list(tokens))['text']

    def steer(self, direction=None, layer=-1):
        values = [] if direction is None else list(map(float, direction))
        return self.request('steer', direction=values, layer=layer)

    def clear(self):
        self.request('clear')
        self.tokens = []

    def evaluate(self, tokens, layers=(), logits=False):
        tokens = list(map(int, tokens))
        result = self.request('eval', tokens=tokens, capture_layers=list(layers), logits=logits)
        self.tokens.extend(tokens)
        if result['position'] != len(self.tokens):
            raise RuntimeError('Client/worker position disagreement')
        return result

    def snapshot(self, name):
        result = self.request('snapshot', name=name)
        self.snapshots[name] = tuple(self.tokens)
        return result

    def restore(self, name):
        result = self.request('restore', name=name)
        self.tokens = list(self.snapshots[name])
        if result['position'] != len(self.tokens):
            raise RuntimeError('Restored position disagreement')
        return result

    def drop(self, name):
        self.request('drop', name=name)
        self.snapshots.pop(name, None)

    def replay(self, tokens, layers=(), logits=False):
        self.clear()
        return self.evaluate(tokens, layers, logits)

    def generate(self, response, max_tokens):
        generated = []
        stopping = 'token_limit'
        for _ in range(max_tokens):
            token = response['next_token']
            is_eog = response['next_is_eog']
            generated.append(token)
            response = self.evaluate([token])
            if is_eog:
                stopping = 'eog'
                break
        return dict(token_ids=generated, text=self.text(generated), stopping=stopping,
                    processed_tokens=len(self.tokens))

    def close(self):
        if getattr(self, 'process', None):
            if self.process.stdin:
                self.process.stdin.close()
            try:
                self.process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            if self.process.stdout:
                self.process.stdout.close()
        if getattr(self, 'log', None):
            self.log.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
