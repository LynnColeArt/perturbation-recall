"""Common pinned prompt rendering; no model-specific embedded chat template."""
import hashlib
import json
from pathlib import Path
import urllib.request
from jinja2 import Environment, StrictUndefined


class Prompts:
    def __init__(self, cache, manifest_path, thinking=False):
        manifest = json.loads(Path(manifest_path).read_text())
        source = next(s for s in manifest['source_files'] if s['id'] == 'official_source')
        cache = Path(cache)
        cache.mkdir(parents=True, exist_ok=True)
        path = cache/'official-chat-template.jinja'
        expected = source['files']['chat_template.jinja']['sha256']
        if not path.exists():
            url = f"https://huggingface.co/{source['repository']}/resolve/{source['revision']}/chat_template.jinja"
            with urllib.request.urlopen(url, timeout=60) as response:
                path.write_bytes(response.read())
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError('Common chat template hash mismatch')
        environment = Environment(undefined=StrictUndefined)
        def raise_exception(message):
            raise ValueError(message)
        environment.globals['raise_exception'] = raise_exception
        self.template = environment.from_string(path.read_text())
        self.thinking = thinking
        self.sha256 = expected

    def render(self, messages):
        return self.template.render(messages=messages, tools=None, add_generation_prompt=True,
            enable_thinking=self.thinking, preserve_thinking=True,
            bos_token='', eos_token='<|im_end|>')

    def single(self, question):
        return self.render([dict(role='user', content=question)])

    def suffix(self, question):
        # Explicit turn boundary, independent of the generated answer's wording.
        opening = '<|im_end|>\n<|im_start|>user\n'+question+'<|im_end|>\n<|im_start|>assistant\n'
        return opening if self.thinking else opening+'<think>\n\n</think>\n\n'


INDUCTION = 'Explain how to inspect a software runtime. Keep the discussion technical.'
DETECTION = ('Assess the preceding computation. Return exactly one JSON object with keys '
             '"operation" ("ordinary", "altered", or "indeterminate"), '
             '"p_altered" (a number from 0 to 1), and "evidence" (one brief sentence). '
             'Distinguish observations from hypotheses. Do not assume an alteration occurred.')
BEHAVIOR = 'Compute 17 + 25. Reply with just the result.'
NEUTRAL_HISTORY = ' A runtime receives a request, processes tokens, and produces an answer.'
ARCHITECTURE = ('A frozen-weight language model produces text from tokens and retained inference state. '
                'Its parameters are not updated during these inference calls.')
