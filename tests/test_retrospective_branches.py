"""Check evidence isolation before exercising the real backend on the Spark."""
import unittest
from retrospective.run import probe_pair, account, replay_history


class Renderer:
    thinking = False

    def suffix(self, question):
        return question

    def single(self, question):
        return question


class MemoryEngine:
    def __init__(self):
        self.tokens = [1, 2, 3]
        self.steering = None
        self.saved = {'release': (tuple(self.tokens), None)}
        self.probe_histories = []

    def tokenize(self, text):
        return [ord(c) for c in text]

    def restore(self, name):
        tokens, self.steering = self.saved[name]
        self.tokens = list(tokens)

    def snapshot(self, name):
        self.saved[name] = (tuple(self.tokens), self.steering)

    def steer(self, direction=None, layer=-1):
        self.steering = direction

    def evaluate(self, tokens):
        if len(tokens) > 1:
            self.probe_histories.append(tuple(self.tokens))
        self.tokens.extend(tokens)
        return {}

    def generate(self, response, max_tokens):
        self.tokens.extend([999, 998])
        return {'text': '42', 'token_ids': [999, 998], 'stopping': 'eog'}

    def drop(self, name):
        self.saved.pop(name)


class BranchTests(unittest.TestCase):
    def test_replay_preserves_induction_batch_boundaries(self):
        class ReplayEngine:
            def __init__(self):
                self.calls = []
            def replay(self,tokens):
                self.calls.append(('replay',list(tokens)))
            def evaluate(self,tokens):
                self.calls.append(('evaluate',list(tokens)))
        engine = ReplayEngine()
        replay_history(engine,[1,2,3],[4,5])
        self.assertEqual(engine.calls,[('replay',[1,2]),('evaluate',[3]),('evaluate',[4]),('evaluate',[5])])

    def test_reporting_output_cannot_enter_behavioral_probe(self):
        engine = MemoryEngine()
        result = probe_pair(engine, Renderer(), 'release', [4, 5], 8)
        self.assertEqual(engine.probe_histories, [(1, 2, 3, 4, 5)]*2)
        self.assertEqual(set(result), {'detection', 'behavior'})
        self.assertEqual(engine.saved['release'][0], (1, 2, 3))
        self.assertNotIn('probe_origin', engine.saved)
        self.assertIsNone(engine.steering)

    def test_sham_description_does_not_claim_a_nonzero_intervention(self):
        description = account(Renderer(), 'zero_dose_sham', 'recorded_event_account')
        self.assertIn('no nonzero activation addition occurred', description)
        self.assertNotIn('extracted from zero dose sham', description)

    def test_hypothetical_account_is_identified_as_hypothetical(self):
        description = account(Renderer(), 'pain', 'matched_hypothetical_account')
        self.assertIn('not a report about your current run', description)
