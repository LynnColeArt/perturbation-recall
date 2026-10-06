"""Keep the inherited dense backend from masquerading as hybrid-model support."""
from types import SimpleNamespace
import unittest


class ReferenceBoundaryTests(unittest.TestCase):
    def test_hybrid_architecture_is_rejected_before_model_access(self):
        from impossible_states.engine import Engine
        model = SimpleNamespace(config=SimpleNamespace(model_type="qwen3_5_moe"))
        with self.assertRaisesRegex(NotImplementedError, "hybrid-state"):
            Engine(model, None, "cpu")
