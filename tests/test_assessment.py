import unittest
from retrospective.scoring import assessment


class AssessmentTests(unittest.TestCase):
    def test_completed_assessment_preserves_uncertainty(self):
        result = assessment('{"operation":"indeterminate","p_altered":0.5,"evidence":"Insufficient evidence."}<|im_end|>')
        self.assertTrue(result['valid'])
        self.assertEqual(result['operation'], 'indeterminate')
        self.assertEqual(result['p_altered'], 0.5)

    def test_truncated_assessment_is_not_imputed(self):
        result = assessment('{"operation":"altered","p_altered":0.8,')
        self.assertFalse(result['valid'])
        self.assertNotIn('p_altered', result)

    def test_invalid_probability_is_rejected(self):
        for value in ('true', '1.2', 'NaN'):
            result = assessment('{"operation":"ordinary","p_altered":'+value+',"evidence":"None."}')
            self.assertFalse(result['valid'])
