import unittest
import torch
from src.pipelines.factories import EvaluatorsFactory
from src.evaluators import CosineSimilarity


class TestEvaluatorFactory(unittest.TestCase):
    def setUp(self):
        """Set up a factory instance for each test."""
        self.factory = EvaluatorsFactory()

    def test_create_argmax_evaluator(self):
        """Test creating the custom 'argmax' evaluator."""
        config = [{"mode": "single", "instances": {"argmax": {"dim": 1}}}]

        evaluator_fn = self.factory.build(config)
        self.assertTrue(callable(evaluator_fn))

        logits = torch.randn(4, 10)
        expected_preds = torch.argmax(logits, dim=1)
        actual_preds = evaluator_fn(logits)

        self.assertTrue(torch.equal(actual_preds, expected_preds))

    def test_create_cosine_similarity_evaluator(self):
        """Test creating the 'cosine_similarity' evaluator as a class instance."""
        config = [{"mode": "single", "instances": {"cosine_similarity": {}}}]

        evaluator_module = self.factory.build(config)

        self.assertIsInstance(evaluator_module, CosineSimilarity)
        # Check new default parameters from YAML spec
        self.assertEqual(evaluator_module.use_l2_norm, True)
        self.assertEqual(evaluator_module.temperature, 1.0)

    def test_build_multiple_evaluators_returns_list(self):
        """Test that building multiple evaluators returns a list of instances."""
        configs = [
            {"mode": "single", "instances": {"argmax": {"dim": 1}}},
            {"mode": "single", "instances": {"cosine_similarity": {"use_l2_norm": False}}},
        ]

        evaluators = self.factory.build(configs)

        self.assertIsInstance(evaluators, list)
        self.assertEqual(len(evaluators), 2)
        self.assertTrue(callable(evaluators[0]))
        self.assertIsInstance(evaluators[1], CosineSimilarity)
        self.assertEqual(evaluators[1].use_l2_norm, False)

    def test_fail_fast_on_unknown_evaluator(self):
        """Test that requesting an unknown evaluator raises an error."""
        config = [{"mode": "single", "instances": {"non_existent_evaluator": {}}}]

        with self.assertRaises(ValueError, msg="Should fail on unknown evaluator type"):
            self.factory.build(config)

    def test_pass_yaml_params_to_evaluator(self):
        """Test that parameters from YAML are correctly passed to the evaluator."""
        config = [{"mode": "single", "instances": {"cosine_similarity": {"use_l2_norm": False, "temperature": 0.5}}}]

        evaluator_module = self.factory.build(config)

        self.assertIsInstance(evaluator_module, CosineSimilarity)
        self.assertEqual(evaluator_module.use_l2_norm, False)
        self.assertEqual(evaluator_module.temperature, 0.5)


if __name__ == "__main__":
    unittest.main()
