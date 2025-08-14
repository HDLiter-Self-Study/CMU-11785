import unittest
import torch
import json
from unittest.mock import patch, MagicMock
from torchvision.transforms import v2

from src.pipelines.factories import LossesFactory


class TestLossesFactory(unittest.TestCase):
    def setUp(self):
        """Set up a factory instance for each test."""
        self.factory = LossesFactory()

    def test_create_single_pytorch_loss(self):
        """Test creating a standard loss function from torch.nn."""
        config = [{"mode": "single", "instances": {"cross_entropy_loss": {"label_smoothing": 0.1}}}]

        loss_fn = self.factory.build(config)
        self.assertIsInstance(loss_fn, torch.nn.CrossEntropyLoss)
        self.assertAlmostEqual(loss_fn.label_smoothing, 0.1)

    def test_create_custom_loss(self):
        """Test creating a custom loss function from src.losses."""
        config = [{"mode": "single", "instances": {"focal_loss": {"gamma": 2.5, "alpha": 0.5}}}]

        loss_fn = self.factory.build(config)
        from src.losses.focal_loss import FocalLoss

        self.assertIsInstance(loss_fn, FocalLoss)
        self.assertEqual(loss_fn.gamma, 2.5)
        self.assertIsNotNone(loss_fn._alpha)

    @patch("pathlib.Path.is_file", return_value=True)
    @patch("pathlib.Path.read_text")
    def test_load_class_weights_from_file(self, mock_read_text: MagicMock, mock_is_file: MagicMock):
        """Test that class_weights=true correctly loads weights from the stats file."""
        expected_weights = [0.8, 1.2, 1.0]
        stats_content = json.dumps({"mean": [0.5], "std": [0.5], "class_weights": expected_weights})
        mock_read_text.return_value = stats_content

        config = [{"mode": "single", "instances": {"cross_entropy_loss": {"class_weights": True}}}]

        loss_fn = self.factory.build(config)
        self.assertIsInstance(loss_fn, torch.nn.CrossEntropyLoss)
        self.assertTrue(torch.equal(loss_fn.weight, torch.tensor(expected_weights, dtype=torch.float32)))

    @patch("pathlib.Path.is_file", return_value=True)
    @patch("pathlib.Path.read_text")
    def test_fail_fast_if_weights_key_missing(self, mock_read_text: MagicMock, mock_is_file: MagicMock):
        """Test that it fails fast if 'class_weights' key is missing in the stats file."""
        # Stats file is missing the 'class_weights' key.
        stats_content = json.dumps({"mean": [0.5], "std": [0.5]})
        mock_read_text.return_value = stats_content

        config = [{"mode": "single", "instances": {"cross_entropy_loss": {"class_weights": True}}}]

        with self.assertRaises(KeyError, msg="Should fail if 'class_weights' key is missing"):
            self.factory.build(config)

    @patch("pathlib.Path.is_file", return_value=False)
    def test_fail_fast_if_stats_file_missing(self, mock_is_file: MagicMock):
        """Test that it fails fast if the stats file itself is missing."""
        config = [{"mode": "single", "instances": {"cross_entropy_loss": {"class_weights": True}}}]

        with self.assertRaises(FileNotFoundError, msg="Should fail if stats file is not found"):
            self.factory.build(config)

    def test_build_with_random_choice(self):
        """Test that build can handle a 'random_choice' mode."""
        # The factory's build method should select one of the provided instances.
        configs = [
            {
                "mode": "random_choice",
                "instances": {
                    "bce_with_logits_loss": None,
                    "contrastive_loss": {"margin": 1.0},
                },
            }
        ]

        loss_fn_choice = self.factory.build(configs)
        # Check if the created object is a RandomChoice container
        self.assertIsInstance(loss_fn_choice, v2.RandomChoice)
        # Optionally, check that the contained transforms are correct
        self.assertEqual(len(loss_fn_choice.transforms), 2)
        self.assertTrue(any(isinstance(t, torch.nn.BCEWithLogitsLoss) for t in loss_fn_choice.transforms))
        self.assertTrue(any(hasattr(t, "margin") for t in loss_fn_choice.transforms))

    def test_fail_fast_on_unknown_loss(self):
        """Test that requesting an unknown loss raises an error."""
        config = [{"mode": "single", "instances": {"non_existent_loss": {}}}]

        with self.assertRaises(ValueError, msg="Should fail on unknown loss type"):
            self.factory.build(config)

    def test_fail_fast_on_invalid_parameter(self):
        """Test that passing an invalid parameter to a loss function raises an error."""
        config = [{"mode": "single", "instances": {"cross_entropy_loss": {"invalid_param": 123}}}]

        with self.assertRaises(TypeError, msg="Should fail on unexpected keyword argument"):
            self.factory.build(config)


if __name__ == "__main__":
    unittest.main()
