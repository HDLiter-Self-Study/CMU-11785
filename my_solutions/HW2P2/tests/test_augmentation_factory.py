import json
import unittest
from pathlib import Path
from unittest.mock import patch, mock_open

import torch
from torchvision.transforms import v2

from src.pipelines.factories import AugmentationFactory
from src.data.transforms import ImpulseNoise, GridMask


class TestAugmentationFactory(unittest.TestCase):

    def setUp(self):
        """Set up a factory instance for all tests."""
        self.factory = AugmentationFactory()
        self.stats_data = '{"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}'
        # Mock the STATS_FILE path in the factory instance
        self.factory.STATS_FILE = Path("mock_stats.json")

    @patch("pathlib.Path.read_text")
    @patch("pathlib.Path.is_file")
    def test_build_successful_with_wrapping(self, mock_is_file, mock_read_text):
        """
        Tests if the factory can successfully build a pipeline with intelligent RandomApply wrapping.

        This test verifies:
        1. Configs with 'p' or 'prob' are wrapped with RandomApply
        2. Configs without probability are used directly
        3. Final pipeline includes all transforms plus ToDtype and Normalize
        """
        mock_is_file.return_value = True
        mock_read_text.return_value = self.stats_data

        configs = [
            {"mode": "single", "instances": {"trivial_augment_wide": {}}},  # No probability - should not be wrapped
            {
                "mode": "single",
                "instances": {"gaussian_noise": {"sigma": 0.1, "p": 0.5}},
            },  # With 'p' - should be wrapped
            {
                "mode": "single",
                "instances": {"impulse_noise": {"amount": 0.03, "prob": 0.7}},
            },  # With 'prob' - should be wrapped
            {
                "mode": "single",
                "instances": {"grid_mask": {"d_ratio_range": (0.1, 0.2)}},
            },  # No probability - should not be wrapped
        ]

        pipeline = self.factory.build(configs, return_base_transform=False)

        # Basic pipeline structure validation
        self.assertIsInstance(pipeline, v2.Compose)
        self.assertEqual(len(pipeline.transforms), len(configs) + 2)  # +2 for ToDtype and Normalize

        # Check specific transform types and wrapping
        self.assertIsInstance(pipeline.transforms[0], v2.TrivialAugmentWide)  # Direct transform
        self.assertIsInstance(pipeline.transforms[1], v2.RandomApply)  # Wrapped gaussian_noise
        self.assertIsInstance(pipeline.transforms[2], v2.RandomApply)  # Wrapped impulse_noise
        self.assertIsInstance(pipeline.transforms[3], GridMask)  # Direct transform
        self.assertIsInstance(pipeline.transforms[4], v2.ToDtype)  # Auto-added
        self.assertIsInstance(pipeline.transforms[5], v2.Normalize)  # Auto-added

        # Verify RandomApply configurations
        wrapped_gaussian = pipeline.transforms[1]
        self.assertEqual(wrapped_gaussian.p, 0.5)
        self.assertIsInstance(wrapped_gaussian.transforms[0], v2.GaussianNoise)

        wrapped_impulse = pipeline.transforms[2]
        self.assertEqual(wrapped_impulse.p, 0.7)
        self.assertIsInstance(wrapped_impulse.transforms[0], ImpulseNoise)

        # Check normalization values
        normalize_transform = pipeline.transforms[5]
        self.assertEqual(normalize_transform.mean, [0.485, 0.456, 0.406])
        self.assertEqual(normalize_transform.std, [0.229, 0.224, 0.225])

    @patch("pathlib.Path.read_text")
    @patch("pathlib.Path.is_file")
    def test_build_with_edge_case_probabilities(self, mock_is_file, mock_read_text):
        """
        Tests handling of edge case probability values (0.0, 1.0, invalid values).
        """
        mock_is_file.return_value = True
        mock_read_text.return_value = self.stats_data

        configs = [
            {"mode": "single", "instances": {"gaussian_noise": {"sigma": 0.1, "p": 0.0}}},  # p=0.0 should still wrap
            {"mode": "single", "instances": {"impulse_noise": {"amount": 0.03, "p": 1.0}}},  # p=1.0 should still wrap
            {
                "mode": "single",
                "instances": {"grid_mask": {"d_ratio_range": (0.1, 0.2), "p": -0.5}},
            },  # Invalid p - should not wrap
            {"mode": "single", "instances": {"trivial_augment_wide": {"prob": 1.5}}},  # Invalid prob - should not wrap
        ]

        pipeline = self.factory.build(configs, return_base_transform=False)

        # Check wrapping behavior
        self.assertIsInstance(pipeline.transforms[0], v2.RandomApply)  # p=0.0 wrapped
        self.assertIsInstance(pipeline.transforms[1], v2.RandomApply)  # p=1.0 wrapped
        self.assertIsInstance(pipeline.transforms[2], GridMask)  # Invalid p - not wrapped
        self.assertIsInstance(pipeline.transforms[3], v2.TrivialAugmentWide)  # Invalid prob - not wrapped

        # Verify probability values
        self.assertEqual(pipeline.transforms[0].p, 0.0)
        self.assertEqual(pipeline.transforms[1].p, 1.0)

    @patch("pathlib.Path.is_file", return_value=False)
    def test_build_fails_if_stats_file_missing(self, mock_is_file):
        """
        Tests if the factory correctly raises FileNotFoundError when stats are missing.
        """
        with self.assertRaises(FileNotFoundError) as context:
            self.factory.build([], return_base_transform=False)

        # Verify error message contains helpful information
        self.assertIn("Dataset statistics file not found", str(context.exception))
        self.assertIn("calculate_dataset_stats.py", str(context.exception))

    def test_build_empty_configs(self):
        """
        Tests building a pipeline with an empty config list.
        """
        with patch("pathlib.Path.is_file", return_value=True), patch(
            "pathlib.Path.read_text", return_value=self.stats_data
        ):

            pipeline = self.factory.build([], return_base_transform=False)

            # Should only contain the mandatory final transforms
            self.assertIsInstance(pipeline, v2.Compose)
            self.assertEqual(len(pipeline.transforms), 2)  # Only ToDtype and Normalize
            self.assertIsInstance(pipeline.transforms[0], v2.ToDtype)
            self.assertIsInstance(pipeline.transforms[1], v2.Normalize)

    def test_create_torchvision_component(self):
        """Tests direct creation of a standard torchvision component."""
        config = {"gaussian_blur": {"kernel_size": 3}}
        component = self.factory.create(config)
        self.assertIsInstance(component, v2.GaussianBlur)

    def test_create_custom_noise_component(self):
        """Tests direct creation of custom noise components."""
        config = {"impulse_noise": {"amount": 0.1}}
        component = self.factory.create(config)
        self.assertIsInstance(component, ImpulseNoise)
        self.assertEqual(component.amount, 0.1)

    def test_create_custom_spatial_component(self):
        """Tests direct creation of custom spatial components."""
        config = {"grid_mask": {"ratio": 0.8}}
        component = self.factory.create(config)
        self.assertIsInstance(component, GridMask)
        self.assertEqual(component.ratio, 0.8)

    def test_create_official_gaussian_noise(self):
        """Tests creation of the official v2.GaussianNoise with correct parameters."""
        config = {"gaussian_noise": {"sigma": 0.05, "mean": 0.1}}
        component = self.factory.create(config)
        self.assertIsInstance(component, v2.GaussianNoise)
        # Note: v2.GaussianNoise stores sigma internally, but we can't easily access it
        # This test mainly verifies that it can be created without errors

    @patch("pathlib.Path.read_text")
    @patch("pathlib.Path.is_file")
    def test_parameter_isolation_between_configs(self, mock_is_file, mock_read_text):
        """
        Tests that probability parameters are correctly isolated between different configs.

        This ensures that popping 'p' from one config doesn't affect others.
        """
        mock_is_file.return_value = True
        mock_read_text.return_value = self.stats_data

        # Create configs with same structure but different probability values
        configs = [
            {"mode": "single", "instances": {"impulse_noise": {"amount": 0.1, "p": 0.3}}},
            {"mode": "single", "instances": {"impulse_noise": {"amount": 0.2, "p": 0.7}}},
        ]

        pipeline = self.factory.build(configs, return_base_transform=False)

        # Both should be wrapped but with different probabilities
        self.assertIsInstance(pipeline.transforms[0], v2.RandomApply)
        self.assertIsInstance(pipeline.transforms[1], v2.RandomApply)

        self.assertEqual(pipeline.transforms[0].p, 0.3)
        self.assertEqual(pipeline.transforms[1].p, 0.7)

        # Verify the underlying transforms have correct parameters
        first_transform = pipeline.transforms[0].transforms[0]
        second_transform = pipeline.transforms[1].transforms[0]

        self.assertIsInstance(first_transform, ImpulseNoise)
        self.assertIsInstance(second_transform, ImpulseNoise)
        self.assertEqual(first_transform.amount, 0.1)
        self.assertEqual(second_transform.amount, 0.2)

    def test_malformed_config_handling(self):
        """Tests handling of malformed configurations."""
        # Test empty config
        with self.assertRaises(ValueError):
            self.factory.create({})

        # Test config with multiple keys (should raise ValueError)
        config = {"gaussian_noise": {"sigma": 0.1}, "extra_key": "ignored"}
        with self.assertRaises(ValueError):
            self.factory.create(config)

    @patch("pathlib.Path.read_text")
    @patch("pathlib.Path.is_file")
    def test_build_with_mode_format_single(self, mock_is_file, mock_read_text):
        """Tests building with new mode format - single mode."""
        mock_is_file.return_value = True
        mock_read_text.return_value = self.stats_data

        configs = [{"mode": "single", "instances": {"gaussian_noise": {"sigma": 0.1}}}]

        pipeline = self.factory.build(configs)

        # Should have gaussian_noise + ToDtype + Normalize
        self.assertIsInstance(pipeline, v2.Compose)
        self.assertEqual(len(pipeline.transforms), 3)
        self.assertIsInstance(pipeline.transforms[0], v2.GaussianNoise)
        self.assertIsInstance(pipeline.transforms[1], v2.ToDtype)
        self.assertIsInstance(pipeline.transforms[2], v2.Normalize)

    @patch("pathlib.Path.read_text")
    @patch("pathlib.Path.is_file")
    def test_build_with_mode_format_random_choice(self, mock_is_file, mock_read_text):
        """Tests building with new mode format - random_choice mode."""
        mock_is_file.return_value = True
        mock_read_text.return_value = self.stats_data

        configs = [
            {
                "mode": "random_choice",
                "instances": {"gaussian_noise": {"sigma": 0.1}, "gaussian_blur": {"kernel_size": 3}},
            }
        ]

        pipeline = self.factory.build(configs, return_base_transform=False)

        # Should have 1 RandomChoice (containing 2 transforms) + ToDtype + Normalize
        self.assertIsInstance(pipeline, v2.Compose)
        self.assertEqual(len(pipeline.transforms), 3)  # 1 RandomChoice + ToDtype + Normalize

        # Check the RandomChoice contains the correct transforms
        random_choice = pipeline.transforms[0]
        self.assertIsInstance(random_choice, v2.RandomChoice)
        self.assertEqual(len(random_choice.transforms), 2)
        self.assertIsInstance(random_choice.transforms[0], v2.GaussianNoise)
        self.assertIsInstance(random_choice.transforms[1], v2.GaussianBlur)

        # Check final transforms
        self.assertIsInstance(pipeline.transforms[1], v2.ToDtype)
        self.assertIsInstance(pipeline.transforms[2], v2.Normalize)

    @patch("pathlib.Path.read_text")
    @patch("pathlib.Path.is_file")
    def test_build_multiple_single_mode_configs(self, mock_is_file, mock_read_text):
        """Tests building with multiple single mode configurations."""
        mock_is_file.return_value = True
        mock_read_text.return_value = self.stats_data

        configs = [
            # All using proper mode format
            {"mode": "single", "instances": {"trivial_augment_wide": {}}},
            {"mode": "single", "instances": {"gaussian_noise": {"sigma": 0.1}}},
        ]

        pipeline = self.factory.build(configs, return_base_transform=False)

        # Should have trivial_augment_wide + gaussian_noise + ToDtype + Normalize
        self.assertIsInstance(pipeline, v2.Compose)
        self.assertEqual(len(pipeline.transforms), 4)
        self.assertIsInstance(pipeline.transforms[0], v2.TrivialAugmentWide)
        self.assertIsInstance(pipeline.transforms[1], v2.GaussianNoise)
        self.assertIsInstance(pipeline.transforms[2], v2.ToDtype)
        self.assertIsInstance(pipeline.transforms[3], v2.Normalize)

    @patch("pathlib.Path.read_text")
    @patch("pathlib.Path.is_file")
    def test_build_mode_format_invalid_single_multiple_instances(self, mock_is_file, mock_read_text):
        """Tests that single mode with multiple instances raises ValueError."""
        mock_is_file.return_value = True
        mock_read_text.return_value = self.stats_data

        configs = [
            {"mode": "single", "instances": {"gaussian_noise": {"sigma": 0.1}, "gaussian_blur": {"kernel_size": 3}}}
        ]

        with self.assertRaises(ValueError) as context:
            self.factory.build(configs, return_base_transform=False)
        self.assertIn("Single mode requires exactly one instance", str(context.exception))


if __name__ == "__main__":
    unittest.main()
