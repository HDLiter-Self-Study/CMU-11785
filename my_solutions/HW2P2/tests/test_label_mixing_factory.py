import unittest
from unittest.mock import patch, MagicMock

import torch
from torchvision.transforms import v2

from src.pipelines.factories import LabelMixingFactory


class TestLabelMixingFactory(unittest.TestCase):

    def setUp(self):
        """Set up a factory instance for all tests."""
        self.factory = LabelMixingFactory()
        self.num_classes = 10

    def test_create_mixup(self):
        """Tests direct creation of MixUp strategy."""
        config = {"mixup": {"alpha": 1.0}}
        strategy = self.factory.create(config, num_classes=self.num_classes)
        self.assertIsInstance(strategy, v2.MixUp)

    def test_create_cutmix(self):
        """Tests direct creation of CutMix strategy."""
        config = {"cutmix": {"alpha": 1.0}}
        strategy = self.factory.create(config, num_classes=self.num_classes)
        self.assertIsInstance(strategy, v2.CutMix)

    def test_create_fmix(self):
        """Tests direct creation of FMix strategy using custom implementation."""
        config = {"fmix": {"alpha": 1.0, "decay_power": 3.0}}
        strategy = self.factory.create(config, num_classes=self.num_classes)
        # Should create our custom FMix implementation
        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.num_classes, self.num_classes)

    def test_build_single_mode(self):
        """Tests building a single strategy in single mode."""
        configs = [{"mode": "single", "instances": {"mixup": {"alpha": 0.5}}}]
        strategy = self.factory.build(configs, num_classes=self.num_classes)
        self.assertIsInstance(strategy, v2.MixUp)

    def test_build_random_choice_mode(self):
        """Tests building multiple strategies in random_choice mode."""
        configs = [{"mode": "random_choice", "instances": {"mixup": {"alpha": 0.5}, "cutmix": {"alpha": 1.0}}}]
        strategy = self.factory.build(configs, num_classes=self.num_classes)
        self.assertIsInstance(strategy, v2.RandomChoice)

    def test_build_fmix_single_mode(self):
        """Tests building FMix strategy in single mode."""
        configs = [{"mode": "single", "instances": {"fmix": {"alpha": 0.6, "decay_power": 1.0, "max_soft": 0.18}}}]
        strategy = self.factory.build(configs, num_classes=self.num_classes)
        self.assertIsNotNone(strategy)
        # Verify FMix specific parameters
        self.assertEqual(strategy.alpha, 0.6)
        self.assertEqual(strategy.decay_power, 1.0)
        self.assertEqual(strategy.max_soft, 0.18)

    def test_build_single_group_random_choice_multiple(self):
        """Tests single group with random_choice of multiple strategies."""
        configs = [{"mode": "random_choice", "instances": {"mixup": {"alpha": 0.5}, "cutmix": {"alpha": 1.0}}}]
        strategy = self.factory.build(configs, num_classes=self.num_classes)
        self.assertIsInstance(strategy, v2.RandomChoice)

    def test_build_empty_configs(self):
        """Tests building with empty configs returns None."""
        strategy = self.factory.build([], num_classes=self.num_classes)
        self.assertIsNone(strategy)

    def test_build_missing_num_classes(self):
        """Building without num_classes should succeed; runtime requires one-hot labels if omitted."""
        configs = [{"mode": "single", "instances": {"mixup": {"alpha": 0.5}}}]
        strategy = self.factory.build(configs)
        self.assertIsNotNone(strategy)

    def test_build_invalid_single_mode_multiple_instances(self):
        """Tests that single mode with multiple instances raises ValueError."""
        configs = [{"mode": "single", "instances": {"mixup": {"alpha": 0.5}, "cutmix": {"alpha": 1.0}}}]
        with self.assertRaises(ValueError) as context:
            self.factory.build(configs, num_classes=self.num_classes)
        self.assertIn("Single mode requires exactly one instance", str(context.exception))

    def test_build_unsupported_mode(self):
        """Tests that unsupported mode raises ValueError."""
        configs = [{"mode": "invalid_mode", "instances": {"mixup": {"alpha": 0.5}}}]
        with self.assertRaises(ValueError) as context:
            self.factory.build(configs, num_classes=self.num_classes)
        self.assertIn("Unsupported mode", str(context.exception))

    def test_create_requires_num_classes(self):
        """Tests that create method passes num_classes to the component."""
        config = {"mixup": {"alpha": 1.0}}
        strategy = self.factory.create(config, num_classes=self.num_classes)
        # Verify that the created MixUp has the correct num_classes
        self.assertEqual(strategy.num_classes, self.num_classes)

    def test_functional_mixup_operation(self):
        """Tests that created MixUp strategy actually works on sample data."""
        config = {"mixup": {"alpha": 1.0}}
        strategy = self.factory.create(config, num_classes=self.num_classes)

        # Create sample data
        batch_size = 4
        images = torch.randn(batch_size, 3, 32, 32)
        labels = torch.randint(0, self.num_classes, (batch_size,))

        # Apply strategy
        mixed_images, mixed_labels = strategy(images, labels)

        # Verify outputs
        self.assertEqual(mixed_images.shape, images.shape)
        self.assertEqual(mixed_labels.shape, (batch_size, self.num_classes))  # One-hot encoded

    def test_functional_fmix_operation(self):
        """Tests that created FMix strategy actually works on sample data."""
        config = {"fmix": {"alpha": 1.0, "decay_power": 3.0}}
        strategy = self.factory.create(config, num_classes=self.num_classes)

        # Create sample data
        batch_size = 4
        images = torch.randn(batch_size, 3, 32, 32)
        labels = torch.randint(0, self.num_classes, (batch_size,))

        # Apply strategy
        mixed_images, mixed_labels = strategy(images, labels)

        # Verify outputs
        self.assertEqual(mixed_images.shape, images.shape)
        self.assertEqual(mixed_labels.shape, (batch_size, self.num_classes))  # One-hot encoded


if __name__ == "__main__":
    unittest.main()
