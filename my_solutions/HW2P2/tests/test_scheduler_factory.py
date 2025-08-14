import unittest
import torch
from torch import optim
from src.pipelines.factories import SchedulerFactory


class TestSchedulerFactory(unittest.TestCase):
    def setUp(self):
        """Set up a mock optimizer and factory for each test."""
        self.factory = SchedulerFactory()
        # A mock optimizer is required by all schedulers.
        self.model_params = [torch.nn.Parameter(torch.randn(10, 10))]
        self.optimizer = optim.SGD(self.model_params, lr=0.1)
        # Context parameters that would be injected by the pipeline builder.
        self.total_epochs = 100
        self.steps_per_epoch = 50

    def test_create_single_scheduler(self):
        """Test creating a single, standalone scheduler without warmup."""
        config = [{"mode": "single", "instances": {"cosine_annealing_lr": {"T_max": self.total_epochs}}}]

        scheduler = self.factory.build(
            configs=config,
            optimizer=self.optimizer,
            total_epochs=self.total_epochs,
            steps_per_epoch=self.steps_per_epoch,
        )

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)

    def test_create_warmup_only(self):
        """Test creating a warmup scheduler by itself."""
        config = [{"mode": "single", "instances": {"warmup": {"warmup_epochs": 5, "warmup_start_factor": 0.1}}}]

        scheduler = self.factory.build(
            configs=config,
            optimizer=self.optimizer,
            total_epochs=self.total_epochs,
            steps_per_epoch=self.steps_per_epoch,
        )

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.LinearLR)

    def test_chaining_with_warmup(self):
        """Test that a main scheduler and warmup are chained into a SequentialLR."""
        configs = [
            {"mode": "single", "instances": {"exponential_lr": {"gamma": 0.95}}},
            {"mode": "single", "instances": {"warmup": {"warmup_epochs": 5, "warmup_start_factor": 0.1}}},
        ]

        scheduler = self.factory.build(
            configs=configs,
            optimizer=self.optimizer,
            total_epochs=self.total_epochs,
            steps_per_epoch=self.steps_per_epoch,
        )

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
        # Check that it contains both the warmup and the main scheduler.
        self.assertEqual(len(scheduler._schedulers), 2)
        self.assertIsInstance(scheduler._schedulers[0], torch.optim.lr_scheduler.LinearLR)
        self.assertIsInstance(scheduler._schedulers[1], torch.optim.lr_scheduler.ExponentialLR)
        # Check that the milestone for switching is correct.
        self.assertEqual(scheduler._milestones, [5 * self.steps_per_epoch])

    def test_multi_step_lr_with_warmup_shifting(self):
        """Test that MultiStepLR milestones are correctly shifted when warmup is present."""
        warmup_epochs = 10
        configs = [
            {"mode": "single", "instances": {"multi_step_lr": {"milestones_ratio": "[0.3, 0.8]", "gamma": 0.1}}},
            {"mode": "single", "instances": {"warmup": {"warmup_epochs": warmup_epochs, "warmup_start_factor": 0.01}}},
        ]

        scheduler = self.factory.build(
            configs=configs,
            optimizer=self.optimizer,
            total_epochs=self.total_epochs,
            steps_per_epoch=self.steps_per_epoch,
        )

        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.SequentialLR)
        main_scheduler = scheduler._schedulers[1]
        self.assertIsInstance(main_scheduler, torch.optim.lr_scheduler.MultiStepLR)

        # Original milestones would be [30, 80].
        # After shifting by warmup_epochs, they should be [40, 90].
        expected_milestones = [30 + warmup_epochs, 80 + warmup_epochs]
        self.assertEqual(list(main_scheduler.milestones), expected_milestones)

    def test_handle_warmup_epochs_zero(self):
        """Test that no warmup is created and no chaining happens if warmup_epochs is 0."""
        configs = [
            {"mode": "single", "instances": {"multi_step_lr": {"milestones_ratio": "[0.5, 0.9]", "gamma": 0.1}}},
            {"mode": "single", "instances": {"warmup": {"warmup_epochs": 0, "warmup_start_factor": 0.1}}},
        ]

        scheduler = self.factory.build(
            configs=configs,
            optimizer=self.optimizer,
            total_epochs=self.total_epochs,
            steps_per_epoch=self.steps_per_epoch,
        )

        # Should return the main scheduler directly, not a SequentialLR.
        self.assertIsInstance(scheduler, torch.optim.lr_scheduler.MultiStepLR)
        # The milestones should not be shifted.
        self.assertEqual(list(scheduler.milestones), [50, 90])

    def test_fail_fast_on_missing_params(self):
        """Test that the factory raises KeyError for missing mandatory parameters."""
        # Config for multi_step_lr is missing 'milestones_ratio'.
        configs = [{"mode": "single", "instances": {"multi_step_lr": {"gamma": 0.1}}}]

        with self.assertRaises(KeyError, msg="Should fail fast on missing 'milestones_ratio'"):
            self.factory.build(
                configs=configs,
                optimizer=self.optimizer,
                total_epochs=self.total_epochs,
                steps_per_epoch=self.steps_per_epoch,
            )

        # Config for warmup is missing 'warmup_epochs'.
        configs_warmup = [{"mode": "single", "instances": {"warmup": {"warmup_start_factor": 0.1}}}]

        with self.assertRaises(KeyError, msg="Should fail fast on missing 'warmup_epochs'"):
            self.factory.build(
                configs=configs_warmup,
                optimizer=self.optimizer,
                total_epochs=self.total_epochs,
                steps_per_epoch=self.steps_per_epoch,
            )


if __name__ == "__main__":
    unittest.main()
