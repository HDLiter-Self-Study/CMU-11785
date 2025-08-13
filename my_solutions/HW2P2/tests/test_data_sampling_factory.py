import unittest

from torch.utils.data import Dataset

from src.pipelines.factories import DataSamplingFactory


class _DummyDataset(Dataset):
    def __init__(self, length: int = 5):
        self._len = length

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        return idx


class TestDataSamplingFactory(unittest.TestCase):
    def setUp(self):
        self.factory = DataSamplingFactory()

    def test_repeated_augmentation_dataset_length(self):
        base = _DummyDataset(length=7)
        wrapper = self.factory.create({"repeated_augmentation": {"base_dataset": base, "num_repeats": 3}})
        self.assertEqual(len(wrapper), 21)

    def test_repeated_augmentation_dataset_index_mapping(self):
        base = _DummyDataset(length=4)
        wrapper = self.factory.create({"repeated_augmentation": {"base_dataset": base, "num_repeats": 2}})
        # indices 0..7 map to base indices 0,0,1,1,2,2,3,3
        mapped = [wrapper[i] for i in range(len(wrapper))]
        self.assertEqual(mapped, [0, 0, 1, 1, 2, 2, 3, 3])

    def test_invalid_repeats_fast_fail(self):
        base = _DummyDataset(length=2)
        with self.assertRaises(RuntimeError) as ctx:
            self.factory.create({"repeated_augmentation": {"base_dataset": base, "num_repeats": 0}})
        self.assertIn("num_repeats", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
