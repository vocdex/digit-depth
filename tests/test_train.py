import os
import unittest
import torch
from digit_depth.train import MLP, Color2NormalDataset

base_path = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))


class Train(unittest.TestCase):
    def test_shape(self):
        model = MLP()
        x = torch.randn(1, 5)
        y = model(x)
        self.assertEqual(torch.Size([1, 3]), y.size())

    def test_dataset(self):
        dataset = Color2NormalDataset(f'{base_path}/datasets/train_test_split/train.csv')
        x, y = dataset[0]
        self.assertEqual(torch.Size([5]), x.size())
        self.assertEqual(torch.Size([3]), y.size())
        self.assertLessEqual(x.max(), 1)
        self.assertGreaterEqual(x.min(), 0)


if __name__ == '__main__':
    unittest.main()
