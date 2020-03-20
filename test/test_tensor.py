import unittest

import torch
from torch_tensor_eq.tensor import Tensor


class TestTensor(unittest.TestCase):
    def test___eq__(self):
        """Tests basic __eq__ functionality."""
        a = Tensor(torch.ones(5))
        b = Tensor(torch.ones(5))
        c = Tensor(torch.zeros(5))
        self.assertEqual(a, b)
        self.assertTrue(a == b)

        self.assertNotEqual(a, c)
        self.assertFalse(a == c)

    def test_recursive(self):
        """Tests that Tensors can be compared while in collection objects."""
        a = [Tensor(torch.ones(5)), Tensor(torch.zeros(3))]
        b = [Tensor(torch.ones(5)), Tensor(torch.zeros(3))]
        self.assertEqual(a, b)

        c = {
            17: Tensor(torch.ones((5, 5))),
            "Hi mom": Tensor(torch.zeros((4, 8))),
        }
        d = {
            17: Tensor(torch.ones((5, 5))),
            "Hi mom": Tensor(torch.zeros((4, 8))),
        }
        self.assertEqual(c, d)

    def test_not_break_torch(self):
        """Tests that nothing we've done breaks the default torch implementations."""
        length = 5
        a = torch.zeros(length)
        b = torch.zeros(length)
        self.assertEqual(len(a == b), length)
        self.assertTrue(torch.all(a == b))

        c = Tensor(torch.ones(5))
        # If Tensor is either argument, it uses the equality method that returns bool.
        self.assertNotEqual(c, a)
        self.assertNotEqual(a, c)


if __name__ == "__main__":
    unittest.main()
