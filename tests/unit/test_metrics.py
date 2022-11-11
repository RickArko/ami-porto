import unittest
import numpy as np
from src.metrics import Gini


class Test(unittest.TestCase):
    def test_basic(self):
        """Test basic fact."""
        self.assertEqual(1, 1)
        self.assertEqual(True, True)
        
    def test_gini(self):
        n = 10_000  # samples
        perfect = Gini(np.ones(n), np.ones(n))
        self.assertEqual(perfect, 1)

        r1 = np.random.choice([0, 1], size=(n,), p=[.5, .5])
        r2 = np.random.choice([0, 1], size=(n,), p=[.5, .5])

        random = Gini(r1, r2)
        self.assertGreaterEqual(random, -.1) # TODO: verify correct implementation
        self.assertLessEqual(random, 1)
        
        return
