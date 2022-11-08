import unittest

from tests.conftest import train


class Test(unittest.TestCase):
    def test_basic(self):
        """Test basic fact."""
        self.assertEqual(1, 1)
        self.assertEqual(True, True)
        
    def test_sample_data(self):
        """Test basic fact."""
        self.assertTupleEqual(train.shape, (20, 59))
