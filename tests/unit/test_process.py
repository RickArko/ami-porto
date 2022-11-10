import unittest

from tests.conftest import train
from src.process import load_train_test, process_data


class Test(unittest.TestCase):
    def test_basic(self):
        """Test basic fact."""
        self.assertEqual(1, 1)
        self.assertEqual(True, True)
        
    def test_sample_data(self):
        """Test basic fact."""
        self.assertTupleEqual(train.shape, (20, 59))
        
    def test_process_data(self):
        train, test = load_train_test()
        self.assertTupleEqual(train.shape, (595_212, 59))
        self.assertTupleEqual(test.shape, (892_816, 58))

        X_train = process_data(train, "target")
        X_test = process_data(test, "target")
        
        self.assertNotEquals(X_train.shape[0], X_test.shape[0])
        self.assertListEqual(X_train.columns.tolist(), X_test.columns.tolist())
