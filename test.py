import unittest
import numpy as np
from main import calculate_dev


class TestCalculations(unittest.TestCase):

    def test_deviation(self):
        deviation = calculate_dev(np.array([8, 2]), np.array([7, 5]))
        self.assertEqual(deviation, 1.4142135623730951, "The deviation is wrong")



if __name__ == '__main__':
    unittest.main()