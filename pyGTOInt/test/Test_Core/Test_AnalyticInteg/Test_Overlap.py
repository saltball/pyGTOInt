import unittest

import numpy as np

from pyGTOInt.core.AnalyticInteg.overlap import *
from pyGTOInt.test.test_main import CALCU_PERCISION

a_array = np.array([0, 0, 0])
b_array = np.array([0, 0, 0])


class TestOverLap(unittest.TestCase):

    def test_self_normalized(self):
        a = 1
        b = 1
        self.assertAlmostEqual(SxyzDefold(a, b, a_array, b_array, 0, 0, 0, 0, 0, 0), 1, CALCU_PERCISION)
        self.assertAlmostEqual(SxyzDefold(a, b, a_array, b_array, 1, 0, 0, 1, 0, 0), 1, CALCU_PERCISION)
        self.assertAlmostEqual(SxyzDefold(a, b, a_array, b_array, 1, 1, 0, 1, 1, 0), 1, CALCU_PERCISION)
        self.assertAlmostEqual(SxyzDefold(a, b, a_array, b_array, 2, 0, 0, 2, 0, 0), 1, CALCU_PERCISION)
        self.assertAlmostEqual(SxyzDefold(a, b, a_array, b_array, 3, 0, 0, 3, 0, 0), 1, CALCU_PERCISION)
        self.assertAlmostEqual(SxyzDefold(a, b, a_array, b_array, 2, 1, 0, 2, 1, 0), 1, CALCU_PERCISION)
        self.assertAlmostEqual(SxyzDefold(a, b, a_array, b_array, 1, 1, 1, 1, 1, 1), 1, CALCU_PERCISION)

    def test_self_orthogonal(self):
        a = 1
        b = 1
        self.assertAlmostEqual(SxyzDefold(a, b, a_array, b_array, 0, 0, 0, 0, 1, 0), 0, CALCU_PERCISION)
        self.assertAlmostEqual(SxyzDefold(a, b, a_array, b_array, 1, 0, 0, 0, 1, 0), 0, CALCU_PERCISION)
        self.assertAlmostEqual(SxyzDefold(a, b, a_array, b_array, 0, 1, 0, 0, 0, 1), 0, CALCU_PERCISION)
        self.assertAlmostEqual(SxyzDefold(a, b, a_array, b_array, 0, 0, 1, 0, 1, 0), 0, CALCU_PERCISION)


if __name__ == '__main__':
    unittest.main()
