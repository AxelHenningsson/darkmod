import unittest
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
import numpy as np

from darkmod.transforms import HighPrecisionRotation

class TestHighPrecisionRotation(unittest.TestCase):

    def setUp(self):
        self.debug=False

    def test_mul(self):
        R1 = Rotation.random()
        R2 = Rotation.random()

        HR1 = HighPrecisionRotation(R1)
        HR2 = HighPrecisionRotation(R2)

        R = R1 * R2
        HR = HR1 * HR2

        np.testing.assert_allclose( HR.as_matrix(), R.as_matrix() )
    
        R = R2 * R1
        HR = HR2 * HR1

        np.testing.assert_allclose( HR.as_matrix(), R.as_matrix() )

        R3 = Rotation.random()
        R = R2 * R1 * R3
        HR = HR2 * HR1 * HighPrecisionRotation(R3)

        np.testing.assert_allclose( HR.as_matrix(), R.as_matrix() )
    
    def test_transpose(self):
        R = Rotation.random()
        HR = HighPrecisionRotation(R)

        np.testing.assert_allclose( HR.T.as_matrix(), R.as_matrix().T )
        np.testing.assert_allclose( HR.T.as_matrix(), R.inv().as_matrix() )

    def test_vecmul(self):
        R = Rotation.random()
        HR = HighPrecisionRotation(R)
        vectors = np.random.rand(3, 12)
        np.testing.assert_allclose( HR @ vectors, R.as_matrix() @ vectors )
        np.testing.assert_allclose( HR @ vectors, R.apply( vectors.T ).T )

if __name__ == '__main__':
    unittest.main()