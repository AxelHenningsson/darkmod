import unittest

import matplotlib.pyplot as plt
import numpy as np

from darkmod.crl import CompundRefractiveLens


class TestCompundRefractiveLens(unittest.TestCase):

    def setUp(self):
        self.number_of_lenses = 50 # N
        self.lens_space = 2 * 1e-3 # T
        self.lens_radius = 50 * 1e-6 # R
        self.refractive_decrement = 1.65 * 1e-6  # delta
        self.magnification = 10
        self.crl = CompundRefractiveLens(self.number_of_lenses,
                                        self.lens_space,
                                        self.lens_radius,
                                        self.refractive_decrement,
                                        self.magnification)

    def test_magnification(self):
        np.testing.assert_almost_equal(-self.crl.K[0, 0], self.magnification)

    def test_imaging_condition(self):
        np.testing.assert_almost_equal( self.crl.K[0, 1], 0 )

    def test_ray_matrix(self):
        focal = self.lens_radius / ( 2 * self.refractive_decrement)
        M = self._get_MN_numerical(self.lens_space, focal, self.number_of_lenses)
        np.testing.assert_array_almost_equal(self.crl.M_N, M)

    def _get_MN_numerical(self, distance, focal_length, number_of_lenses):
        """
        Based on Ray matrix transfer theory.

        c.f Simons 2016:
        Simulating and optimizing compound refractive lens-based X-ray microscopes
        J. Synchrotron Rad. (2017). 24, 392–401
        https://doi.org/10.1107/S160057751602049X J.

        """
        Mf = np.array([[ 1.,   distance/2.],
                    [ 0,        1.     ]]) # free space propagate

        Ml = np.array([[1.,               0 ],
                    [-1./focal_length, 1.]]) # thin lens propagate

        M = Mf @ Ml @ Mf

        Mnumerical = M.copy()
        for i in range(number_of_lenses-1):
            Mnumerical = Mnumerical @ M

        return Mnumerical


if __name__ == '__main__':
    unittest.main()