import unittest

import matplotlib.pyplot as plt
import numpy as np

from darkmod.crl import CompundRefractiveLens


class TestCompundRefractiveLens(unittest.TestCase):

    def setUp(self):
        self.DEBUG = True
        self.number_of_lenses = 50 # N
        self.lens_space = 2000 # T
        self.lens_radius = 50 # R
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

    def test_imaging_system(self):
        self.crl.theta = np.radians(10)
        self.crl.eta = np.radians(-20)
        I = self.crl.imaging_system
        np.testing.assert_allclose(I[:,0], self.crl.optical_axis)
        self.assertGreater( I[0,0], I[1,0] )
        self.assertGreater( I[0,0], I[2,0] )
        self.assertGreater( I[0,0], 0 )
        self.assertGreater( I[1,0], 0 )
        self.assertGreater( I[2,0], 0 )

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

    def test_get_angular_shifts(self):

        theta=np.radians(10)
        self.crl.goto(theta, eta=0)

        xi, yi, zi = self.crl.imaging_system.T

        expected_angle = np.radians(30) 
        L = self.crl.d1 * np.tan( expected_angle )

        horizontal, vertical = self.crl.get_angular_shifts( L*yi.reshape(3,1) )
        print('yi', horizontal, vertical, -expected_angle)
        self.assertAlmostEqual(horizontal[0], -expected_angle)
        self.assertAlmostEqual(vertical[0], 0)

        horizontal, vertical = self.crl.get_angular_shifts( L*xi.reshape(3,1) )
        print('xi', horizontal, vertical, expected_angle)
        self.assertAlmostEqual(horizontal[0], 0)
        self.assertAlmostEqual(vertical[0], 0)

        horizontal, vertical = self.crl.get_angular_shifts( L*zi.reshape(3,1) )
        print('zi', horizontal, vertical, expected_angle)
        self.assertAlmostEqual(horizontal[0], 0)
        self.assertAlmostEqual(vertical[0], expected_angle)

        if self.DEBUG:
            self.crl.goto(theta=np.radians(20), eta=0)
            x_lab = np.random.rand(3, 5000)-0.5
            shifts = self.crl.get_angular_shifts(x_lab)
            x_im = self.crl.imaging_system.T @ x_lab
            fig, ax = plt.subplots(1,2,figsize=(12,6))
            ax[0].scatter(x_im[1], x_im[2], c=shifts[0])
            ax[0].set_title('Horizontal shifts')
            ax[1].scatter(x_im[1], x_im[2], c=shifts[1])
            ax[1].set_title('Vertical shifts')
            ax[0].grid(True)
            ax[1].grid(True)
            for a in ax.flatten():
                a.set_xlabel('y')
                a.set_ylabel('z')

            plt.show()

if __name__ == '__main__':
    unittest.main()