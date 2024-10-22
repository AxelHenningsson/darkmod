import unittest

import matplotlib.pyplot as plt
import numpy as np

from darkmod.resolution import DualKentGauss
from darkmod import laue


class TestCompundDualKentGauss(unittest.TestCase):

    def setUp(self):
        self.DEBUG = False

        # Q and gamma_C corresponds to:
        # hkl = 002 and cubic AL, a = 4.0493 with U=I
        self.Q = np.array([-5.44137060e-01 ,-1.90025011e-16 , 3.05526733e+00])
        lambda_0 = 0.71
        energy_0 = laue.angstrom_to_keV(lambda_0)
        sigma_e = 1.4*1e-4

        # Beam divergence params
        gamma_N = np.eye(3, 3)
        desired_FWHM_N = 0.53*1e-3
        kappa_N = np.log(2)/(1-np.cos((desired_FWHM_N)/2.))
        beta_N  = 0

        # Beam wavelength params
        epsilon = np.random.normal(0, sigma_e, size=(20000,))
        random_energy = energy_0 + epsilon*energy_0
        sigma_lambda = laue.keV_to_angstrom(random_energy).std()
        mu_lambda = lambda_0

        # CRL acceptance params
        gamma_C = np.array([[ 0.93851251,  0. ,        -0.34524524],
                            [ 0.     ,     1. ,         0.        ],
                            [ 0.34524524,  0. ,        0.93851251]])
        desired_FWHM_C = 0.731*1e-3
        kappa_C = np.log(2)/(1-np.cos((desired_FWHM_C)/2.))
        beta_C  = 0

        self.res = DualKentGauss(
                            gamma_C,
                            kappa_C,
                            beta_C,
                            gamma_N,
                            kappa_N,
                            beta_N,
                            mu_lambda,
                            sigma_lambda,
                            )

    def test_compile(self):
        # Test that the compiled resolution function behanves as expected.
        self.res.compile(self.Q,
                         resolution=(8*1e-4, 8*1e-4, 8*1e-4),
                         ranges=(1, 1, 1),
                         number_of_samples=50000)

        # test that the mode is at the mean
        m,n,o = self.res.p_Q.shape
        i,j,k = np.unravel_index(np.argmax(self.res.p_Q), self.res.p_Q.shape)
        self.assertEqual(i, m//2)
        self.assertEqual(j, n//2)
        self.assertEqual(k, o//2)

        # test that the distirbution is thin in x (q-system)
        self.assertGreater(n, m)
        self.assertGreater(o, m)

        # test that the probability is greater than 0
        self.assertGreater(self.res.p_Q.min(), 0)

        # test that the probability sums to unity
        dv = np.prod((8*1e-4, 8*1e-4, 8*1e-4))
        self.assertAlmostEqual(self.res.p_Q.sum()*dv, 1)

        # test that the interpolation is similar to the estimation
        residual = np.abs(self.res(self.Q.reshape(3,1)) - self.res.p_Q.max())
        residual = residual / self.res.p_Q.max()
        self.assertLess(residual, 0.025) # tolerate 2.5% error due to grid displacement

        # test that the interpolation is vectorized correctly
        x, y, z = self.res._integration_points
        x = x.reshape( (m, n, o) )[1:-1, 1:-1, 1:-1].flatten()
        y = y.reshape( (m, n, o) )[1:-1, 1:-1, 1:-1].flatten()
        z = z.reshape( (m, n, o) )[1:-1, 1:-1, 1:-1].flatten()
        interp = self.res(np.array([x, y, z]))
        expected_value = self.res.p_Q[1:-1, 1:-1, 1:-1].flatten()
        np.testing.assert_allclose(interp, expected_value)

        if self.DEBUG:
            a, b, c = self.cartesian_slices(self.res.p_Q)
            d, e, f = self.cartesian_slices(self.res.std_p_Q)
            for field in (a, b, c):
                self._plot(field, cmap='viridis')
            for field in (d, e, f):
                self._plot(field, cmap='magma')
            for field in (d/a, e/b, f/c):
                self._plot(field, cmap='jet')
            plt.show()

    def test_sample(self):
        sample = self.res.sample(number_of_samples=10000)
        qmean = np.mean(sample, axis=1)
        np.testing.assert_almost_equal(qmean, self.Q, decimal=3)

    def _plot(self, field, cmap):
        plt.figure()
        plt.imshow(field, cmap=cmap)
        plt.colorbar()

    def cartesian_slices(self, field):
        a = field[field.shape[0]//2,:,:]
        b = field[:, field.shape[1]//2,:]
        c = field[:, :, field.shape[2]//2]
        return a, b, c

if __name__ == '__main__':
    unittest.main()