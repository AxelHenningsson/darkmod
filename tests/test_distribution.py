import unittest

import matplotlib.pyplot as plt
import numpy as np

from darkmod.distribution import UniformSpherical
from darkmod.distribution import UniformSphericalCone
from darkmod.distribution import Normal
from darkmod.distribution import Kent
from darkmod.distribution import TruncatedNormal


class TestUniformSpherical(unittest.TestCase):

    def setUp(self):
        self.uniform_spherical = UniformSpherical()
        self.number_of_samples = 100
        self.samples = self.uniform_spherical.sample(self.number_of_samples)

    def test_sample_shape(self):
        # Test if the output shape of the sample method is correct
        self.assertEqual(self.samples.shape, (3, self.number_of_samples))

    def test_sample_values(self):
        # Test if the samples lie on the unit sphere
        norms = np.linalg.norm(self.samples, axis=0)
        np.testing.assert_allclose(norms, 1, atol=1e-6)

    def test_spherical_to_cartesian(self):
        # Test if polar to cartesian conversion works correctly
        phi = np.array([0, np.pi/2, np.pi])
        theta = np.array([0, np.pi/2, 0.264])
        expected_output = np.array([
            [0, 0, 1],
            [0, 1, 0],
            [0, 0, -1]
        ]).T
        output = self.uniform_spherical._spherical_to_cartesian(phi, theta)
        np.testing.assert_allclose(output, expected_output, atol=1e-6)

    def test_pdf(self):
        # Test if the PDF returns the correct likelihood
        result = self.uniform_spherical(self.samples)
        expected_result = np.ones((self.number_of_samples,)) / (4 * np.pi)
        np.testing.assert_allclose(result, expected_result, atol=1e-6)


class TestUniformSphericalCone(unittest.TestCase):

    def setUp(self):
        self.phi_max = np.radians(10)
        self.pole = np.array([1, 1, 1]) / np.sqrt(3)
        self.uniform_spherical_cone = UniformSphericalCone(self.phi_max, self.pole)
        self.number_of_samples = 100
        self.samples = self.uniform_spherical_cone.sample(self.number_of_samples)

    def test_sample_shape(self):
        # Test if the output shape of the sample method is correct
        self.assertEqual(self.samples.shape, (3, self.number_of_samples))

    def test_sample_values(self):
        # Test if the samples lie on the unit sphere
        norms = np.linalg.norm(self.samples, axis=0)
        np.testing.assert_allclose(norms, 1, atol=1e-6)

    def test_sample_range(self):
        # Test if the samples lie on the cone segment
        angels = np.arccos(self.samples.T @ self.pole)
        np.testing.assert_array_less(angels, self.phi_max)

    def test_pdf(self):
        # Test if the PDF returns the correct likelihood
        result = self.uniform_spherical_cone(self.samples)
        area = ( 2*np.pi*(1-np.cos(self.phi_max/2.)))
        expected_result = np.ones((self.number_of_samples,)) / area
        np.testing.assert_allclose(result, expected_result, atol=1e-6)

    def test_pdf_support(self):
        # Test if the PDF gives zero outside the cone
        result = self.uniform_spherical_cone( np.array([[1], [0], [1]]) / np.sqrt(2) )
        expected_result = 0
        np.testing.assert_allclose(result, expected_result, atol=1e-6)


class TestNormal(unittest.TestCase):

    def setUp(self):
        self.mu = -1.9236
        self.sigma = 1.15730
        self.normal = Normal(self.mu, self.sigma)
        self.number_of_samples = 20000
        self.samples = self.normal.sample(self.number_of_samples)

    def test_sample_shape(self):
        # Test if the sample method returns the correct shape
        self.assertEqual(self.samples.shape, (self.number_of_samples,))

    def test_sample_mean_std(self):
        # Test if the samples have approximately the correct mean and standard deviation
        self.assertAlmostEqual(np.mean(self.samples), self.mu, places=1)
        self.assertAlmostEqual(np.std(self.samples), self.sigma, places=1)

    def test_call_method_normalised(self):
        # Test the __call__ method when normalise=True and log=False
        x = np.array([0.0, 1.0, 2.0])
        result = self.normal(x, normalise=True, log=False)
        c = 1 / np.sqrt(np.pi*2*self.sigma*self.sigma)
        expected_result = c * np.exp(-0.5 * (x-self.mu)**2 / (self.sigma**2) )
        np.testing.assert_allclose(result, expected_result, atol=1e-6)

    def test_call_method_log(self):
        # Test the __call__ method when normalise=True and log=True
        x = np.array([0.0, 1.0, 2.0])
        result = self.normal(x, normalise=True, log=True)
        c = 1 / np.sqrt(np.pi*2*self.sigma*self.sigma)
        expected_result = np.log(c * np.exp(-0.5 * (x-self.mu)**2 / (self.sigma**2) ))
        np.testing.assert_allclose(result, expected_result, atol=1e-6)

    def test_call_method_non_normalised(self):
        # Test the __call__ method when normalise=False and log=False
        x = np.array([0.0, 1.0, 2.0])
        result = self.normal(x, normalise=False, log=False)
        expected_result = np.exp(-0.5 * (x-self.mu)**2 / (self.sigma**2))
        np.testing.assert_allclose(result, expected_result, atol=1e-6)

    def test_call_method_non_normalised_log(self):
        # Test the __call__ method when normalise=False and log=True
        x = np.array([0.0, 1.0, 2.0])
        result = self.normal(x, normalise=False, log=True)
        expected_result = -0.5 * (x-self.mu)**2 / (self.sigma**2)
        np.testing.assert_allclose(result, expected_result, atol=1e-6)

    def test_log_gauss_pdf(self):
        # Test the _log_gauss_pdf method
        x = np.array([0.0, 1.0, 2.0])
        result = self.normal._log_gauss_pdf(x)
        expected_result = -0.5 * (x-self.mu)**2 / (self.sigma**2)
        np.testing.assert_allclose(result, expected_result, atol=1e-6)

    def test_norm_factor(self):
        # Test the _norm_factor method
        result = self.normal._norm_factor()
        expected_result = np.sqrt(2 * np.pi * self.sigma**2)
        self.assertAlmostEqual(result, expected_result, places=6)


class TestTruncatedNormal(unittest.TestCase):

    def setUp(self):
        self.DEBUG=False
        self.sigma = 1.15730
        self.a, self.b = -1.3, 1.6 
        self.mu = self.a + (self.b - self.a)/2.
        self.truncated_normal = TruncatedNormal(self.mu, self.sigma, self.a, self.b)
        self.number_of_samples = 30000
        self.samples = self.truncated_normal.sample(self.number_of_samples)

        if self.DEBUG:
            plt.figure()
            plt.hist(self.samples, bins = self.number_of_samples//1000 )
            print(self.mu)
            plt.vlines(self.mu, 0, 1000, color='r')

            plt.figure()
            x = np.linspace(2*self.a, 2*self.b, 256)
            y = self.truncated_normal(x, normalise=True, log=False)
            y2 = np.exp(-0.5 * (x-self.mu)**2 / (self.sigma**2) )
            c2 = np.sqrt(np.pi*2*self.sigma*self.sigma)
            plt.plot(x,y)
            plt.plot(x,y2/c2, 'r--')
            plt.show()

    def test_sample_shape(self):
        # Test if the sample method returns the correct shape
        self.assertEqual(self.samples.shape, (self.number_of_samples,))

    def test_call_method_normalised(self):
        # Test the __call__ method when normalise=True and log=False
        x = np.array([0.0, 1.0, 2.0])
        result = self.truncated_normal(x, normalise=True, log=False)
        expected_result = np.exp(-0.5 * (x-self.mu)**2 / (self.sigma**2) )
        expected_result[2] = 0
        expected_result = expected_result / expected_result[0]
        result = result / result[0]
        np.testing.assert_allclose(result, expected_result, atol=1e-6)

    def test_norm_factor(self):
        # Test that the normalisation constant is smaller than for 
        # a normal distirbution.
        x = np.array([0.0, 1.0, 2.0])
        c1 = self.truncated_normal._norm_factor()
        c2 = np.sqrt(np.pi*2*self.sigma*self.sigma)
        self.assertGreater(c2, c1)

class TestUniformKent(unittest.TestCase):

    def setUp(self):
        self.gamma = np.eye(3)  # Identity matrix as a simple example for orientation vectors
        self.kappa = 2.0
        self.beta = 0.5
        self.kent = Kent(self.gamma, self.kappa, self.beta)
        self.number_of_samples = 1000
        self.samples = self.kent.sample(self.number_of_samples)

    def test_sample_shape(self):
        # Test if the sample method returns the correct shape
        self.assertEqual(self.samples.shape, (3, self.number_of_samples))

    def test_sample_mean(self):
        # Test if that the sample mean is approximately at the mode.
        mean = np.mean(self.samples, axis=1)
        mean = mean / np.linalg.norm(mean)
        np.testing.assert_allclose(mean, self.gamma[:,0], atol=0.1)

    def test_sample_unit_norm(self):
        # Test if the samples lie on the unit sphere
        norms = np.linalg.norm(self.samples, axis=0)
        np.testing.assert_allclose(norms, 1, atol=1e-6)

    def test_call_method_normalised(self):
        # Test the PDF when normalise=True and log=False
        x = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])  # Unit vectors along z and x axes
        result = self.kent(x, normalise=True, log=False)
        self.assertTrue(np.all(result >= 0))  # Probabilities should be non-negative

    def test_call_method_log(self):
        # Test the PDF when normalise=True and log=True
        x = np.array([[0.0, 1.0], [0.0, 0.0], [1.0, 0.0]])  # Unit vectors along z and x axes
        result = self.kent(x, normalise=True, log=True)
        self.assertTrue(np.all(np.isfinite(result)))  # Log probabilities should be finite

    def test_norm_factor(self):
        # Test the _norm_factor method for normalisation constant
        gamma = np.eye(3)
        kappa = 4.34
        kent = Kent(gamma, kappa, beta=0)
        result = kent._norm_factor()
        expected_result = (4*np.pi/kappa)*np.sinh(kappa)
        np.testing.assert_allclose(result, expected_result, atol=1e-6)

        kent = Kent(gamma, kappa=1e-8, beta=0)
        result = kent._norm_factor()
        expected_result = 4*np.pi
        np.testing.assert_allclose(result, expected_result, atol=1e-6)

    def test_narrow_kent(self):
        # Test Kent with large kappa
        gamma = np.eye(3)
        kappa = 1500
        beta = 1
        kent = Kent(gamma, kappa, beta)
        phi_max = kent._uniform_spherical_cone.phi_max
        phi_max_approx = np.arccos(1 + np.log(kent._eps) / kappa) # if beta << kappa
        self.assertAlmostEqual(np.degrees(phi_max), np.degrees(phi_max_approx), places=1)

if __name__ == '__main__':
    unittest.main()