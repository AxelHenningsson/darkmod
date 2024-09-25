import matplotlib.pyplot as plt
import numpy as np
from scipy.special import gamma as gamma_function
from scipy.special import iv


class UniformSpherical:
    """
    The Uniform Spherical distribution.
    """

    def __init__(self):
        self._phi_max = np.pi

    def sample(number_of_samples, phi_max):
        """
        Generate uniform samples on a sphere.

        Args:
            number_of_samples (:obj:`int`): Number of samples to generate.

        Returns:
            :obj:`np.ndarray`: A (3, number_of_samples) array of sampled points on the unit sphere.
        """
        u, v = np.random.rand(2, number_of_samples)
        v_max = (1 - np.cos(self._phi_max)) / 2
        v_clipped = v * v_max
        theta = 2 * np.pi * u
        phi = np.arccos(1 - 2 * v_clipped)
        return self._polar_to_cartesian(phi, theta)

    def __call__(self, x, normalise=True, log=False):
            pass

    def _polar_to_cartesian(self, phi, theta):
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.array([x, y, z])

class UniformSphericalCone:
    """
    The Uniform Spherical distribution on cone segment.

    Args:
        phi_max (:obj:`float`): Max cone opening angle measured form north pole (radians).
    """

    def __init__(self, phi_max, pole):
        self.phi_max = phi_max
        self.pole = pole
        self._pole_rotation = self._get_rotation_to_pole(pole)
        self_us = UniformSpherical()
        self_us._phi_max = self.phi_max

    def _get_rotation_to_pole(self, pole):
        zhat = np.array([0, 0, 1])
        rot_ax = np.cross(zhat, pole)
        rot_ax = rot_ax / np.linalg.norm(rot_ax)
        angle = np.arccos(np.dot(pole, zhat))
        return Rotation.from_rotvec(rot_ax * angle)

    def sample(number_of_samples, phi_max):
        """
        Generate uniform samples from a spherical cap of angle `phi_max` on the unit sphere S^2.

        Args:
            number_of_samples (:obj:`int`): Number of samples to generate.

        Returns:
            :obj:`np.ndarray`: A (3, number_of_samples) array of sampled points on the unit sphere.
        """
        nhat = self._us.sample()
        return self.pole_rotation.apply(nhat.T).T

    def __call__(self, x, normalise=True, log=False):
            pass


class Normal:
    """
    Gaussian distribution.

    Args:
        mu (:obj:`float`): Mean value.
        sigma (:obj:`float`): Standard deviation.
    """


    def __init__(self, mu, sigma):
        self.mu = mu
        self.sigma = sigma


    def __call__(self, x, normalise=True, log=False):
        """
        Calculate the likelihood of a set of Q vectors.

        Args:
            x (:obj:`np.ndarray`): A shape (N,) array probe locations.
            normalise (:obj:`bool`): if true; normalise the disitrbution. Defaults to True.
            log (:obj:`bool`): if true; return a log probability. Defaults to False.

        Returns:
            :obj:`float`: Likelihood of the given x.
        """
        log_exp = self._log_gauss_pdf(x)
        if normalise and not log:
            return np.exp(log_exp) / self._norm_factor()
        elif normalise and log:
            return log_exp - np.log(self._norm_factor())
        elif not normalise and not log:
            return np.exp(log_exp)
        elif not normalise and log:
            return log_exp


    def sample(self, number_of_samples):
        """
        Generate a sample from the Normal Distribution.

        Args:
            number_of_samples (:obj:`int`): Number of samples to generate.

        Returns:
            :obj:`np.ndarray`: A sample of shape (number_of_samples,) from the Normal distribution.
        """
        return np.random.normal(self.mu, scale=self.sigma, size=(number_of_samples,))


    def _log_gauss_pdf(self, x):
        s2 = 2*self.sigma*self.sigma
        dx = (x - self.mu)
        return - dx * dx / s2


    def _norm_factor(self):
        return np.sqrt(np.pi*2*self.sigma*self.sigma)




class Kent:
    """
    Parent class for reciprocal space resolution functions.

    Args:
        gamma (:obj:`np.ndarray`): Orientation vectors.
        kappa (:obj:`float`): Concentration parameter).
        beta (:obj:`float`): Ellipticity parameter.
    """

    def __init__(self, gamma, kappa, beta):
        self.gamma = gamma
        self.kappa = kappa
        self.beta = beta

        eps = 1e-4
        phi_max = np.arccos(1 + np.log(eps) / self.kappa)
        phi_max *= (1 + 2 * self.beta / self.kappa)
        self._uniform_spherical_cone = UniformSphericalCone(phi_max, self.gamma[:,0])

    def __call__(self, x, normalise=True, log=False):
        """
        Calculate the likelihood of a set of Q vectors.

        Args:
            x (:obj:`np.ndarray`): A shape (3, N) array of unit normals, i.e the probe locations.
            normalise (:obj:`bool`): if true; normalise the disitrbution. Defaults to True.
            log (:obj:`bool`): if true; return a log probability. Defaults to False.

        Returns:
            :obj:`float`: Likelihood of the given x vectors.
        """
        log_exp = self._log_kent_pdf(x)
        if normalise and not log:
            return np.exp(log_exp) / self._norm_factor()
        elif normalise and log:
            return log_exp - np.log(self._norm_factor())
        elif not normalise and not log:
            return np.exp(log_exp)
        elif not normalise and log:
            return log_exp


    def sample(self, number_of_samples):
        """
        Generate a sample from the Kent Distribution using rejection sampling.

        Args:
            number_of_samples (:obj:`int`): Number of samples to generate.

        Returns:
            :obj:`np.ndarray`: A sample of shape (3, number_of_samples) from the Kent distribution.
        """
        sample = np.empty((3, 0))

        while sample.shape[1] < number_of_samples:
            proposal_sample = self._uniform_spherical_cone.sample(number_of_samples)

            log_f = self._log_kent_pdf(proposal_sample)
            log_max_p = self._log_kent_pdf(self.gamma[:, 0])

            rand_nums = np.random.rand(number_of_samples)
            mask = np.log(rand_nums) < (log_f - log_max_p)

            sample = np.concatenate((sample, proposal_sample[:, mask]), axis=1)

        return sample[:, :number_of_samples]


    def _log_kent_pdf(self, x):
        """
        Compute the un-nomralised-log-PDF of the Kent distribution for a set of points x on S^2.

        Args:
            x (:obj:`np.ndarray`): A set of points on the unit sphere of shape (3, N).

        Returns:
            :obj:`np.ndarray`: The un-nomralised-log-PDF values of the Kent distribution for each point in x, shape (N,).
        """
        g1, g2, g3 = self.gamma.T @ x
        return self.kappa * g1 + self.beta * (g2**2 - g3**2)


    def _norm_factor(self, rtol=1e-16):
        """Approximate normalisation constant for the Kent distribution.

        (This is the c(kappa, beta) function @ https://en.wikipedia.org/wiki/Kent_distribution)
        """
        if self.beta == 0:
            if self.kappa == 0:
                return 4*np.pi
            return 4*np.pi * (1/self.kappa) * np.sinh(self.kappa)
        normalisation_const = 0
        term_number_j = 0
        j = 0
        while j < 10 or np.abs(term_number_j) > np.abs(normalisation_const)*rtol:
            f1 = np.exp( np.log(self.beta)*2*j + np.log(0.5*self.kappa)*(-2*j-0.5) )
            f2 = iv(2*j+0.5, self.kappa)
            f3 = gamma_function(j+0.5) / gamma_function(j+1)
            term_number_j = f1*f2*f3
            normalisation_const += term_number_j
            j+=1
        return 2*np.pi*normalisation_const


if __name__ == "__main__":
    pass