import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import gamma as gamma_function
from scipy.special import iv


class UniformSpherical:
    """
    The Uniform Spherical distribution.
    """

    def __init__(self):
        self._phi_max = np.pi

    def __call__(self, x):
        """
        Uniform Spherical PDF - i.e the likelihood of x.

        Args:
            x (:obj:`np.ndarray`): A shape (3, N) array of locations on the unit sphere.

        Returns:
            :obj:`float`: Likelihood of the given x.
        """
        return np.ones( (x.shape[1],) ) / (4*np.pi)

    def sample(self, number_of_samples):
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
        return self._spherical_to_cartesian(phi, theta)

    def _spherical_to_cartesian(self, phi, theta):
        """Spherical (angular) to Cartesian coordinate conversion.

        Args:
            phi (:obj:`np.ndarray`): Azimuthal angle [0, pi], measured 0 at zhat. radians. shape=(N,).
            theta (:obj:`np.ndarray`):  Polar angle [0, 2*pi],  measured 0 at xhat. radians. shape=(N,).

        Returns:
            :obj:`np.ndarray`: Cartesian coordinates. shape=(3,N).
        """
        x = np.sin(phi) * np.cos(theta)
        y = np.sin(phi) * np.sin(theta)
        z = np.cos(phi)
        return np.array([x, y, z])


class UniformSphericalCone:
    """
    The Uniform Spherical distribution supported on a finite cone segment.

    Args:
        phi_max (:obj:`float`): Max cone opening angle measured from the pole (radians).
    """

    def __init__(self, phi_max, pole):
        assert phi_max > 0
        self.phi_max = phi_max
        self.pole = pole / np.linalg.norm(pole)
        self._pole_rotation = self._get_rotation_to_pole(pole)
        self._us = UniformSpherical()
        self._us._phi_max = self.phi_max

    def __call__(self, x):
        """
        Uniform Spherical Cone PDF - i.e the likelihood of observing x.

        Args:
            x (:obj:`np.ndarray`): A shape (3, N) array of locations on the unit sphere.

        Returns:
            :obj:`float`: Likelihood of the given x.
        """
        area = 2*np.pi*(1-np.cos(self.phi_max/2.))
        pdf_support_array = np.arccos( x.T@self.pole ) < self.phi_max
        return pdf_support_array * np.ones( (x.shape[1],) ) / area

    def sample(self, number_of_samples):
        """
        Generate uniform samples from a spherical cap of angle `phi_max` on the unit sphere S^2.

        Args:
            number_of_samples (:obj:`int`): Number of samples to generate.

        Returns:
            :obj:`np.ndarray`: A (3, number_of_samples) array of sampled points on the unit sphere.
        """
        nhat = self._us.sample(number_of_samples)
        return self._pole_rotation.apply(nhat.T).T

    def _get_rotation_to_pole(self, pole):
        zhat = np.array([0, 0, 1])
        rot_ax = np.cross(zhat, pole)
        rot_ax = rot_ax / np.linalg.norm(rot_ax)
        angle = np.arccos(np.dot(pole, zhat))
        return Rotation.from_rotvec(rot_ax * angle)


class Normal:
    """
    Gaussian distribution.

    Args:
        mu (:obj:`float`): Mean value.
        sigma (:obj:`float`): Standard deviation.
    """

    def __init__(self, mu, sigma):
        assert sigma > 0
        self.mu = mu
        self.sigma = sigma

    def __call__(self, x, normalise=True, log=False):
        """
        Gaussian PDF - i.e the likelihood of observing x.

        Args:
            x (:obj:`np.ndarray`): A shape (N,) array probe locations.
            normalise (:obj:`bool`): if true; normalise the distribution. Defaults to True.
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
        gamma (:obj:`np.ndarray`): Orthonormal orientation vectors, shape=(3,3).
        kappa (:obj:`float`): Concentration parameter.
        beta (:obj:`float`): Ellipticity parameter.
    """

    def __init__(self, gamma, kappa, beta):
        assert kappa > 2*beta
        assert 2*beta >= 0
        assert np.allclose( gamma.T @ gamma, np.eye(3, 3))

        self.gamma = gamma
        self.kappa = kappa
        self.beta = beta



        if self.kappa < 100:

            phi_max = np.pi

        else:

            # A very narrow kent distirbution
            #
            # The idea is to set the phi_max to an angle such that the
            # likelihood raito between p_max and a point as phi_max is
            # equal to eps.
            #
            # Derivation:
            #
            # p1 = c*np.exp( kappa ), the mode, max(p)
            # p2 = c*np.exp( kappa*g1 + beta*g2*g2  ), walk along major axis
            # p2 / p1 = eps = np.exp( -kappa + kappa*g1 + beta*g2*g2  )
            # log(eps) = -kappa + kappa*g1 + beta*g2*g2
            # log(eps) = kappa*(-1 + g1) + beta*g2*g2
            # g1 = cos(alpha)
            # g2 = cos(np.pi/2 - alpha) = sin(alpha)
            # log(eps) + kappa = beta*sin(alpha)**2 + kappa*cos(alpha)
            # 1 + log(eps)/kappa = (beta/kappa)*sin(alpha)**2 + cos(alpha)
            #
            # let a = (beta/kappa) and b = 1 + log(eps)/kappa
            # b = a*sin(alpha)**2 + cos(alpha)
            # b = a*(1 - cos(alpha)**2) + cos(alpha)
            # 0 = -a*cos(alpha)**2 + cos(alpha) - b + a
            # 0 = cos(alpha)**2 + (-1/a)*cos(alpha) + ((b - a)/a)
            # cos(alpha) = (1/2*a) + sqrt( (1/4*a**2) - ((b - a)/a))
            #
            # In analogy we can consider a normal dist with mu=0:
            # log(eps) = -x**2 / ( 2 * std * std )
            # x = sqrt( -2*log(eps)*std**2 )
            # Then if we seek for instance x = N*std we get
            # N = sqrt( -2*log(eps)*std )
            # eps = np.exp( -N**2/(2*std) )
            # so N=3.5 gives eps=0.0021, and N=4 eps=0.000335
            # we select eps = 1e-5 which should be something like 4.5 stds,
            # given that the Kent is very similar to a normal on the sphere.

            eps = 1e-5
            b = 1 + np.log(eps)/kappa
            a = (beta/kappa)
            if a==0:
                phi_max = np.arccos( b )
            else:
                cc = (1/(2*a)) - np.sqrt( (1/(4*a*a)) - ((b - a)/a) )
                phi_max = np.arccos( cc )

        self._uniform_spherical_cone = UniformSphericalCone(phi_max, self.gamma[:,0])

    def __call__(self, x, normalise=True, log=False):
        """
        Kent PDF - i.e the likelihood of observing x.

        Args:
            x (:obj:`np.ndarray`): A shape (3, N) array of unit normals, i.e the probe locations.
            normalise (:obj:`bool`): if true; normalise the distribution. Defaults to True.
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
            return 4*np.pi * (1/self.kappa) * np.sinh(self.kappa)
        else:
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