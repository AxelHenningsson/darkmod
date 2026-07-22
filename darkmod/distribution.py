import numba as nb
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.special import erf, iv
from scipy.special import gamma as gamma_function


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
        return np.ones((x.shape[1],)) / (4 * np.pi)

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
        area = 2 * np.pi * (1 - np.cos(self.phi_max / 2.0))
        pdf_support_array = np.arccos(x.T @ self.pole) < self.phi_max
        return pdf_support_array * np.ones((x.shape[1],)) / area

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
        s2 = 2 * self.sigma * self.sigma
        dx = x - self.mu
        return -dx * dx / s2

    def _norm_factor(self):
        return np.sqrt(np.pi * 2 * self.sigma * self.sigma)


class TruncatedNormal:
    """
    Truncated Gaussian distribution.

    Args:
        mu (:obj:`float`): Mean value.
        sigma (:obj:`float`): Standard deviation.
        a (:obj:`float`): Lower truncation bound.
        b (:obj:`float`): Upper truncation bound.
    """

    def __init__(self, mu, sigma, a, b):
        assert sigma > 0
        assert a < b
        self.mu = mu
        self.sigma = sigma
        self.a = a
        self.b = b

    def __call__(self, x, normalise=True, log=False):
        """
        Truncated Gaussian PDF - likelihood of observing x.

        Args:
            x (:obj:`np.ndarray`): A shape (N,) array of probe locations.
            normalise (:obj:`bool`): If true; normalise the distribution. Defaults to True.
            log (:obj:`bool`): If true; return a log probability. Defaults to False.

        Returns:
            :obj:`float`: Likelihood of the given x.
        """
        log_exp = self._log_truncated_pdf(x)
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
        Generate a sample from the Truncated Normal Distribution.

        NOTE: This function implements simple rejection sampling using a Gaussian
        prior. Improvements can easily be made if speed becomes relevant.

        Args:
            number_of_samples (:obj:`int`): Number of samples to generate.

        Returns:
            :obj:`np.ndarray`: A sample of shape (number_of_samples,) from the Truncated Normal distribution.
        """
        samples = np.array([])
        while len(samples) < number_of_samples:
            s = np.random.normal(self.mu, self.sigma, size=(number_of_samples,))
            m = (self.a < s) * (self.b > s)
            samples = np.concatenate((samples, s[m]))
        return samples[0:number_of_samples]

    def _log_truncated_pdf(self, x):
        s2 = 2 * self.sigma * self.sigma
        dx = x - self.mu
        log_exp = -dx * dx / s2
        m = (self.a < x) * (self.b > x)
        log_exp[~m] = -np.inf
        return log_exp

    def _cdf(self, x):
        """Cumulative distribution function for a normal."""
        z = (x - self.mu) / self.sigma
        return 0.5 * (1 + erf(z / np.sqrt(2)))

    def _norm_factor(self):
        """Factor such that exp(...)/factor is integrates to 1 on a to b."""
        return np.sqrt(np.pi * 2) * self.sigma * (self._cdf(self.b) - self._cdf(self.a))


class MultivariateTruncatedNormal(object):
    """
    Multivariate Gaussian distribution.

    Args:
        mu (:obj:`float`): Mean vector. shape=(n,).
        cov (:obj:`float`): Covariance matrix. shape=(n,n).
        a (:obj:`float`): Lower truncation bounds. shape=(n,).
        b (:obj:`float`): Upper truncation bounds. shape=(n,).

    NOTE: This class does not implement PDF normalisation.

    """

    def __init__(self, mu, cov, a, b):
        assert cov.shape[0] == mu.shape[0], "covariance and mean shape do not match"
        assert cov.shape[0] == cov.shape[1], "covariance is not square"
        assert np.linalg.matrix_rank(cov) == len(mu), "ill conditioned covariance"
        assert np.linalg.cond(cov) < 1e12, "ill conditioned covariance"
        assert np.allclose(cov, cov.T), "Covariance is not symmetric"
        self.mu = mu
        self.cov = cov
        self._cov_inv = np.linalg.inv(cov)
        self.a = a
        self.b = b

    def __call__(self, x, log=False):
        """
        The non-normalised Multivariate Truncated aussian PDF.

        NOTE: this class does not implement PDF normalisation.

        Args:
            x (:obj:`np.ndarray`): Array of probe locations. Each column is a
                probe locaiton. shape=(dim, number_of_probe_locations).
            log (:obj:`bool`): if true; return a log probability. Defaults to False.

        Returns:
            :obj:`float`: Likelihood of the given x.
        """
        log_exp = self._log_mult_trunc_gauss_pdf(x)
        if log:
            return log_exp
        else:
            return np.exp(log_exp)

    def _is_supported(self, sample):
        ub = np.all(sample < self.b, axis=0)
        lb = np.all(sample > self.a, axis=0)
        return ub & lb

    def sample(self, number_of_samples):
        """
        Generate a sample from the multivariate Normal Distribution.

        Args:
            number_of_samples (:obj:`int`): Number of samples to generate.

        Returns:
            :obj:`np.ndarray`: Sample from a multivariate Gaussian. shape=(n, number_of_samples).
        """
        sample = np.random.multivariate_normal(
            self.mu, cov=self.cov, size=(number_of_samples,)
        ).T
        x = sample[:, self._is_supported(sample)]
        while x.shape[1] < number_of_samples:
            sample = np.random.multivariate_normal(
                self.mu, cov=self.cov, size=(number_of_samples,)
            ).T
            x = np.concatenate((x, sample[:, self._is_supported(sample)]), axis=1)
        return x[:, 0:number_of_samples]

    def _log_mult_trunc_gauss_pdf(self, x):
        dx = x - self.mu[:, np.newaxis]
        exponent = -0.5 * np.sum(dx * (self._cov_inv @ dx), axis=0)
        exponent[np.any(x < self.a, axis=0)] = -np.inf
        exponent[np.any(x > self.b, axis=0)] = -np.inf
        return exponent


@nb.njit(
    parallel=True,
    nogil=True,
    cache=True,
)
def _sample_diagonal_truncated_normal_numba(
    mu,
    std,
    lower,
    upper,
    out,
):
    """
    Sample independent truncated normal variables.

    Each column of `out` is one multivariate sample. Since the
    covariance matrix is diagonal, each component can be sampled
    independently.
    """
    dim = out.shape[0]
    n = out.shape[1]

    for i in nb.prange(n):
        for a in range(dim):
            mean = mu[a]
            sigma = std[a]
            lo = lower[a]
            hi = upper[a]

            while True:
                value = mean + sigma * np.random.standard_normal()

                if lo < value < hi:
                    out[a, i] = value
                    break


@nb.njit(
    parallel=True,
    nogil=True,
    cache=True,
)
def _apply_truncation_numba(
    x,
    lower,
    upper,
    log_pdf,
):
    dim = x.shape[0]
    n = x.shape[1]

    for i in nb.prange(n):
        supported = True

        for a in range(dim):
            value = x[a, i]

            if value < lower[a] or value > upper[a]:
                supported = False
                break

        if not supported:
            log_pdf[i] = -np.inf


class MultivariateDiagonalTruncatedNormal:
    """
    Multivariate Gaussian distribution with diagonal covariance and
    independent lower and upper truncation bounds.

    Args:
        mu (:obj:`np.ndarray`):
            Mean vector with shape ``(n,)``.

        cov (:obj:`np.ndarray`):
            Diagonal covariance matrix with shape ``(n, n)``.

        a (:obj:`np.ndarray`):
            Lower truncation bounds with shape ``(n,)`` or ``(n, 1)``.

        b (:obj:`np.ndarray`):
            Upper truncation bounds with shape ``(n,)`` or ``(n, 1)``.

    Notes:
        This class does not implement PDF normalization.

        Sampling is exact for a diagonal covariance matrix. Each component
        is sampled independently from its one-dimensional truncated normal
        distribution using rejection sampling.

        Parallel sampling uses Numba ``prange`` over output samples.
    """

    def __init__(self, mu, cov, a, b):
        mu = np.asarray(mu, dtype=np.float64).reshape(-1)
        cov = np.asarray(cov, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64).reshape(-1)
        b = np.asarray(b, dtype=np.float64).reshape(-1)

        dim = mu.size

        if cov.shape != (dim, dim):
            raise ValueError("Covariance and mean shapes do not match.")

        if a.shape != (dim,):
            raise ValueError("Lower-bound and mean shapes do not match.")

        if b.shape != (dim,):
            raise ValueError("Upper-bound and mean shapes do not match.")

        if not np.allclose(cov, cov.T):
            raise ValueError("Covariance matrix is not symmetric.")

        variance = np.diag(cov)

        if not np.allclose(
            cov,
            np.diag(variance),
        ):
            raise ValueError(
                "MultivariateDiagonalTruncatedNormal requires "
                "a diagonal covariance matrix."
            )

        if np.any(~np.isfinite(variance)):
            raise ValueError("Covariance contains non-finite values.")

        if np.any(variance <= 0):
            raise ValueError("All diagonal covariance entries must be positive.")

        if np.any(a >= b):
            raise ValueError("Every lower bound must be smaller than its upper bound.")

        self.mu = np.ascontiguousarray(mu)
        self.cov = np.ascontiguousarray(cov)

        self.a = np.ascontiguousarray(a)
        self.b = np.ascontiguousarray(b)

        self.std = np.ascontiguousarray(np.sqrt(variance))

        self._cov_inv = np.ascontiguousarray(np.diag(1.0 / variance))

    def __call__(self, x, log=False):
        """
        Evaluate the non-normalized truncated Gaussian PDF.

        Args:
            x (:obj:`np.ndarray`):
                Probe locations. Each column is one location, with shape
                ``(dim, number_of_probe_locations)``.

            log (:obj:`bool`):
                Return log probabilities when True.

        Returns:
            :obj:`np.ndarray`:
                Probability or log-probability for each column of ``x``.
        """
        log_pdf = self._log_mult_trunc_gauss_pdf(x)

        if log:
            return log_pdf

        return np.exp(log_pdf)

    def _is_supported(self, sample):
        """
        Return a Boolean mask indicating which columns satisfy all bounds.
        """
        sample = np.asarray(sample)

        if sample.ndim != 2:
            raise ValueError("Sample must have shape (dimension, number_of_samples).")

        if sample.shape[0] != self.mu.size:
            raise ValueError("Sample dimension does not match distribution dimension.")

        return np.all(
            (sample > self.a[:, None]) & (sample < self.b[:, None]),
            axis=0,
        )

    def sample(self, number_of_samples):
        """
        Generate samples from the diagonal multivariate truncated normal.

        Args:
            number_of_samples (:obj:`int`):
                Number of samples to generate.

        Returns:
            :obj:`np.ndarray`:
                Samples with shape ``(dimension, number_of_samples)``.
        """
        number_of_samples = int(number_of_samples)

        if number_of_samples < 0:
            raise ValueError("number_of_samples must be non-negative.")

        out = np.empty(
            (self.mu.size, number_of_samples),
            dtype=np.float64,
        )

        if number_of_samples == 0:
            return out

        _sample_diagonal_truncated_normal_numba(
            self.mu,
            self.std,
            self.a,
            self.b,
            out,
        )

        return out

    def _log_mult_trunc_gauss_pdf(self, x):
        x = np.asarray(
            x,
            dtype=np.float64,
            order="C",
        )

        if x.ndim != 2:
            raise ValueError("x must have shape (dimension, number_of_points).")

        if x.shape[0] != self.mu.size:
            raise ValueError("x dimension does not match distribution dimension.")

        out = np.empty(
            x.shape[1],
            dtype=np.float64,
        )

        _mvn_logpdf_numba(
            x,
            self.mu,
            self._cov_inv,
            0.0,
            False,
            out,
        )

        _apply_truncation_numba(
            x,
            self.a,
            self.b,
            out,
        )

        return out


@nb.njit(parallel=True, fastmath=True)
def _mvn_logpdf_numba(x, mu, precision, log_norm, normalise, out):
    dim = x.shape[0]
    n = x.shape[1]

    for i in nb.prange(n):
        q = 0.0

        for a in range(dim):
            da = x[a, i] - mu[a]

            for b in range(dim):
                db = x[b, i] - mu[b]
                q += da * precision[a, b] * db

        y = -0.5 * q

        if normalise:
            y -= log_norm

        out[i] = y


@nb.njit(parallel=True, fastmath=True)
def _mvn_pdf_numba(x, mu, precision, log_norm, normalise, out):
    dim = x.shape[0]
    n = x.shape[1]

    for i in nb.prange(n):
        q = 0.0

        for a in range(dim):
            da = x[a, i] - mu[a]

            for b in range(dim):
                db = x[b, i] - mu[b]
                q += da * precision[a, b] * db

        y = -0.5 * q

        if normalise:
            y -= log_norm

        out[i] = np.exp(y)


class MultivariateNormal:
    """
    Fast multivariate Gaussian distribution.

    x must have shape (dim, n), where each column is one point.
    """

    def __init__(self, mu, cov, dtype=np.float64):
        self.mu = np.asarray(mu, dtype=dtype)
        self.cov = np.asarray(cov, dtype=dtype)

        if self.mu.ndim != 1:
            raise ValueError("mu must have shape (dim,)")

        if self.cov.ndim != 2:
            raise ValueError("cov must have shape (dim, dim)")

        if self.cov.shape[0] != self.cov.shape[1]:
            raise ValueError("covariance is not square")

        if self.cov.shape[0] != self.mu.shape[0]:
            raise ValueError("covariance and mean shape do not match")

        if not np.allclose(self.cov, self.cov.T):
            raise ValueError("covariance is not symmetric")

        try:
            np.linalg.cholesky(self.cov)
        except np.linalg.LinAlgError:
            raise ValueError("covariance must be positive definite")

        cond = np.linalg.cond(self.cov)
        if cond > 1e12:
            raise ValueError(f"ill-conditioned covariance, cond={cond}")

        sign, logdet = np.linalg.slogdet(self.cov)
        if sign <= 0:
            raise ValueError("covariance must be positive definite")

        self.dim = self.mu.shape[0]
        self.precision = np.linalg.inv(self.cov).astype(dtype, copy=False)
        self.log_norm = dtype(0.5 * (self.dim * np.log(2.0 * np.pi) + logdet))
        self.dtype = dtype

    def __call__(self, x, normalise=True, log=False, out=None):
        x = np.asarray(x)

        if x.ndim != 2:
            raise ValueError("x must have shape (dim, n)")

        if x.shape[0] != self.dim:
            raise ValueError(f"x has dim {x.shape[0]}, expected {self.dim}")

        if out is None:
            out = np.empty(x.shape[1], dtype=x.dtype)

        if out.shape != (x.shape[1],):
            raise ValueError("out must have shape (n,)")

        if log:
            _mvn_logpdf_numba(
                x,
                self.mu,
                self.precision,
                self.log_norm,
                normalise,
                out,
            )
        else:
            _mvn_pdf_numba(
                x,
                self.mu,
                self.precision,
                self.log_norm,
                normalise,
                out,
            )

        return out

    def logpdf(self, x, normalise=True, out=None):
        return self(x, normalise=normalise, log=True, out=out)

    def pdf(self, x, normalise=True, out=None):
        return self(x, normalise=normalise, log=False, out=out)

    def sample(self, number_of_samples):
        return np.random.multivariate_normal(
            self.mu,
            self.cov,
            size=number_of_samples,
        ).T

    def _log_mult_gauss_pdf(self, x):
        return self.logpdf(x, normalise=False)

    def _norm_factor(self):
        return np.exp(self.log_norm)


# class MultivariateNormal(object):
#     """
#     Multivariate Gaussian distribution.

#     Args:
#         mu (:obj:`float`): Mean vector. shape=(n,).
#         cov (:obj:`float`): Covariance matrix. shape=(n,n).
#     """

#     def __init__(self, mu, cov):
#         assert cov.shape[0] == mu.shape[0], "covariance and mean shape do not match"
#         assert cov.shape[0] == cov.shape[1], "covariance is not square"
#         assert np.linalg.matrix_rank(cov) == len(mu), "ill conditioned covariance"
#         assert np.linalg.cond(cov) < 1e12, "ill conditioned covariance"
#         assert np.allclose(cov, cov.T), "Covariance is not symmetric"
#         self.mu = mu
#         self.cov = cov
#         self._cov_inv = np.linalg.inv(cov)

#     def __call__(self, x, normalise=True, log=False):
#         """
#         Multivariate Gaussian PDF - i.e the likelihood of observing x.

#         Args:
#             x (:obj:`np.ndarray`): Array of probe locations. Each column is a
#                 probe locaiton. shape=(dim, number_of_probe_locations).
#             normalise (:obj:`bool`): if true; normalise the distribution. Defaults to True.
#             log (:obj:`bool`): if true; return a log probability. Defaults to False.

#         Returns:
#             :obj:`float`: Likelihood of the given x.
#         """
#         log_exp = self._log_mult_gauss_pdf(x)
#         if normalise and not log:
#             return np.exp(log_exp) / self._norm_factor()
#         elif normalise and log:
#             return log_exp - np.log(self._norm_factor())
#         elif not normalise and not log:
#             return np.exp(log_exp)
#         elif not normalise and log:
#             return log_exp

#     def sample(self, number_of_samples):
#         """
#         Generate a sample from the multivariate Normal Distribution.

#         Args:
#             number_of_samples (:obj:`int`): Number of samples to generate.

#         Returns:
#             :obj:`np.ndarray`: Sample from a multivariate Gaussian. shape=(n, number_of_samples).
#         """
#         return np.random.multivariate_normal(
#             self.mu, cov=self.cov, size=(number_of_samples,)
#         ).T

#     def _log_mult_gauss_pdf(self, x):
#         dx = x - self.mu[:, np.newaxis]
#         return -0.5 * np.sum(dx * (self._cov_inv @ dx), axis=0)

#     def _norm_factor(self):
#         return np.sqrt((2 * np.pi) ** len(self.mu) * np.linalg.det(self.cov))


class Kent:
    """
    Parent class for reciprocal space resolution functions.

    Args:
        gamma (:obj:`np.ndarray`): Orthonormal orientation vectors, shape=(3,3).
        kappa (:obj:`float`): Concentration parameter.
        beta (:obj:`float`): Ellipticity parameter.
    """

    def __init__(self, gamma, kappa, beta):
        assert kappa > 2 * beta
        assert 2 * beta >= 0
        assert np.allclose(gamma.T @ gamma, np.eye(3, 3))

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

            self._eps = 1e-5
            b = 1 + np.log(self._eps) / kappa
            a = beta / kappa
            if a == 0:
                phi_max = np.arccos(b)
            else:
                cc = (1 / (2 * a)) - np.sqrt((1 / (4 * a * a)) - ((b - a) / a))
                phi_max = np.arccos(cc)

        self._uniform_spherical_cone = UniformSphericalCone(phi_max, self.gamma[:, 0])

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
            return 4 * np.pi * (1 / self.kappa) * np.sinh(self.kappa)
        else:
            normalisation_const = 0
            term_number_j = 0
            j = 0
            while j < 10 or np.abs(term_number_j) > np.abs(normalisation_const) * rtol:
                f1 = np.exp(
                    np.log(self.beta) * 2 * j
                    + np.log(0.5 * self.kappa) * (-2 * j - 0.5)
                )
                f2 = iv(2 * j + 0.5, self.kappa)
                f3 = gamma_function(j + 0.5) / gamma_function(j + 1)
                term_number_j = f1 * f2 * f3
                normalisation_const += term_number_j
                j += 1
            return 2 * np.pi * normalisation_const


if __name__ == "__main__":
    pass
