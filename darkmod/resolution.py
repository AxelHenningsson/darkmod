import matplotlib.pyplot as plt
import numba as nb
import numpy as np
from numba import njit, prange
from scipy.interpolate import RegularGridInterpolator

from darkmod import laue
from darkmod.distribution import (
    Kent,
    MultivariateDiagonalTruncatedNormal,
    MultivariateNormal,
    Normal,
)
from darkmod.transforms import Q_to_lab, lab_to_Q


@nb.njit(
    nogil=True,
    cache=True,
)
def _accumulate_histogramdd_uniform_numba(
    sample,
    lower_edges,
    spacing,
    counts,
):
    nx, ny, nz = counts.shape
    n = sample.shape[1]

    x0, y0, z0 = lower_edges
    dx, dy, dz = spacing

    inv_dx = 1.0 / dx
    inv_dy = 1.0 / dy
    inv_dz = 1.0 / dz

    xmax = x0 + nx * dx
    ymax = y0 + ny * dy
    zmax = z0 + nz * dz

    for i in range(n):
        x = sample[0, i]
        y = sample[1, i]
        z = sample[2, i]

        if x < x0 or x > xmax or y < y0 or y > ymax or z < z0 or z > zmax:
            continue

        ix = int((x - x0) * inv_dx)
        iy = int((y - y0) * inv_dy)
        iz = int((z - z0) * inv_dz)

        if ix == nx:
            ix = nx - 1

        if iy == ny:
            iy = ny - 1

        if iz == nz:
            iz = nz - 1

        counts[ix, iy, iz] += 1


class TruncatedPentaGauss(object):
    """This model is a generalisation of the PentaGauss model in that each of the gaussian variables
    can be clipped by an upper and lower bound, allowing, for instance, the simulation of a fixed
    aperture CRL.

    This is similar to Poulsen 2021.

    NOTE the CRL is implicitly treated to have a square opening - not a round.

    Args:
        optical_axis_0 (:obj:`float`): Nominal diffracted ray path, shape=(3,).
        mean_wavelength (:obj:`float`): Mean of the wavelength distribution
        xray_parameters (:obj:`dict` of `float`): The keys are:

            "std_beam_horizontal": Standard deviation of the angular horizontal spread (beam).
            "lower_bound_beam_horizontal": Lower truncation bound of the angular horizontal spread (beam).
            "upper_bound_beam_horizontal": Upper truncation bound of the angular horizontal spread (beam).

            "std_beam_vertical": Standard deviation of the angular vertical spread (beam).
            "lower_bound_beam_vertical": Lower truncation bound of the angular vertical spread (beam).
            "upper_bound_beam_vertical": Upper truncation bound of the angular vertical spread (beam).

            "std_CRL_horizontal": Standard deviation of the angular horizontal spread (CRL).
            "lower_bound_CRL_horizontal": Lower truncation bound of the angular horizontal spread (CRL).
            "upper_bound_CRL_horizontal": Upper truncation bound of the angular horizontal spread (CRL).

            "std_CRL_vertical": Standard deviation of the angular vertical spread (CRL).
            "lower_bound_CRL_vertical": Lower truncation bound of the angular vertical spread (CRL).
            "upper_bound_CRL_vertical": Upper truncation bound of the angular vertical spread (CRL).

            "std_energy_shift": Standard deviation of the epsilon=dk/k distribution.
            "lower_bound_energy_shift": Lower truncation bound of epsilon.
            "upper_bound_energy_shift": Upper truncation bound of epsilon.

        when a bound is set to None it will not be applied, likewise if a bound is missing
        from the dict the default is to not apply the trunctation.

    """

    def __init__(self, optical_axis_0, mean_wavelength, xray_parameters):

        self._par = self._extract_parameters(xray_parameters)
        self.optical_axis_0 = optical_axis_0
        self.theta_0, self.eta_0 = self._get_theta_eta()
        self._mean_wavelength = mean_wavelength
        self.Q = self._get_M()[:, 0]
        self.dQ_theta_shift = 0  # tth shifts
        self._is_compiled = False

        self._cov_x = np.eye(5, 5)
        self._cov_x[0, 0] = self._par["std_energy_shift"] ** 2
        self._cov_x[1, 1] = self._par["std_beam_horizontal"] ** 2
        self._cov_x[2, 2] = self._par["std_beam_vertical"] ** 2
        self._cov_x[3, 3] = self._par["std_CRL_horizontal"] ** 2
        self._cov_x[4, 4] = self._par["std_CRL_vertical"] ** 2

        self._lower_bound_x = np.array(
            [
                self._par["lower_bound_energy_shift"],
                self._par["lower_bound_beam_horizontal"],
                self._par["lower_bound_beam_vertical"],
                self._par["lower_bound_CRL_horizontal"],
                self._par["lower_bound_CRL_vertical"],
            ]
        ).reshape(5, 1)

        self._upper_bound_x = np.array(
            [
                self._par["upper_bound_energy_shift"],
                self._par["upper_bound_beam_horizontal"],
                self._par["upper_bound_beam_vertical"],
                self._par["upper_bound_CRL_horizontal"],
                self._par["upper_bound_CRL_vertical"],
            ]
        ).reshape(5, 1)
        print("using MultivariateDiagonalTruncatedNormal")
        self._x = MultivariateDiagonalTruncatedNormal(
            np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
            self._cov_x,
            self._lower_bound_x,
            self._upper_bound_x,
        )

    def _extract_parameters(self, params):
        keys = [
            "std_beam_horizontal",
            "lower_bound_beam_horizontal",
            "upper_bound_beam_horizontal",
            "std_beam_vertical",
            "lower_bound_beam_vertical",
            "upper_bound_beam_vertical",
            "std_CRL_horizontal",
            "lower_bound_CRL_horizontal",
            "upper_bound_CRL_horizontal",
            "std_CRL_vertical",
            "lower_bound_CRL_vertical",
            "upper_bound_CRL_vertical",
            "std_energy_shift",
            "lower_bound_energy_shift",
            "upper_bound_energy_shift",
        ]
        new_pars = {}
        for key in keys:
            if key not in params.keys():
                params[key] = None
            else:
                if key.startswith("std"):
                    assert params[key] > 0
                    new_pars[key] = params[key]
                elif key.startswith("lower"):
                    if params[key] is None:
                        new_pars[key] = -np.inf
                    else:
                        new_pars[key] = params[key]
                elif key.startswith("upper"):
                    if params[key] is None:
                        new_pars[key] = np.inf
                    else:
                        new_pars[key] = params[key]
        assert set(keys) == set(params.keys())
        return new_pars

    def sample(self, number_of_samples):
        """
        Generate samples of Q vectors.

        Returns a sample in lab-coordinates by default.

        Args:
            number_of_samples (:obj:`int`): Number of samples to generate.

        Returns:
            :obj:`np.ndarray`: A sample of Q vectors of shape (3, number_of_samples).
        """
        M = self._get_M()
        return M @ self._x.sample(number_of_samples)

    def compile(
        self,
        resolution=(5e-4, 5e-4, 5e-4),
        ranges=(5, 5, 5),
        number_of_samples=None,
        support_samples=1_000_000,
        chunk_size=10_000_000,
        store_integration_points=False,
    ):
        """
        Compile a Monte Carlo approximation of the reciprocal-space
        resolution function.

        Sampling is performed in chunks, while all samples are accumulated
        directly into a single dense histogram.

        Args:
            resolution:
                Reciprocal-space grid spacing in each Q direction.

            ranges:
                Approximate number of standard deviations included in each
                Q direction.

            number_of_samples:
                Total number of Monte Carlo samples. If None, approximately
                100 samples per bin are used, up to 40 million samples.

            support_samples:
                Number of samples used to estimate the lookup-grid support.

            chunk_size:
                Maximum number of Monte Carlo samples processed at once.

            store_integration_points:
                Store all lookup-grid points in laboratory coordinates.
                This is expensive and normally unnecessary.
        """
        resolution = np.asarray(
            resolution,
            dtype=np.float64,
        )

        ranges = np.asarray(
            ranges,
            dtype=np.float64,
        )

        if resolution.shape != (3,):
            raise ValueError("resolution must contain exactly three values")

        if ranges.shape != (3,):
            raise ValueError("ranges must contain exactly three values")

        if np.any(resolution <= 0):
            raise ValueError("all resolution values must be positive")

        if np.any(ranges <= 0):
            raise ValueError("all range values must be positive")

        support_samples = int(support_samples)
        chunk_size = int(chunk_size)

        if support_samples <= 0:
            raise ValueError("support_samples must be positive")

        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")

        q_ranges = self.estimate_p_Q_support(
            self.Q,
            resolution,
            ranges,
            number_of_samples=support_samples,
        )

        grid_shape = tuple(len(axis) for axis in q_ranges)

        nbins = int(np.prod(grid_shape, dtype=np.int64))

        if number_of_samples is None:
            number_of_samples = min(
                100 * nbins,
                40_000_000,
            )

        number_of_samples = int(number_of_samples)

        if number_of_samples <= 0:
            raise ValueError("number_of_samples must be positive")

        number_of_chunks = (number_of_samples + chunk_size - 1) // chunk_size

        lower_edges = np.array(
            [
                q_ranges[0][0] - 0.5 * resolution[0],
                q_ranges[1][0] - 0.5 * resolution[1],
                q_ranges[2][0] - 0.5 * resolution[2],
            ],
            dtype=np.float64,
        )

        print(f"Using {number_of_samples:,} samples in {number_of_chunks} chunks")

        print(f"Grid shape: {grid_shape} ({nbins:,} bins)")

        samples_per_bin = number_of_samples / nbins

        print(f"Average samples per bin: {samples_per_bin:.4g}")

        if samples_per_bin < 1:
            print("Warning: fewer than one sample per bin on average")

        # uint32 is sufficient while total sample count remains below 2**32.
        if number_of_samples >= np.iinfo(np.uint32).max:
            raise ValueError(
                "number_of_samples is too large for uint32 histogram counts"
            )

        total_counts = np.zeros(
            grid_shape,
            dtype=np.uint32,
        )

        # Combine:
        #
        #     x -> self._get_M() @ x -> lab_to_Q(...)
        #
        # into one matrix multiplication.
        M_q = np.ascontiguousarray(
            lab_to_Q(
                self._get_M(),
                self.Q,
            ),
            dtype=np.float64,
        )

        collected_samples = 0

        while collected_samples < number_of_samples:
            current_chunk_size = min(
                chunk_size,
                number_of_samples - collected_samples,
            )

            x_sample = self._x.sample(current_chunk_size)

            sample_q = np.ascontiguousarray(
                M_q @ x_sample,
                dtype=np.float64,
            )

            _accumulate_histogramdd_uniform_numba(
                sample_q,
                lower_edges,
                resolution,
                total_counts,
            )

            collected_samples += current_chunk_size

        maximum_count = int(np.max(total_counts))

        if maximum_count <= 0:
            raise RuntimeError("No Monte Carlo samples entered the lookup grid")

        # Division by number_of_samples is unnecessary because the lookup
        # function is normalized by its maximum.
        self.p_Q = total_counts.astype(np.float32)

        self.p_Q /= maximum_count

        # Poisson uncertainty in the same peak-normalized units.
        self.std_p_Q = np.sqrt(
            total_counts,
            dtype=np.float32,
        )

        self.std_p_Q /= maximum_count

        if store_integration_points:
            self._integration_points, _ = self._get_integration_points(q_ranges)
        else:
            self._integration_points = None

        self._set_interpolation(
            q_ranges,
            self.p_Q,
            self.std_p_Q,
        )

        self._is_compiled = True

    def _bin_centers_to_edges(self, bin_centers):
        edges = (bin_centers[1:] + bin_centers[:-1]) / 2
        first_edge = bin_centers[0] - (bin_centers[1] - bin_centers[0]) / 2
        last_edge = bin_centers[-1] + (bin_centers[-1] - bin_centers[-2]) / 2
        return np.concatenate([[first_edge], edges, [last_edge]])

    def __call__(self, Q_vectors, error_estimate=False, angular_crl_shifts=None):
        """
        Calculate the likelihood of a set of Q vectors.

        Args:
            Q_vectors (:obj:`np.ndarray`): A shape (3, N) array of Q vectors.
            error_estimate (:obj:`bool`): If true, returns an estimated upper bound
                uncertainty for each data point. The integration errror in the resolution
                function is assumed to be bounded by this value.

        Returns:
            :obj:`np.ndarray`: Likelihood of the given Q vectors. shape (N, )
        """
        assert len(Q_vectors.shape) == 2 and Q_vectors.shape[0] == 3

        if self._is_compiled:
            if angular_crl_shifts is not None:
                dQ_angular_shift = self._get_Q_shifts(angular_crl_shifts)
            else:
                dQ_angular_shift = 0

                Q_vectors_q_system = lab_to_Q(
                    Q_vectors + dQ_angular_shift + self.dQ_theta_shift,
                    self.Q,
                )

                p_Q = trilinear_uniform_grid(
                    Q_vectors_q_system,
                    self._p_Q_values,
                    self._interp_origin,
                    self._interp_spacing,
                )

                if error_estimate:
                    std_p_Q = trilinear_uniform_grid(
                        Q_vectors_q_system,
                        self._std_p_Q_values,
                        self._interp_origin,
                        self._interp_spacing,
                    )
                    return p_Q, std_p_Q

                return p_Q
        else:
            raise ValueError(
                "The resolution function requires compiling before any calls can be made to the PDF."
            )

    def _get_Q_shifts(self, angular_crl_shifts):
        M = self._get_M()
        dQ = M[:, -2:] @ angular_crl_shifts
        return dQ

    def theta_shift(self, theta):
        """Approximate shift the resolution in theta, corresponds to moving the CRL.

        This will not move the truncation bounds as this would require recompiling.

        This moves the mean of the vertical dsitirbution of the CRL.

        NOTE: this will not recompile the resolution fuction. I.e the covariance
        will not change, only the mean.

        Args:
            theta (:obj:`float`): The new theta position in radians.

        """
        if self._is_compiled:
            M = self._get_M()
            delta_two_theta = 2 * theta - 2 * self.theta_0
            self._x.mu[4] = (
                delta_two_theta  # mean in vertical CRL position. TODO: include the eta stuff.
            )
            self.dQ_theta_shift = (self.Q - M @ self._x.mu).reshape(3, 1)
        else:
            raise ValueError(
                "The resolution function requires compiling before any theta shifts can be introduced."
            )

    def _get_integration_points(self, q_ranges):
        """Get lab-cooridnate integration points for MC integration from Q-system grid.

        Args:
            q_ranges (:obj:`iterable` of `np.ndarray`): qx_range, qy_range, qz_range given in Q-system.
                These are the monotonically increasing arrays that define the integration grid in the
                local Q-system.

        Returns:
            :obj:`tuple` of `np.ndarray` and `tuple`: q_points_lab, Qx.shape integration
                points and the 3d shape of the grid. q_points_lab.shape = (3,N).

        """
        Qx, Qy, Qz = np.meshgrid(*q_ranges, indexing="ij")
        q_points = np.array([Qx.flatten(), Qy.flatten(), Qz.flatten()])
        q_points_lab = Q_to_lab(q_points, self.Q)
        return q_points_lab, Qx.shape

    def estimate_p_Q_support(self, Q, resolution, ranges, number_of_samples):
        """Estimate the support of p_Q from samples of Q rawn in Q-system.

        The range of the output grid is determined by sampling Q and selecting
        the dimensions in each direction (x,y,z) as a multiple of the numerical
        standard deviations of this sample. The multiples are determined by the
        `ranges` parameter and apply in the local Q-coordinate-system.

        Args:
            Q (:obj:`np.ndarray`): Nominal Q-vector. shape=(3,)
            resolution (:obj:`iterable` of `float`): Reciprocal space resolution.
                Defaults to (5*1e-5,5*1e-5,5*1e-5).
            ranges (:obj:`iterable` of `float`): Number of standard deviations that will
                define the range over which p_Q is integrated. Higher multiples gives a
                larger support for p_Q. Defaults to (3,3,3).
            number_of_samples (:obj:`int`): Number of samples to use in the estimation.

        Returns:
            :obj:`iterable` of `np.ndarray`: qx_range, qy_range, qz_range given in Q-system.
        """
        rx, ry, rz = resolution
        Nx, Ny, Nz = ranges
        Q_sample = self.sample(number_of_samples)
        Q_sample_q_system = lab_to_Q(Q_sample, Q)
        mx, my, mz = np.mean(Q_sample_q_system, axis=1)
        stdx, stdy, stdz = np.std(Q_sample_q_system, axis=1)

        print("Q_sample_q_system stdx", stdx)
        print("Q_sample_q_system stdy", stdy)
        print("Q_sample_q_system stdz", stdz)

        xmin = -rx - rx * ((Nx * stdx) // rx) + mx
        xmax = +rx + rx * ((Nx * stdx) // rx) + mx

        ymin = -ry - ry * ((Ny * stdy) // ry) + my
        ymax = +ry + ry * ((Ny * stdy) // ry) + my

        zmin = -rz - rz * ((Nz * stdz) // rz) + mz
        zmax = +rz + rz * ((Nz * stdz) // rz) + mz

        qx_range = np.arange(xmin, xmax + rx, rx)
        qy_range = np.arange(ymin, ymax + ry, ry)
        qz_range = np.arange(zmin, zmax + rz, rz)

        return qx_range, qy_range, qz_range

    def _set_interpolation(self, points, p_Q, std_p_Q):
        """Set up fast interpolation on the uniform Q-system grid."""

        qx, qy, qz = points

        self._interp_origin = np.array(
            [qx[0], qy[0], qz[0]],
            dtype=np.float64,
        )

        self._interp_spacing = np.array(
            [
                qx[1] - qx[0],
                qy[1] - qy[0],
                qz[1] - qz[0],
            ],
            dtype=np.float64,
        )

        if not (
            np.allclose(np.diff(qx), self._interp_spacing[0])
            and np.allclose(np.diff(qy), self._interp_spacing[1])
            and np.allclose(np.diff(qz), self._interp_spacing[2])
        ):
            raise ValueError("Interpolation grid must be uniformly spaced.")

        self._p_Q_values = np.ascontiguousarray(
            p_Q,
            dtype=np.float64,
        )

        self._std_p_Q_values = np.ascontiguousarray(
            std_p_Q,
            dtype=np.float64,
        )

        # Retain SciPy interpolators temporarily for validation.
        self._p_Q_interp = self._rgi(points, p_Q)
        self._std_p_Q_interp = self._rgi(points, std_p_Q)

    # def _set_interpolation(self, points, p_Q, std_p_Q):
    #     """Setup regular grid interpolators defined in Q-system."""
    #     self._p_Q_interp = self._rgi(points, p_Q)
    #     self._std_p_Q_interp = self._rgi(points, std_p_Q)

    def _rgi(self, points, values):
        """Setup a regular grid interpolator."""
        return RegularGridInterpolator(
            points, values, method="linear", bounds_error=False, fill_value=0
        )

    def _get_M(self):
        """Vectorization of Poulsen 2017, dQ = M @ x.

        Returns:
            :obj:`np.ndarray`: The M matrix shape=(3,5).
        """
        th0, e0 = self.theta_0, self.eta_0
        k = 2 * np.pi / self._mean_wavelength
        M = k * np.array(
            [
                [np.cos(2 * th0) - 1, 0, 0, 0, -np.sin(2 * th0)],
                [
                    -np.sin(e0) * np.sin(2 * th0),
                    -1,
                    0,
                    np.cos(e0),
                    -np.sin(e0) * np.cos(2 * th0),
                ],
                [
                    np.cos(e0) * np.sin(2 * th0),
                    0,
                    -1,
                    np.sin(e0),
                    np.cos(e0) * np.cos(2 * th0),
                ],
            ]
        )
        return M

    def _get_theta_eta(self):
        """The Nominal Bragg angle and eta angle

        Returns:
            :obj:`iterable` of :obj:`float`: theta, eta
        """
        theta = np.arccos(self.optical_axis_0[0]) / 2.0
        yz = self.optical_axis_0[1:] / np.linalg.norm(self.optical_axis_0[1:])
        eta = np.arccos(yz[1])
        return theta, eta

    def _get_R(self):
        """The theta and eta rotation matrices such that

        Rx @ Ry @ xhat

        Returns:
            :obj:`np.ndarray`: The M matrix shape=(3,3).
        """
        s, c = np.sin(-2 * self.theta_0), np.cos(-2 * self.theta_0)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        s, c = np.sin(self.eta_0), np.cos(self.eta_0)
        Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        return Rx @ Ry


class PentaGauss(object):
    """
    Class to model a reciprocal resolution funciton. The underlying ray model uses a four Guassian distributions,
    two for the primary and and two for the secondary ray bundle. The wavelength is also modelled with a Guassian.
    In total there are five driving gaussian stochastic variables, and hence the name - PentaGauss.
    The model is approximately elastic.

    The model was proposed by Poulsen 2017.

    NOTE: This model does not implement CRL aperture trunctation nor beam angular truncation. For these features
    please refer to the TruncatedPentaGauss model. The PentaGauss model enjoys analytical solutions as a result
    of not implementing truncations of the ingoing angular distirbutions and is thus extreemely fast to compile
    and call.

    Args:
        optical_axis_0 (:obj:`np.ndarray`): Nominal diffracted ray path.shape=(3,) for zero mean angular
            spreads. To scan the CRL please use the theta_shift function.
        std_beam_horizontal (:obj:`np.ndarray`): Standard deviation of the angular horizontal spread (beam).
        std_beam_vertical (:obj:`float`): Standard deviation of the angular vertical spread (beam).
        std_CRL_horizontal (:obj:`float`):  Standard deviation of the angular horizontal spread (CRL).
        std_CRL_vertical (:obj:`np.ndarray`): Standard deviation of the angular vertical spread (CRL).
        mean_wavelength (:obj:`float`): Mean of the wavelength distribution.
        std_wavelength (:obj:`float`): Standard deviation of the wavelength distribution.
    """

    def __init__(
        self,
        optical_axis_0,
        std_beam_horizontal,
        std_beam_vertical,
        std_CRL_horizontal,
        std_CRL_vertical,
        mean_wavelength,
        std_wavelength,
    ):
        self.optical_axis_0 = optical_axis_0
        self.theta_0, self.eta_0 = self._get_theta_eta()

        # Motivation of self._cov_x[0, 0] is:
        # let dl = lamda0 - lamda
        # and epsilon = (lamda0 - lamda) / lamda such that
        # epsilon = -dl/(lamda0+dl) =(taylor 1st order)= dl * (-1 / lamda0)
        # so : E[dl] = E[-epsilon/lamda0] = 0
        # and E[dl*dl] = E[epsilon**2 / lamda0**2] = std_eps**2 / lamda0**2
        # so to first order dl is Gaussian with mean 0 and std=std_eps / lamda0
        # thus lamda = lamda0 + dl si Gaussian to first order with mean lamda0 and std=std_eps / lamda0
        # given that dl is in range 1e-4 then this is accurate to 1e-7 or 1e-8 which is faar beyond
        # the accuracy needed for most applications....

        self._cov_x = np.eye(5, 5)
        self._cov_x[0, 0] = (std_wavelength / mean_wavelength) ** 2
        self._cov_x[1, 1] = std_beam_horizontal**2
        self._cov_x[2, 2] = std_beam_vertical**2
        self._cov_x[3, 3] = std_CRL_horizontal**2
        self._cov_x[4, 4] = std_CRL_vertical**2

        self._mean_x = np.zeros((5,))
        self._mean_x[0] = 1  # this corresponds to cenetring around the nominal Q.
        self._x = MultivariateNormal(self._mean_x, self._cov_x)
        self._mean_wavelength = mean_wavelength
        self._p_Q = None

    def sample(self, number_of_samples):
        """
        Generate samples of Q vectors.

        Returns a sample in lab-coordinates by default.

        Args:
            number_of_samples (:obj:`int`): Number of samples to generate.

        Returns:
            :obj:`np.ndarray`: A sample of Q vectors of shape (3, number_of_samples).
        """
        M = self._get_M()
        return M @ self._x.sample(number_of_samples)

    def compile(self):
        """Compile the analytical expression of the reciprocal resolution function (p_Q) in lab frame."""
        M = self._get_M()
        self.cov_Q_lab = M @ self._cov_x @ M.T
        self.mean_Q_lab = M @ self._mean_x
        self._p_Q = MultivariateNormal(self.mean_Q_lab, self.cov_Q_lab)

    def __call__(self, Q_vectors, angular_crl_shifts=None):
        """
        Calculate the likelihood of a set of Q vectors.

        Args:
            Q_vectors (:obj:`np.ndarray`): A shape (3, N) array of Q vectors.

        Returns:
            :obj:`np.ndarray`: Likelihood of the given Q vectors. shape (N, )
        """
        assert len(Q_vectors.shape) == 2 and Q_vectors.shape[0] == 3
        if self._p_Q is None:
            raise ValueError(
                "The resolution function requires compiling before any calls can be made to the PDF."
            )
        else:
            if angular_crl_shifts is not None:
                dQ = self._get_Q_shifts(angular_crl_shifts)
                return self._p_Q(Q_vectors + dQ, normalise=False)
            else:
                return self._p_Q(Q_vectors, normalise=False)

    def theta_shift(self, theta):
        """Shift the mean of the resolution in theta, corresponds to moving the CRL.

        This moves the mean of the vertical dsitirbution of the CRL.

        NOTE: this will not recompile the resolution fuction. I.e the covariance
        will not change, only the mean.

        Args:
            theta (:obj:`float`): The new theta position in radians.

        """
        if self._p_Q is not None:
            M = self._get_M()
            delta_two_theta = 2 * theta - 2 * self.theta_0
            self._mean_x[4] = (
                delta_two_theta  # mean in vertical CRL position. TODO: include the eta stuff.
            )
            self.mean_Q_lab = M @ self._mean_x
            self._p_Q.mu = self.mean_Q_lab
        else:
            raise ValueError(
                "The resolution function requires compiling before any theta shifts can be introduced."
            )

    def _get_Q_shifts(self, angular_crl_shifts):
        M = self._get_M()
        dQ = M[:, -2:] @ angular_crl_shifts
        return dQ

    def _get_M(self):
        """Vectorization of Poulsen 2017, dQ = M @ x.

        Returns:
            :obj:`np.ndarray`: The M matrix shape=(3,5).
        """
        th0, e0 = self.theta_0, self.eta_0
        k = 2 * np.pi / self._mean_wavelength

        M = k * np.array(
            [
                [np.cos(2 * th0) - 1, 0, 0, 0, -np.sin(2 * th0)],
                [
                    -np.sin(e0) * np.sin(2 * th0),
                    -1,
                    0,
                    np.cos(e0),
                    -np.sin(e0) * np.cos(2 * th0),
                ],
                [
                    np.cos(e0) * np.sin(2 * th0),
                    0,
                    -1,
                    np.sin(e0),
                    np.cos(e0) * np.cos(2 * th0),
                ],
            ]
        )
        return M

    def _get_theta_eta(self):
        """The Nominal Bragg angle and eta angle

        Returns:
            :obj:`iterable` of :obj:`float`: theta, eta
        """
        theta = np.arccos(self.optical_axis_0[0]) / 2.0
        yz = self.optical_axis_0[1:] / np.linalg.norm(self.optical_axis_0[1:])
        eta = np.arccos(yz[1])
        return theta, eta

    def _get_R(self):
        """The theta and eta rotation matrices such that

        Rx @ Ry @ xhat

        Returns:
            :obj:`np.ndarray`: The M matrix shape=(3,3).
        """
        s, c = np.sin(-2 * self.theta_0), np.cos(-2 * self.theta_0)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        s, c = np.sin(self.eta_0), np.cos(self.eta_0)
        Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        return Rx @ Ry


@njit(parallel=True, nogil=True, cache=True)
def trilinear_uniform_grid(points, values, origin, spacing):
    """
    Trilinear interpolation on a uniform 3D grid.

    Parameters
    ----------
    points : (3, N) float array
        Query coordinates.
    values : (nx, ny, nz) float array
        Grid values.
    origin : (3,) float array
        Coordinate of values[0, 0, 0].
    spacing : (3,) float array
        Grid spacing.

    Returns
    -------
    out : (N,) float array

    Points outside the grid return zero, matching:
        RegularGridInterpolator(..., bounds_error=False, fill_value=0)
    """
    nx, ny, nz = values.shape
    n_points = points.shape[1]

    out = np.empty(n_points, dtype=np.float64)

    x0, y0, z0 = origin
    dx, dy, dz = spacing

    xmax = x0 + dx * (nx - 1)
    ymax = y0 + dy * (ny - 1)
    zmax = z0 + dz * (nz - 1)

    for n in prange(n_points):
        x = points[0, n]
        y = points[1, n]
        z = points[2, n]

        if np.isnan(x) or np.isnan(y) or np.isnan(z):
            out[n] = np.nan
            continue

        if x < x0 or x > xmax or y < y0 or y > ymax or z < z0 or z > zmax:
            out[n] = 0.0
            continue

        ux = (x - x0) / dx
        uy = (y - y0) / dy
        uz = (z - z0) / dz

        ix = int(np.floor(ux))
        iy = int(np.floor(uy))
        iz = int(np.floor(uz))

        # Handle points exactly on the final grid planes.
        if ix == nx - 1:
            ix = nx - 2
            tx = 1.0
        else:
            tx = ux - ix

        if iy == ny - 1:
            iy = ny - 2
            ty = 1.0
        else:
            ty = uy - iy

        if iz == nz - 1:
            iz = nz - 2
            tz = 1.0
        else:
            tz = uz - iz

        c000 = values[ix, iy, iz]
        c001 = values[ix, iy, iz + 1]
        c010 = values[ix, iy + 1, iz]
        c011 = values[ix, iy + 1, iz + 1]
        c100 = values[ix + 1, iy, iz]
        c101 = values[ix + 1, iy, iz + 1]
        c110 = values[ix + 1, iy + 1, iz]
        c111 = values[ix + 1, iy + 1, iz + 1]

        c00 = c000 + tz * (c001 - c000)
        c01 = c010 + tz * (c011 - c010)
        c10 = c100 + tz * (c101 - c100)
        c11 = c110 + tz * (c111 - c110)

        c0 = c00 + ty * (c01 - c00)
        c1 = c10 + ty * (c11 - c10)

        out[n] = c0 + tx * (c1 - c0)

    return out


class DualKentGauss(object):
    """
    Class to model a reciprocal resolution funciton. The underlying ray model uses a two Kent distributions,
    one for the primary and and one for the secondary ray bundle. The wavelength is modelled with a Guassian.
    The model is fully elastic such that wavelengths are preserved throughout scattering.

    The model was proposed by Henningsson 2024.

    Args:
        nominal_Q (:obj:`np.ndarray`): The nominal scattering vector (3,).
        gamma_CRL (:obj:`np.ndarray`): Orientation vector for the scattered ray bundle (CRL).
        kappa_CRL (:obj:`float`): Concentration parameter for the scattered ray bundle (CRL).
        beta_CRL (:obj:`float`): Ellipticity parameter for the scattered ray bundle (CRL).
        gamma_beam (:obj:`np.ndarray`): Orientation vector for the primary ray bundle.
        kappa_beam (:obj:`float`): Concentration parameter for the primary ray bundle.
        beta_beam (:obj:`float`): Ellipticity parameter for the primary ray bundle.
        mean_wavelength (:obj:`float`): Mean of the wavelength distribution.
        std_wavelength (:obj:`float`): Standard deviation of the wavelength distribution.
    """

    def __init__(
        self,
        gamma_CRL,
        kappa_CRL,
        beta_CRL,
        gamma_beam,
        kappa_beam,
        beta_beam,
        mean_wavelength,
        std_wavelength,
    ):
        self.primary_ray_direction = Kent(gamma_beam, kappa_beam, beta_beam)
        self.secondary_ray_direction = Kent(gamma_CRL, kappa_CRL, beta_CRL)
        self.ray_wavelength = Normal(mean_wavelength, std_wavelength)
        self.Q = None

    def compile(
        self,
        Q,
        resolution=(5 * 1e-4, 5 * 1e-4, 5 * 1e-4),
        ranges=(3, 3, 3),
        number_of_samples=25000,
    ):
        """Compile an approximation of the reciprocal resolution function (p_Q) in lab frame.

        This function will run monte-carlo integration for p_Q at a series of selected
        grid-points that are equidistantly spaced by the bin width `resolution`.

        The range of the query grid is determined by sampling the distirbution of Q and
        selecting the dimensions in each direction (x,y,z) as a multiple of the numerical
        standard deviations of the sample. The multiples are determined by the `ranges`
        parameter and apply in the local Q-coordinate-system.

        NOTE: The resolution function as interfaced in the __call__ method of this class
        (and as given by the attribute `p_Q`) is always given in the lab-system. The grid
        over which p_Q is internally defined is however taken in the Q-system, since, in
        general, p_Q is observed to have a close to diagonal covariacne in the Q-system.
        Coordinate conversions to map an input lab-vector to Q-system is handled internally.

        Args:
            Q (:obj:`np.ndarray`): Nominal Q-vector. shape=(3,)
            resolution (:obj:`iterable` of `float`): Reciprocal space resolution.
                Defaults to (5*1e-5,5*1e-5,5*1e-5).
            ranges (:obj:`iterable` of `float`): Number of standard deviations that will
                define the range over which p_Q is integrated. Higher multiples gives a
                larger support for p_Q. Defaults to (3,3,3).
            number_of_samples (:obj:`int`): Number of samples per integration point. More samples
                gives less error in p_Q at the cost of computational speed. Defaults to 25000.

        """
        self.Q = Q
        q_ranges = self.estimate_p_Q_support(
            Q, resolution, ranges, number_of_samples=20000
        )
        q_points_lab, grid_shape = self._get_integration_points(q_ranges)
        voxel_volume = np.prod(
            resolution
        )  # the volume associated to an integration point
        p_Q, std_p_Q = self._monte_carlo_integrate(
            q_points_lab, voxel_volume, number_of_samples
        )
        self.p_Q_flat = p_Q[:]
        self.p_Q = p_Q.reshape(grid_shape)
        self.std_p_Q = std_p_Q.reshape(grid_shape)
        self._integration_points = q_points_lab  # for testing purpose we store these.
        self._set_interpolation(q_ranges, self.p_Q, self.std_p_Q)

    def _get_integration_points(self, q_ranges):
        """Get lab-cooridnate integration points for MC integration from Q-system grid.

        Args:
            q_ranges (:obj:`iterable` of `np.ndarray`): qx_range, qy_range, qz_range given in Q-system.
                These are the monotonically increasing arrays that define the integration grid in the
                local Q-system.

        Returns:
            :obj:`tuple` of `np.ndarray` and `tuple`: q_points_lab, Qx.shape integration
                points and the 3d shape of the grid. q_points_lab.shape = (3,N).

        """
        Qx, Qy, Qz = np.meshgrid(*q_ranges, indexing="ij")
        q_points = np.array([Qx.flatten(), Qy.flatten(), Qz.flatten()])
        q_points_lab = Q_to_lab(q_points, self.Q)
        return q_points_lab, Qx.shape

    def estimate_p_Q_support(self, Q, resolution, ranges, number_of_samples):
        """Estimate the support of p_Q from samples of Q rawn in Q-system.

        The range of the output grid is determined by sampling Q and selecting
        the dimensions in each direction (x,y,z) as a multiple of the numerical
        standard deviations of this sample. The multiples are determined by the
        `ranges` parameter and apply in the local Q-coordinate-system.

        Args:
            Q (:obj:`np.ndarray`): Nominal Q-vector. shape=(3,)
            resolution (:obj:`iterable` of `float`): Reciprocal space resolution.
                Defaults to (5*1e-5,5*1e-5,5*1e-5).
            ranges (:obj:`iterable` of `float`): Number of standard deviations that will
                define the range over which p_Q is integrated. Higher multiples gives a
                larger support for p_Q. Defaults to (3,3,3).
            number_of_samples (:obj:`int`): Number of samples to use in the estimation.

        Returns:
            :obj:`iterable` of `np.ndarray`: qx_range, qy_range, qz_range given in Q-system.
        """
        rx, ry, rz = resolution
        Nx, Ny, Nz = ranges
        Q_sample = self.sample(number_of_samples)
        Q_sample_q_system = lab_to_Q(Q_sample, Q)
        mx, my, mz = np.mean(Q_sample_q_system, axis=1)
        stdx, stdy, stdz = np.std(Q_sample_q_system, axis=1)

        xmin = -rx - rx * ((Nx * stdx) // rx) + mx
        xmax = +rx + rx * ((Nx * stdx) // rx) + mx

        ymin = -ry - ry * ((Ny * stdy) // ry) + my
        ymax = +ry + ry * ((Ny * stdy) // ry) + my

        zmin = -rz - rz * ((Nz * stdz) // rz) + mz
        zmax = +rz + rz * ((Nz * stdz) // rz) + mz

        qx_range = np.arange(xmin, xmax + rx, rx)
        qy_range = np.arange(ymin, ymax + ry, ry)
        qz_range = np.arange(zmin, zmax + rz, rz)

        return qx_range, qy_range, qz_range

    def _set_interpolation(self, points, p_Q, std_p_Q):
        """Setup regular grid interpolators defined in Q-system."""
        self._p_Q_interp = self._rgi(points, p_Q)
        self._std_p_Q_interp = self._rgi(points, std_p_Q)

    def _rgi(self, points, values):
        """Setup a regular grid interpolator."""
        return RegularGridInterpolator(
            points, values, method="linear", bounds_error=False, fill_value=0
        )

    def _monte_carlo_integrate(self, q_points_lab, dv, number_of_samples):
        """Integrate for p_Q at a series of locations.

        The integration takes place in lab frame and is either driven by sampling from the
        CRL acceptance or the beam divergence. I.e either from the primary or from the
        secondary ray distribution. Sampling from the distirbution with highest concentration
        of these two ensures that few samples are wasted (i.e have zero probablility).

        To preserve numerical precision probabilities are evalueated in log-base and only
        after the compound ray log-probability has been formed will exponents be taken.
        Contributions that are significantly smaller than the floating point precision
        will be ignored for performance reasons.

        NOTE: For performace resons the same samples are used in each integration bin.

        TODO: to accelerate the integration an auxiliary disitrbution q(x) could be introduced.
            This would amount to re-sampling for different integration bins to reduce the number
            of wasted samples. On the other hand this requires more sampling.

        TODO: Alternatively, we may look into numerical integration schemes (i.e q uniform).
            we could rotate the integration grid based on the mode of the compound ray
            distribution (analytical analysis required).

        Args:
            q_points_lab (:obj:`np.ndarray`): Integration points in lab-frame. shape=(3,N)
            dv (:obj:`float`): The volume associated to each integration point.
            number_of_samples (:obj:`int`): Number of samples per integration point.

        Returns:
            :obj:`np.ndarray`: p_Q integrated values normalied into a PDF. shape=(m,n,o).
            :obj:`np.ndarray`: std_p_Q estimated standard deviation of the error assicated to each
                integration bin. shape=(m,n,o).

        """

        if self.secondary_ray_direction.kappa > self.primary_ray_direction.kappa:
            prior = "CRL"
            ghat = self.secondary_ray_direction.sample(number_of_samples)
            mode = self.primary_ray_direction.gamma[:, 0]
            log_norm_const = self.primary_ray_direction(mode, normalise=False, log=True)
        else:
            prior = "beam"
            nhat = self.primary_ray_direction.sample(number_of_samples)
            mode = self.secondary_ray_direction.gamma[:, 0]
            log_norm_const = self.secondary_ray_direction(
                mode, normalise=False, log=True
            )

        Qnorms = np.linalg.norm(q_points_lab, axis=0)
        dmap = (2 * np.pi) / Qnorms

        p_Q = np.zeros((q_points_lab.shape[1],))
        std_p_Q = np.zeros((q_points_lab.shape[1],))

        for i, Q_probe in enumerate(q_points_lab.T):
            d = dmap[i]

            if prior == "CRL":
                nhat = self._get_nhat(ghat, d, Q_probe)
                log_p_sample = self.primary_ray_direction(
                    nhat, normalise=False, log=True
                )
            elif prior == "beam":
                ghat = self._get_ghat(nhat, d, Q_probe)
                log_p_sample = self.secondary_ray_direction(
                    ghat, normalise=False, log=True
                )

            log_c_p = log_p_sample - log_norm_const
            lamda = self._get_wavelength(nhat, d, Q_probe)
            log_p_A = self.ray_wavelength(lamda, normalise=False, log=True)

            p_tot_log = log_c_p + log_p_A

            # some safe removals to save the costly exp call
            samples_to_keep = p_tot_log > np.log((1 / number_of_samples) * 1e-16)

            # conclusion is that we waste a lot of samples at the edge of the dist.
            # DEBUG = True
            # if DEBUG:
            #    nbr_bad_samples = np.sum(~samples_to_keep)
            #    ratio_keep = nbr_bad_samples/number_of_samples
            #    Qres = np.linalg.norm(self.Q-Q_probe)
            #    print( 'Bin nbr: ', i, ' with ', ratio_keep, ' wasted samples, Qres is ',  Qres)

            if np.sum(samples_to_keep) == 0:
                continue
            else:
                p_tot = self._exp(p_tot_log)  # [samples_to_keep] )
                p_Q[i] = np.sum(p_tot) / number_of_samples
                std_p_Q[i] = np.std(p_tot) / np.sqrt(number_of_samples)

        norm_const = np.sum(p_Q * dv)
        p_Q = p_Q / norm_const
        std_p_Q = std_p_Q / norm_const

        return p_Q, std_p_Q

    def _get_wavelength(self, nhat, d, Q):
        """Find the wavelength required for diffraction."""
        return -(d * d / np.pi) * nhat.T @ Q

    def _get_ghat(self, nhat, d, Q):
        """Find the scattering direction required for diffraction."""
        return (np.eye(3, 3) - ((d * d) / (2 * np.pi * np.pi)) * np.outer(Q, Q)) @ nhat

    def _get_nhat(self, ghat, d, Q):
        """Find the incident ray direction required for diffraction."""
        return (
            np.linalg.inv(
                np.eye(3, 3) - ((d * d) / (2 * np.pi * np.pi)) * np.outer(Q, Q)
            )
            @ ghat
        )

    def _exp(self, a):
        """ """
        return np.exp(a)

    def __call__(self, Q_vectors, angular_crl_shifts=None):
        """
        Calculate the likelihood of a set of Q vectors.

        Args:
            Q_vectors (:obj:`np.ndarray`): A shape (3, N) array of Q vectors.

        Returns:
            :obj:`np.ndarray`: Likelihood of the given Q vectors. shape (N, )
        """
        assert len(Q_vectors.shape) == 2 and Q_vectors.shape[0] == 3
        if self.Q is None:
            raise ValueError(
                "The resolution function requires compiling before any calls can be made to the PDF."
            )
        else:
            Q_vectors_q_system = lab_to_Q(Q_vectors, self.Q)
            return self._p_Q_interp(Q_vectors_q_system.T)

    def sample(self, number_of_samples):
        """
        Generate samples of Q vectors using the Henningsson method.

        Returns a sample in lab-coordinates by default.

        Args:
            number_of_samples (:obj:`int`): Number of samples to generate.

        Returns:
            :obj:`np.ndarray`: A sample of Q vectors of shape (3, number_of_samples).
        """
        nhat = self.primary_ray_direction.sample(number_of_samples)
        ghat = self.secondary_ray_direction.sample(number_of_samples)
        lamda = self.ray_wavelength.sample(number_of_samples)

        Qhat = (-nhat + ghat) / np.linalg.norm(-nhat + ghat, axis=0)
        d = lamda / (-2 * np.sum(Qhat * nhat, axis=0))
        Qsample = 2 * np.pi * Qhat / d

        return Qsample


if __name__ == "__main__":
    U = np.eye(3, 3)
    a = b = c = 4.0493
    unit_cell = [a, b, c, 90.0, 90.0, 90.0]
    lambda_0 = 0.71
    energy_0 = laue.angstrom_to_keV(lambda_0)
    sigma_e = (1.4 * 1e-4) / (2 * np.sqrt(2 * np.log(2)))
    hkl = np.array([0, 0, 2])

    from dfxm import experiment

    goni = experiment.Goniometer(U, unit_cell, energy_0)
    goni.bring_to_bragg(hkl)
    Q = goni.U @ goni.B @ hkl
    d_0 = (2 * np.pi) / np.linalg.norm(Q)
    theta_0 = np.arcsin(lambda_0 / (2 * d_0))
    k_0 = 2 * np.pi / lambda_0

    # Beam divergence params
    gamma_N = np.eye(3, 3)
    desired_FWHM_N = 0.53 * 1e-3
    kappa_N = np.log(2) / (1 - np.cos((desired_FWHM_N) / 2.0))
    beta_N = 0

    # Beam wavelength params
    epsilon = np.random.normal(0, sigma_e, size=(20000,))
    random_energy = energy_0 + epsilon * energy_0
    sigma_lambda = laue.keV_to_angstrom(random_energy).std()
    mu_lambda = lambda_0

    # CRL acceptance params
    gamma_C = goni.imaging_system
    desired_FWHM_C = 0.731 * 1e-3
    kappa_C = np.log(2) / (1 - np.cos((desired_FWHM_C) / 2.0))
    beta_C = 0

    # res = DualKentGauss(
    #                 gamma_C,
    #                 kappa_C,
    #                 beta_C,
    #                 gamma_N,
    #                 kappa_N,
    #                 beta_N,
    #                 mu_lambda,
    #                 sigma_lambda,
    #                 )

    res1 = PentaGauss(
        goni.optical_axis,
        desired_FWHM_N / (2 * np.sqrt(2 * np.log(2))),
        desired_FWHM_N / (2 * np.sqrt(2 * np.log(2))),
        desired_FWHM_C / (2 * np.sqrt(2 * np.log(2))),
        desired_FWHM_C / (2 * np.sqrt(2 * np.log(2))),
        mu_lambda,
        sigma_lambda,
    )

    physical_aperture = 2 * desired_FWHM_C / (2 * np.sqrt(2 * np.log(2)))
    D = 0.477 * 1e-3
    d1 = 0.274
    physical_aperture = np.arctan(D / (2 * d1))
    print(
        physical_aperture,
        desired_FWHM_C / (2 * np.sqrt(2 * np.log(2))),
        desired_FWHM_N / 2.0,
    )

    xray_params = {
        "std_beam_horizontal": 1e-5 / (2 * np.sqrt(2 * np.log(2))),
        "lower_bound_beam_horizontal": None,
        "upper_bound_beam_horizontal": None,
        "std_beam_vertical": desired_FWHM_N / (2 * np.sqrt(2 * np.log(2))),
        "lower_bound_beam_vertical": -desired_FWHM_N / 2.0,
        "upper_bound_beam_vertical": desired_FWHM_N / 2.0,
        "std_CRL_horizontal": desired_FWHM_C / (2 * np.sqrt(2 * np.log(2))),
        "lower_bound_CRL_horizontal": -physical_aperture,
        "upper_bound_CRL_horizontal": physical_aperture,
        "std_CRL_vertical": desired_FWHM_C / (2 * np.sqrt(2 * np.log(2))),
        "lower_bound_CRL_vertical": -physical_aperture,
        "upper_bound_CRL_vertical": physical_aperture,
        "std_energy_shift": sigma_e,
        "lower_bound_energy_shift": None,
        "upper_bound_energy_shift": None,
    }

    for k in xray_params:
        print(k, xray_params[k])
    print(desired_FWHM_N / 2.0)

    res2 = TruncatedPentaGauss(
        goni.optical_axis,
        mu_lambda,
        xray_params,
    )

    if 1:
        plt.style.use("dark_background")
        import cProfile
        import pstats
        import time

        pr = cProfile.Profile()
        pr.enable()
        t1 = time.perf_counter()

        res2.compile(Q, resolution=(5 * 1e-4, 5 * 1e-4, 5 * 1e-4))
        # res.compile(Q)

        t2 = time.perf_counter()
        pr.disable()
        pr.dump_stats("tmp_profile_dump")
        ps = pstats.Stats("tmp_profile_dump").strip_dirs().sort_stats("cumtime")
        ps.print_stats(15)
        print("\n\nCPU time is : ", t2 - t1, "s")

        def _plot(field, cmap, title):
            plt.figure()
            plt.imshow(field, cmap=cmap)
            plt.title(title)
            plt.colorbar()

        def cartesian_slices(field):
            a = field[field.shape[0] // 2, :, :]
            b = field[:, field.shape[1] // 2, :]
            c = field[:, :, field.shape[2] // 2]
            return a, b, c

        # a, b, c = cartesian_slices(res2.p_Q)
        # #d, e, f = cartesian_slices(res2.std_p_Q)
        # for i,field in enumerate((a, b, c)):
        #     _plot(field, cmap='viridis', title='yz xz xy'.split(' ')[i])
        # for field in (d, e, f):
        #     _plot(field, cmap='magma')
        # for field in (d/a, e/b, f/c):
        #     _plot(field, cmap='jet')

        Qs = np.zeros((3, 256))
        Qs[0, :] = np.linspace(Q[0] - 25 * 1e-4, Q[0] + 25 * 1e-4, Qs.shape[1])
        Qs[1, :] = Q[1]
        Qs[2, :] = Q[2]

        res1.compile()
        p_Q = res1(Qs)
        plt.figure()
        plt.plot(Qs[0, :] - Q[0], p_Q, "ro--")
        p_Q = res2(Qs)
        plt.plot(Qs[0, :] - Q[0], p_Q, "ko--")

        Qq = lab_to_Q(Q, Q)
        qx = np.linspace(Qq[0] - 95 * 1e-4, Qq[0] + 95 * 1e-4, 64)
        qy = np.linspace(Qq[1] - 95 * 1e-4, Qq[1] + 95 * 1e-4, 64)
        qz = np.linspace(Qq[2] - 95 * 1e-4, Qq[2] + 95 * 1e-4, 64)
        Qx, Qy, Qz = np.meshgrid(qx, qy, qz, indexing="ij")
        points = np.array([Qx.flatten(), Qy.flatten(), Qz.flatten()])

        def projection_slices(field):
            a = field.sum(axis=0)
            b = field.sum(axis=1)
            c = field.sum(axis=2)
            return a, b, c

        p_Q, std_p_Q = res2(Q_to_lab(points, Q), error_estimate=True)

        p_Q = p_Q.reshape(Qy.shape)
        std_p_Q = std_p_Q.reshape(Qy.shape)

        a, b, c = projection_slices(p_Q)
        for i, field in enumerate((a, b, c)):
            _plot(field, cmap="viridis", title="yz xz xy".split(" ")[i])

        a, b, c = projection_slices(std_p_Q)
        for i, field in enumerate((a, b, c)):
            _plot(field, cmap="magma", title="yz xz xy".split(" ")[i])

        p_Q = res1(Q_to_lab(points, Q)).reshape(Qy.shape)
        a, b, c = projection_slices(p_Q)
        for i, field in enumerate((a, b, c)):
            _plot(field, cmap="jet", title="yz xz xy".split(" ")[i])

    samples1 = res1.sample(number_of_samples=10000)
    samples2 = res2.sample(number_of_samples=10000)

    fig = plt.figure(figsize=(8, 8))

    samples1 -= np.mean(samples1, axis=1).reshape(3, 1)
    qx, qy, qz = lab_to_Q(samples1, Q)
    # qx /= np.linalg.norm(Q)
    # qy /= np.linalg.norm(Q)
    # qz /= np.linalg.norm(Q)

    alpha = 0.1

    print("Cov", (np.cov(np.array([qx, qy, qz])) * 1e6).round(3))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(0.2 + qx * 1e3, qy * 1e3, qz * 1e3, alpha=alpha, label="No-truncation")
    ax.scatter(0.2 + qx * 1e3, qy * 1e3, -15, alpha=alpha)
    ax.scatter(0.2 + qx * 1e3, 15, qz * 1e3, alpha=alpha)
    ax.scatter(5.5, qy * 1e3, qz * 1e3, alpha=alpha)

    samples2 -= np.mean(samples2, axis=1).reshape(3, 1)
    qx, qy, qz = lab_to_Q(samples2, Q)
    # qx /= np.linalg.norm(Q)
    # qy /= np.linalg.norm(Q)
    # qz /= np.linalg.norm(Q)

    print("Cov", (np.cov(np.array([qx, qy, qz])) * 1e6).round(3))
    ax.scatter(-5.3 + qx * 1e3, qy * 1e3, qz * 1e3, alpha=alpha, label="Truncation")
    ax.scatter(-5.3 + qx * 1e3, qy * 1e3, -15, alpha=alpha)
    ax.scatter(-5.3 + qx * 1e3, 15, qz * 1e3, alpha=alpha)
    ax.scatter(5.5, qy * 1e3, qz * 1e3, alpha=alpha)

    ax.set_xlabel("$q_{rock}$")
    ax.set_ylabel("$q_{roll}$")
    ax.set_zlabel("$q_{||}$")
    ax.set_xlim([5.5, -8])
    ax.set_ylim([15.3, -15.3])
    ax.set_zlim([-15.3, 15.3])
    ax.view_init(elev=20, azim=59)

    ax.legend()

    plt.show()
