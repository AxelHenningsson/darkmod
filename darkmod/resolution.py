import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from darkmod import laue
from darkmod.distribution import Kent, Normal, MultivariateNormal, TruncatedNormal, MultivariateTruncatedNormal
from darkmod.transforms import lab_to_Q, Q_to_lab

class TruncatedPentaGauss(object):
    """This model is a generalisation of the PentaGauss model in that each of the gaussian variables
    can be clipped by an upper and lower bound, allowing, for instance, the simulation of a fixed
    aperture CRL.

    This is similar to Poulsen 2021.

    NOTE the CRL is implicitly treated to have a square opening - not a round.

    Args:
        optical_axis (:obj:`float`): Nominal diffracted ray path, shape=(3,).
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

    def __init__(self, optical_axis, mean_wavelength, xray_parameters):

        self._par = self._extract_parameters(xray_parameters) 
        self.optical_axis = optical_axis
        self._mean_wavelength = mean_wavelength
        self.Q = None

        self._cov_x = np.eye(5, 5)
        self._cov_x[0, 0] = self._par["std_energy_shift"]**2
        self._cov_x[1, 1] = self._par["std_beam_horizontal"]**2
        self._cov_x[2, 2] = self._par["std_beam_vertical"]**2
        self._cov_x[3, 3] = self._par["std_CRL_horizontal"]**2
        self._cov_x[4, 4] = self._par["std_CRL_vertical"]**2

        self._lower_bound_x = np.array([
            self._par["lower_bound_energy_shift"],
            self._par["lower_bound_beam_horizontal"],
            self._par["lower_bound_beam_vertical"],
            self._par["lower_bound_CRL_horizontal"],
            self._par["lower_bound_CRL_vertical"],
        ]).reshape(5,1)

        self._upper_bound_x = np.array([
            self._par["upper_bound_energy_shift"],
            self._par["upper_bound_beam_horizontal"],
            self._par["upper_bound_beam_vertical"],
            self._par["upper_bound_CRL_horizontal"],
            self._par["upper_bound_CRL_vertical"],
        ]).reshape(5,1)

        self._x = MultivariateTruncatedNormal(np.zeros((5,)), 
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
            "upper_bound_energy_shift"
        ]
        new_pars = {}
        for key in keys:
            if key not in params.keys():
                params[key] = None
            else:
                if key.startswith("std"):
                    assert params[key]>0
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
        R = self._get_R()
        M = self._get_M()
        k = 2*np.pi / self._mean_wavelength
        x = self._x.sample(number_of_samples)
        k_in_0 = k * np.array([1, 0, 0])
        k_out_0 = R @ k_in_0
        Qsample = (k_out_0 - k_in_0)[:, np.newaxis] + M @ x
        return Qsample

    def compile(self, Q, resolution=(5*1e-4, 5*1e-4, 5*1e-4), ranges=(5, 5, 5), number_of_samples=None):
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
                gives less error in p_Q at the cost of computational speed. Defaults to None in which
                the number of samples are choosen to correspond to 100 samples per bin up to a max of 2*1e7
                samples.

        """
        self.Q = Q
        q_ranges = self.estimate_p_Q_support(Q, resolution, ranges, number_of_samples=200000)
        q_points_lab, grid_shape = self._get_integration_points(q_ranges)
        voxel_volume = np.prod(resolution) # the volume associated to an integration point

        bins_edges = (
                      self._bin_centers_to_edges(q_ranges[0]),
                      self._bin_centers_to_edges(q_ranges[1]),
                      self._bin_centers_to_edges(q_ranges[2])
                      )
        
        nbins = len(bins_edges[0])*len(bins_edges[1])*len(bins_edges[2])

        if number_of_samples is None:
            number_of_samples = np.min( [100 * nbins, 20000000] )
    
        p_Qs = []
        for n in range(5):
            sample = self.sample( number_of_samples // 5 )
            sample = lab_to_Q(sample, self.Q)
            p_Q, edges = np.histogramdd( sample.T, bins=bins_edges, density=True)
            p_Qs.append(p_Q)

        self.p_Q = np.mean(p_Qs, axis=0)
        self.std_p_Q = np.std(p_Qs, axis=0)

        # TODO: this seems like a reasonable KDE like thing to go for...
        #from scipy.ndimage import gaussian_filter
        #self.p_Q = gaussian_filter(self.p_Q, sigma=[r/3. for r in resolution])

        self._integration_points = q_points_lab # for testing purpose we store these.
        self._set_interpolation(q_ranges, self.p_Q, self.std_p_Q)

    def _bin_centers_to_edges(self, bin_centers):
        edges = (bin_centers[1:] + bin_centers[:-1]) / 2
        first_edge = bin_centers[0] - (bin_centers[1] - bin_centers[0]) / 2
        last_edge = bin_centers[-1] + (bin_centers[-1] - bin_centers[-2]) / 2
        return np.concatenate([[first_edge], edges, [last_edge]])

    def __call__(self, Q_vectors, error_estimate=False):
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
        assert len(Q_vectors.shape)==2 and Q_vectors.shape[0]==3
        if self.Q is None:
            raise ValueError('The resolution function requires compiling before any calls can be made to the PDF.')
        else:
            Q_vectors_q_system = lab_to_Q(Q_vectors, self.Q)
            p_Q = self._p_Q_interp(Q_vectors_q_system.T)
            if error_estimate: 
                std_p_Q = self._std_p_Q_interp(Q_vectors_q_system.T)
                return p_Q, std_p_Q
            else: 
                return p_Q

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
        Qx, Qy, Qz = np.meshgrid( *q_ranges, indexing='ij' )
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

        xmin = - rx - rx*((Nx*stdx)//rx) + mx
        xmax = + rx + rx*((Nx*stdx)//rx) + mx

        ymin = - ry - ry*((Ny*stdy)//ry) + my
        ymax = + ry + ry*((Ny*stdy)//ry) + my

        zmin = - rz - rz*((Nz*stdz)//rz) + mz
        zmax = + rz + rz*((Nz*stdz)//rz) + mz

        qx_range = np.arange(xmin, xmax + rx , rx)
        qy_range = np.arange(ymin, ymax + ry , ry)
        qz_range = np.arange(zmin, zmax + rz , rz)

        return qx_range, qy_range, qz_range

    def _set_interpolation(self, points, p_Q, std_p_Q):
        """Setup regular grid interpolators defined in Q-system.
        """
        self._p_Q_interp = self._rgi(points, p_Q)
        self._std_p_Q_interp = self._rgi(points, std_p_Q)

    def _rgi(self, points, values):
        """Setup a regular grid interpolator.
        """
        return RegularGridInterpolator(points,
                                       values,
                                       method='linear',
                                       bounds_error=False,
                                       fill_value=0)

    def _get_M(self):
        """Vectorization of Poulsen 2017, dQ = M @ x.

        Returns:
            :obj:`np.ndarray`: The M matrix shape=(3,5).
        """
        theta = np.arccos(self.optical_axis[0]) / 2.
        yz = self.optical_axis[1:] / np.linalg.norm(self.optical_axis[1:])
        eta = np.arccos(yz[1])
        k = 2*np.pi / self._mean_wavelength
        M = k * np.array([
            [ np.cos(2 * theta)-1           ,   0 ,    0 ,      0       ,  -np.sin(2 * theta)             ],
            [-np.sin(eta)*np.sin(2 * theta) ,  -1 ,    0 ,  np.cos(eta) ,  -np.sin(eta)*np.cos(2 * theta) ],
            [ np.cos(eta)*np.sin(2 * theta) ,   0 ,   -1 ,  np.sin(eta) ,   np.cos(eta)*np.cos(2 * theta) ]
        ])
        return M

    def _get_theta_eta(self):
        """The Nominal Bragg angle and eta angle

        Returns:
            :obj:`iterable` of :obj:`float`: theta, eta
        """
        theta = np.arccos(self.optical_axis[0]) / 2.
        yz = self.optical_axis[1:] / np.linalg.norm(self.optical_axis[1:])
        eta = np.arccos(yz[1])
        return theta, eta

    def _get_R(self):
        """The theta and eta rotation matrices such that

        Rx @ Ry @ xhat

        Returns:
            :obj:`np.ndarray`: The M matrix shape=(3,3).
        """
        theta, eta = self._get_theta_eta()
        s, c = np.sin(-2*theta), np.cos(-2*theta)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        s, c = np.sin(eta), np.cos(eta)
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
        optical_axis (:obj:`np.ndarray`): Nominal diffracted ray path.shape=(3,)
        std_beam_horizontal (:obj:`np.ndarray`): Standard deviation of the angular horizontal spread (beam).
        std_beam_vertical (:obj:`float`): Standard deviation of the angular vertical spread (beam).
        std_CRL_horizontal (:obj:`float`):  Standard deviation of the angular horizontal spread (CRL).
        std_CRL_vertical (:obj:`np.ndarray`): Standard deviation of the angular vertical spread (CRL).
        mean_wavelength (:obj:`float`): Mean of the wavelength distribution.
        std_wavelength (:obj:`float`): Standard deviation of the wavelength distribution.
    """

    def __init__(self,
                 optical_axis,
                 std_beam_horizontal,
                 std_beam_vertical,
                 std_CRL_horizontal,
                 std_CRL_vertical,
                 mean_wavelength,
                 std_wavelength,
                 ):
        self.optical_axis = optical_axis

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
        self._cov_x[0, 0] = (std_wavelength / mean_wavelength)**2
        self._cov_x[1, 1] = std_beam_horizontal**2
        self._cov_x[2, 2] = std_beam_vertical**2
        self._cov_x[3, 3] = std_CRL_horizontal**2
        self._cov_x[4, 4] = std_CRL_vertical**2

        self._mean_x = np.zeros((5,))
        self._mean_x[0] = 1 # thi corresponds to cenetring around the nominal Q.
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
        """Compile the analytical expression of the reciprocal resolution function (p_Q) in lab frame.
        """
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
        assert len(Q_vectors.shape)==2 and Q_vectors.shape[0]==3
        if self._p_Q is None:
            raise ValueError('The resolution function requires compiling before any calls can be made to the PDF.')
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
            theta0 = np.arccos(self.optical_axis[0]) / 2.
            delta_two_theta = 2*theta - 2*theta0
            self._mean_x[4] =  delta_two_theta
            self.mean_Q_lab = M @ self._mean_x
            self._p_Q.mu = self.mean_Q_lab
        else:
            raise ValueError('The resolution function requires compiling before any theta shifts can be introduced.')

    def _get_Q_shifts(self, angular_crl_shifts):
        M = self._get_M()
        dQ = M[:,-2:] @ angular_crl_shifts
        return dQ

    def _get_M(self):
        """Vectorization of Poulsen 2017, dQ = M @ x.

        Returns:
            :obj:`np.ndarray`: The M matrix shape=(3,5).
        """
        theta = np.arccos(self.optical_axis[0]) / 2.
        yz = self.optical_axis[1:] / np.linalg.norm(self.optical_axis[1:])
        eta = np.arccos(yz[1])
        k = 2*np.pi / self._mean_wavelength
        M = k * np.array([
            [ np.cos(2 * theta)-1           ,   0 ,    0 ,      0       ,  -np.sin(2 * theta)             ],
            [-np.sin(eta)*np.sin(2 * theta) ,  -1 ,    0 ,  np.cos(eta) ,  -np.sin(eta)*np.cos(2 * theta) ],
            [ np.cos(eta)*np.sin(2 * theta) ,   0 ,   -1 ,  np.sin(eta) ,   np.cos(eta)*np.cos(2 * theta) ]
        ])
        return M

    def _get_theta_eta(self):
        """The Nominal Bragg angle and eta angle

        Returns:
            :obj:`iterable` of :obj:`float`: theta, eta
        """
        theta = np.arccos(self.optical_axis[0]) / 2.
        yz = self.optical_axis[1:] / np.linalg.norm(self.optical_axis[1:])
        eta = np.arccos(yz[1])
        return theta, eta

    def _get_R(self):
        """The theta and eta rotation matrices such that

        Rx @ Ry @ xhat

        Returns:
            :obj:`np.ndarray`: The M matrix shape=(3,3).
        """
        theta, eta = self._get_theta_eta()
        s, c = np.sin(-2*theta), np.cos(-2*theta)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        s, c = np.sin(eta), np.cos(eta)
        Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
        return Rx @ Ry

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

    def __init__(self,
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

    def compile(self, Q, resolution=(5*1e-4, 5*1e-4, 5*1e-4), ranges=(3, 3, 3), number_of_samples=25000):
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
        q_ranges = self.estimate_p_Q_support(Q, resolution, ranges, number_of_samples=20000)
        q_points_lab, grid_shape = self._get_integration_points(q_ranges)
        voxel_volume = np.prod(resolution) # the volume associated to an integration point
        p_Q, std_p_Q = self._monte_carlo_integrate(q_points_lab, voxel_volume, number_of_samples)
        self.p_Q_flat = p_Q[:]
        self.p_Q = p_Q.reshape(grid_shape)
        self.std_p_Q = std_p_Q.reshape(grid_shape)
        self._integration_points = q_points_lab # for testing purpose we store these.
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
        Qx, Qy, Qz = np.meshgrid( *q_ranges, indexing='ij' )
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

        xmin = - rx - rx*((Nx*stdx)//rx) + mx
        xmax = + rx + rx*((Nx*stdx)//rx) + mx

        ymin = - ry - ry*((Ny*stdy)//ry) + my
        ymax = + ry + ry*((Ny*stdy)//ry) + my

        zmin = - rz - rz*((Nz*stdz)//rz) + mz
        zmax = + rz + rz*((Nz*stdz)//rz) + mz

        qx_range = np.arange(xmin, xmax + rx, rx)
        qy_range = np.arange(ymin, ymax + ry, ry)
        qz_range = np.arange(zmin, zmax + rz, rz)

        return qx_range, qy_range, qz_range

    def _set_interpolation(self, points, p_Q, std_p_Q):
        """Setup regular grid interpolators defined in Q-system.
        """
        self._p_Q_interp = self._rgi(points, p_Q)
        self._std_p_Q_interp = self._rgi(points, std_p_Q)

    def _rgi(self, points, values):
        """Setup a regular grid interpolator.
        """
        return RegularGridInterpolator(points,
                                       values,
                                       method='linear',
                                       bounds_error=False,
                                       fill_value=0)

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
            prior = 'CRL'
            ghat = self.secondary_ray_direction.sample(number_of_samples)
            mode = self.primary_ray_direction.gamma[:, 0]
            log_norm_const = self.primary_ray_direction(mode, normalise=False, log=True)
        else:
            prior = 'beam'
            nhat = self.primary_ray_direction.sample(number_of_samples)
            mode = self.secondary_ray_direction.gamma[:, 0]
            log_norm_const = self.secondary_ray_direction(mode, normalise=False, log=True)

        Qnorms = np.linalg.norm(q_points_lab, axis=0)
        dmap = (2*np.pi)/Qnorms

        p_Q = np.zeros((q_points_lab.shape[1], ))
        std_p_Q = np.zeros((q_points_lab.shape[1], ))

        for i, Q_probe in enumerate(q_points_lab.T):

            d = dmap[i]

            if prior == 'CRL':
                nhat = self._get_nhat(ghat, d, Q_probe)
                log_p_sample = self.primary_ray_direction(nhat, normalise=False, log=True)
            elif prior == 'beam':
                ghat = self._get_ghat(nhat, d, Q_probe)
                log_p_sample = self.secondary_ray_direction(ghat, normalise=False, log=True)

            log_c_p = log_p_sample - log_norm_const
            lamda = self._get_wavelength(nhat, d, Q_probe)
            log_p_A = self.ray_wavelength(lamda, normalise=False, log=True)

            p_tot_log = log_c_p + log_p_A

            # some safe removals to save the costly exp call
            samples_to_keep = p_tot_log > np.log( (1/number_of_samples) * 1e-16 )

            # conclusion is that we waste a lot of samples at the edge of the dist.
            # DEBUG = True
            # if DEBUG:
            #    nbr_bad_samples = np.sum(~samples_to_keep)
            #    ratio_keep = nbr_bad_samples/number_of_samples
            #    Qres = np.linalg.norm(self.Q-Q_probe)
            #    print( 'Bin nbr: ', i, ' with ', ratio_keep, ' wasted samples, Qres is ',  Qres)

            if np.sum(samples_to_keep)==0:
                continue
            else:
                p_tot = self._exp( p_tot_log)#[samples_to_keep] )
                p_Q[i] = np.sum(p_tot) / number_of_samples
                std_p_Q[i] = np.std(p_tot) / np.sqrt(number_of_samples)

        norm_const = np.sum(p_Q  * dv)
        p_Q  = p_Q  / norm_const
        std_p_Q = std_p_Q / norm_const

        return p_Q, std_p_Q

    def _get_wavelength(self, nhat, d, Q):
        """Find the wavelength required for diffraction.
        """
        return -(d*d / np.pi) * nhat.T @ Q

    def _get_ghat(self, nhat, d, Q):
        """Find the scattering direction required for diffraction.
        """
        return (np.eye(3,3) - ((d*d)/(2*np.pi*np.pi))*np.outer(Q, Q)) @ nhat

    def _get_nhat(self, ghat, d, Q):
        """Find the incident ray direction required for diffraction.
        """
        return np.linalg.inv(np.eye(3,3) - ((d*d)/(2*np.pi*np.pi)) * np.outer(Q, Q)) @ ghat

    def _exp(self, a):
        """
        """
        return np.exp(a)

    def __call__(self, Q_vectors, angular_crl_shifts=None):
        """
        Calculate the likelihood of a set of Q vectors.

        Args:
            Q_vectors (:obj:`np.ndarray`): A shape (3, N) array of Q vectors.

        Returns:
            :obj:`np.ndarray`: Likelihood of the given Q vectors. shape (N, )
        """
        assert len(Q_vectors.shape)==2 and Q_vectors.shape[0]==3
        if self.Q is None:
            raise ValueError('The resolution function requires compiling before any calls can be made to the PDF.')
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


    U = np.eye(3,3)
    a = b = c = 4.0493
    unit_cell = [a, b, c, 90., 90., 90.]
    lambda_0 = 0.71
    energy_0 = laue.angstrom_to_keV(lambda_0)
    sigma_e = (1.4*1e-4)/(2*np.sqrt(2*np.log(2)))
    hkl = np.array([0, 0, 2])

    from dfxm import experiment
    goni = experiment.Goniometer(U, unit_cell, energy_0)
    goni.bring_to_bragg(hkl)
    Q = goni.U @ goni.B @ hkl
    d_0 = (2*np.pi)/np.linalg.norm(Q)
    theta_0 = np.arcsin(  lambda_0 / (2*d_0) )
    k_0 = 2 * np.pi / lambda_0

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
    gamma_C = goni.imaging_system
    desired_FWHM_C = 0.731*1e-3
    kappa_C = np.log(2)/(1-np.cos((desired_FWHM_C)/2.))
    beta_C  = 0

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
                sigma_lambda
                    )

    physical_aperture = 2 * desired_FWHM_C / (2 * np.sqrt(2 * np.log(2)))
    D = 0.477 * 1e-3
    d1 = 0.274
    physical_aperture = np.arctan( D / (2*d1) )
    print(physical_aperture, desired_FWHM_C / (2 * np.sqrt(2 * np.log(2))), desired_FWHM_N/2.)

    xray_params = {
        "std_beam_horizontal": 1e-5 / (2 * np.sqrt(2 * np.log(2))),
        "lower_bound_beam_horizontal": None,
        "upper_bound_beam_horizontal": None,
        
        "std_beam_vertical": desired_FWHM_N / (2 * np.sqrt(2 * np.log(2))),
        "lower_bound_beam_vertical": -desired_FWHM_N/2.,
        "upper_bound_beam_vertical":  desired_FWHM_N/2.,
        
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
    print(desired_FWHM_N/2.)

    res2 = TruncatedPentaGauss(
                goni.optical_axis,
                mu_lambda,
                xray_params,
                    )


    if 1:

        plt.style.use('dark_background')
        import cProfile
        import pstats
        import time
        pr = cProfile.Profile()
        pr.enable()
        t1 = time.perf_counter()

        res2.compile(Q, resolution=(5*1e-4, 5*1e-4, 5*1e-4))
        #res.compile(Q)

        t2 = time.perf_counter()
        pr.disable()
        pr.dump_stats('tmp_profile_dump')
        ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
        ps.print_stats(15)
        print('\n\nCPU time is : ', t2-t1, 's')


        def _plot(field, cmap, title):
            plt.figure()
            plt.imshow(field, cmap=cmap)
            plt.title(title)
            plt.colorbar()

        def cartesian_slices(field):
            a = field[field.shape[0]//2,:,:]
            b = field[:, field.shape[1]//2,:]
            c = field[:, :, field.shape[2]//2]
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
        Qs[0,:] = np.linspace(Q[0] - 25*1e-4, Q[0] + 25*1e-4, Qs.shape[1])
        Qs[1,:] = Q[1]
        Qs[2,:] = Q[2]

        res1.compile()
        p_Q = res1( Qs )
        plt.figure()
        plt.plot(Qs[0,:] - Q[0], p_Q, 'ro--')
        p_Q = res2( Qs )
        plt.plot(Qs[0,:] - Q[0], p_Q, 'ko--')

        Qq = lab_to_Q(Q, Q)
        qx = np.linspace(Qq[0] - 95*1e-4, Qq[0] + 95*1e-4, 64)
        qy = np.linspace(Qq[1] - 95*1e-4, Qq[1] + 95*1e-4, 64)
        qz = np.linspace(Qq[2] - 95*1e-4, Qq[2] + 95*1e-4, 64)
        Qx, Qy, Qz = np.meshgrid(qx, qy, qz, indexing='ij')
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
        for i,field in enumerate((a, b, c)):
            _plot(field, cmap='viridis', title='yz xz xy'.split(' ')[i])

        a, b, c = projection_slices(std_p_Q)
        for i,field in enumerate((a, b, c)):
            _plot(field, cmap='magma', title='yz xz xy'.split(' ')[i])

        p_Q = res1(Q_to_lab(points, Q)).reshape(Qy.shape)
        a, b, c = projection_slices(p_Q)
        for i,field in enumerate((a, b, c)):
            _plot(field, cmap='jet', title='yz xz xy'.split(' ')[i])

    samples1 = res1.sample( number_of_samples=10000)
    samples2 = res2.sample( number_of_samples=10000)

    fig = plt.figure(figsize=(8, 8))

    samples1 -= np.mean(samples1, axis=1).reshape(3,1)
    qx, qy, qz = lab_to_Q(samples1, Q)
    # qx /= np.linalg.norm(Q)
    # qy /= np.linalg.norm(Q)
    # qz /= np.linalg.norm(Q)

    alpha = 0.1

    print('Cov', (np.cov(np.array([qx, qy, qz]))*1e6).round(3) )
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(0.2 + qx*1e3, qy*1e3, qz*1e3, alpha=alpha, label='No-truncation')
    ax.scatter(0.2 + qx*1e3, qy*1e3, -15, alpha=alpha)
    ax.scatter(0.2 + qx*1e3, 15, qz*1e3, alpha=alpha)
    ax.scatter(5.5, qy*1e3, qz*1e3, alpha=alpha)

    samples2 -= np.mean(samples2, axis=1).reshape(3,1)
    qx, qy, qz = lab_to_Q(samples2, Q)
    # qx /= np.linalg.norm(Q)
    # qy /= np.linalg.norm(Q)
    # qz /= np.linalg.norm(Q)

    print('Cov', (np.cov(np.array([qx, qy, qz]))*1e6).round(3) )
    ax.scatter(-5.3 + qx*1e3, qy*1e3, qz*1e3, alpha=alpha, label='Truncation')
    ax.scatter(-5.3 + qx*1e3, qy*1e3, -15, alpha=alpha)
    ax.scatter(-5.3 + qx*1e3, 15, qz*1e3, alpha=alpha)
    ax.scatter(5.5, qy*1e3, qz*1e3, alpha=alpha)

    ax.set_xlabel('$q_{rock}$')
    ax.set_ylabel('$q_{roll}$')
    ax.set_zlabel('$q_{||}$')
    ax.set_xlim([5.5, -8])
    ax.set_ylim([ 15.3, -15.3])
    ax.set_zlim([-15.3,  15.3])
    ax.view_init(elev=20, azim=59)

    ax.legend()

    plt.show()
