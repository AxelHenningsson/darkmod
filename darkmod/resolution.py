import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import RegularGridInterpolator

from darkmod import laue
from darkmod.distribution import Kent, Normal
from darkmod.transforms import lab_to_Q, Q_to_lab

# TODO: implement the Poulsen methods in this module as well.

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

    def __call__(self, Q_vectors):
        """
        Calculate the likelihood of a set of Q vectors.

        Args:
            Q_vectors (:obj:`np.ndarray`): A shape (3, N) array of Q vectors.

        Returns:
            :obj:`np.ndarray`: Likelihood of the given Q vectors. shape (N, )
        """
        assert len(Q_vectors.shape)==2 and Q_vectors.shape[0]==3
        if self.Q is None:
            raise ValueError('The reoslution function requires compiling before any calls can be made to the PDF.')
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
    sigma_e = 1.4*1e-4
    hkl = np.array([0, 0, 2])

    from dfxm import experiment
    goni = experiment.Goniometer(U, unit_cell, energy=energy_0)
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


    res = DualKentGauss(
                    gamma_C,
                    kappa_C,
                    beta_C,
                    gamma_N,
                    kappa_N,
                    beta_N,
                    mu_lambda,
                    sigma_lambda,
                    )


    import cProfile
    import pstats
    import time
    pr = cProfile.Profile()
    pr.enable()
    t1 = time.perf_counter()

    res.compile(Q)

    t2 = time.perf_counter()
    pr.disable()
    pr.dump_stats('tmp_profile_dump')
    ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
    ps.print_stats(15)
    print('\n\nCPU time is : ', t2-t1, 's')

    Qs = np.zeros((3, 256))
    Qs[0,:] = np.linspace(Q[0] - 25*1e-4, Q[0] + 25*1e-4, Qs.shape[1])
    Qs[1,:] = Q[1]
    Qs[2,:] = Q[2]
    p_Q = res( Qs )
    plt.figure()
    plt.plot(Qs[0,:] - Q[0], p_Q, 'ko--')

    qy = np.linspace(Q[1] - 25*1e-4, Q[1] + 25*1e-4, 64)
    qz = np.linspace(Q[2] - 25*1e-4, Q[2] + 25*1e-4, 64)
    Qx, Qy = np.meshgrid(qy,qz,indexing='ij')

    m,n,o = res.p_Q.shape
    plt.figure()
    plt.imshow(res.p_Q[m//2, : , :])

    samples = res.sample( number_of_samples=10000)
    samples -= np.mean(samples, axis=1).reshape(3,1)
    qx, qy, qz = lab_to_Q(samples, Q)

    print('Cov', (np.cov(np.array([qx, qy, qz]))*1e6).round(3) )

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(qx*1e3, qy*1e3, qz*1e3, alpha=0.1)
    ax.scatter(qx*1e3, qy*1e3, -15, alpha=0.1)
    ax.scatter(qx*1e3, 15, qz*1e3, alpha=0.1)
    ax.scatter(15, qy*1e3, qz*1e3, alpha=0.1)

    ax.set_xlabel('$q_{rock}$')
    ax.set_ylabel('$q_{roll}$')
    ax.set_zlabel('$q_{||}$')
    ax.set_xlim([15, -15])
    ax.set_ylim([ 15.3, -15.3])
    ax.set_zlim([-15.3,  15.3])
    ax.view_init(elev=20, azim=59)

    plt.show()
