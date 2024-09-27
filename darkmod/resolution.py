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

    def compile(self, Q, res=5 * 1e-4):
        """Compile an approximation of p_Q - the reciprocal resolution function.

        Args:
            Q (:obj:`np.ndarray`): Nominal Q-vector.
        """
        self.Q = Q
        Q_sample = self.sample( number_of_samples=10000 ) #- self.Q.reshape(3,1)
        Q_sample_q_system = lab_to_Q(Q_sample, self.Q)



        mx, my, mz = np.mean(Q_sample_q_system, axis=1)
        stdx, stdy, stdz = np.std(Q_sample_q_system, axis=1)

        qx_range = np.arange( -2.5*stdx + mx, 2.5*stdx + mx + res, res)
        qy_range = np.arange( -2.5*stdy + my, 2.5*stdy + my + res, res)
        qz_range = np.arange( -2.5*stdz + mz, 2.5*stdz + mz + res, res)

        #print(qz_range.shape, qy_range.shape, qz_range.shape)

        Qx, Qy, Qz = np.meshgrid( qx_range, qy_range, qz_range, indexing='ij' )
        q_points = np.array([Qx.flatten(), Qy.flatten(), Qz.flatten()])
        q_points_lab = Q_to_lab(q_points, self.Q)


        p_Q, std_p_Q = self._monte_carlo_integrate(q_points_lab, dv=res**3)
        p_Q = p_Q.reshape(Qx.shape)
        self._p_Q = RegularGridInterpolator((qx_range, qy_range, qz_range), p_Q, method='nearest')


    def _monte_carlo_integrate(self, q_points_lab, dv):

        number_of_samples = 1000

        if self.secondary_ray_direction.kappa > self.primary_ray_direction.kappa:
            prior = 'CRL'
            ghat = self.secondary_ray_direction.sample(number_of_samples)
            mode = self.secondary_ray_direction.gamma[:, 0]
            log_norm_const = self.secondary_ray_direction(mode, normalise=False, log=True)
        else:
            prior = 'beam'
            nhat = self.primary_ray_direction.sample(number_of_samples)
            mode = self.primary_ray_direction.gamma[:, 0]
            log_norm_const = self.primary_ray_direction(mode, normalise=False, log=True)

        Qnorms = np.linalg.norm(q_points_lab, axis=0)
        dmap = (2*np.pi)/Qnorms


        p_Q = np.zeros((q_points_lab.shape[1], ))
        std_p_Q = np.zeros((q_points_lab.shape[1], ))


        for i, Q_probe in enumerate(q_points_lab.T):
            #print(i / float(len(q_points_lab.T)))
            Q_probe = self.Q

            d = dmap[i]

            print(d, (2*np.pi)/np.linalg.norm(Q_probe))
            raise

            if prior=='CRL':
                nhat = self._get_nhat(ghat, d, Q_probe)
                log_p_sample = self.primary_ray_direction(nhat, normalise=False, log=True)
            elif prior=='beam':
                ghat = self._get_ghat(nhat, d, Q_probe)
                log_p_sample = self.secondary_ray_direction(ghat, normalise=False, log=True)

            log_c_p = log_p_sample - log_norm_const
            lamda = self._get_wavelength(nhat, d, Q_probe)
            log_p_A = self.ray_wavelength(lamda, normalise=False, log=True)

            # some safe removals to save the costly exp call
            p_tot = log_c_p + log_p_A
            print(log_p_sample.max(), log_norm_const)
            #print(np.max(p_tot), np.log( (1/number_of_samples) * 1e-16 ))

            m = p_tot < np.log( (1/number_of_samples) * 1e-16 )
            p_tot[m] = 0
            p_tot[~m] = self._exp( p_tot[~m] )

            p_Q[i] = np.sum(p_tot) / number_of_samples
            #raise
            std_p_Q[i] = np.std(p_tot) / np.sqrt(number_of_samples)

        norm_const = np.sum(p_Q  * dv)
        p_Q  = p_Q  / norm_const
        std_p_Q = std_p_Q / norm_const

        return p_Q, std_p_Q

    def _get_wavelength(self, nhat, d, Q):
        return -(d*d / np.pi) * nhat.T @ Q

    def _get_ghat(self, nhat, d, Q):
        return (np.eye(3,3) - ((d*d)/(2*np.pi*np.pi))*np.outer(Q, Q)) @ nhat

    def _get_nhat(self, ghat, d, Q):
        return np.linalg.inv(np.eye(3,3) - ((d*d)/(2*np.pi*np.pi)) * np.outer(Q, Q)) @ ghat

    def _exp(self, a):
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
            return self._p_Q(Q_vectors_q_system.T)

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
    Qs[0,:] = np.linspace(Q[0] - 10*1e-4, Q[0] + 10*1e-4, Qs.shape[1])
    Qs[1,:] = Q[1]
    Qs[2,:] = Q[2]
    p_Q = res( Q.reshape(3,1) )
    print(p_Q)
    plt.plot(Qs[0,:] - Q[0], p_Q, 'ko--')
    plt.show()
    raise

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
