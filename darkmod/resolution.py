import numpy as np
import matplotlib.pyplot as plt
import darkmod

class ReciprocalResolutionFunction:
    """
    Parent class for reciprocal space resolution functions.

    Args:
        nominal_Q (:obj:`np.ndarray`): The nominal scattering vector (3,).
        nominal_energy (:obj:`float`): The nominal energy in keV.
    """

    def __init__(self, nominal_Q, nominal_energy):
        self.nominal_Q = nominal_Q
        self.nominal_energy = nominal_energy

    def compile(self):
        """
        Perform Monte Carlo integration to set up a histogram of the target function.
        """
        pass

    def sample(self, number_of_samples):
        """
        Return a sample of Q vectors from the distribution.

        Args:
            number_of_samples (:obj:`int`): Nnmber of Q to sample.

        Returns:
            :obj:`np.ndarray`: Sampled Q vectors. shape=(3,number_of_samples).
        """
        pass

    def __call__(self, Q_vectors):
        """
        Calculate the likelihood of a set of Q vectors.

        Args:
            Q_vectors (:obj:`np.ndarray`): A shape (3, N) array of Q vectors.

        Returns:
            :obj:`float`: Likelihood of the given Q vectors.
        """
        pass


class DualKentGauss(ReciprocalResolutionFunction):
    """
    Class to model a reciprocal resolution funciton. The underlying ray model uses a two Kent distributions,
    one for the primary and and one for the secondary ray bundle. The wavelength is modelled with a Guassian.
    The model is fully elastic such that wavelengths are preserved throughout scattering.

    The model was proposed by Henningsson 2024.

    Args:
        nominal_Q (:obj:`np.ndarray`): The nominal scattering vector (3,).
        nominal_energy (:obj:`float`): The nominal energy in keV.
        gamma_C (:obj:`np.ndarray`): Orientation vector for the scattered ray bundle (CRL).
        kappa_C (:obj:`float`): Concentration parameter for the scattered ray bundle (CRL).
        beta_C (:obj:`float`): Ellipticity parameter for the scattered ray bundle (CRL).
        gamma_N (:obj:`np.ndarray`): Orientation vector for the primary ray bundle.
        kappa_N (:obj:`float`): Concentration parameter for the primary ray bundle.
        beta_N (:obj:`float`): Ellipticity parameter for the primary ray bundle.
        mu_lambda (:obj:`float`): Mean of the wavelength distribution.
        sigma_lambda (:obj:`float`): Standard deviation of the wavelength distribution.
    """

    def __init__(self, nominal_Q, nominal_energy, gamma_C, kappa_C, beta_C, gamma_N, kappa_N, beta_N, mu_lambda, sigma_lambda):
        super().__init__(nominal_Q, nominal_energy)
        self.primary_ray_direction = darkmod.distributions.Kent(gamma_N, kappa_N, beta_N)
        self.secondary_ray_direction = darkmod.distributions.Kent(gamma_C, kappa_C, beta_C)
        self.ray_wavelength = darkmod.distributions.Normal(mu_lambda, sigma_lambda)

    def compile(self):

    def __call__(self, Q_vectors):
        """
        Calculate the likelihood of a set of Q vectors.

        Args:
            Q_vectors (:obj:`np.ndarray`): A shape (3, N) array of Q vectors.

        Returns:
            :obj:`float`: Likelihood of the given Q vectors.
        """
        pass

    def sample(self, number_of_samples):
        """
        Generate samples of Q vectors using the Henningsson method.

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
    pass