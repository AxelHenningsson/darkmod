import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from darkmod import laue

# TODO: implement non-zero etas

class Goniometer:
    """
    A class to represent a Dark-Field X-ray Microscopy (DFXM) goniometer setup.

    Attributes:
        phi, chi, mu, omega (:obj:`float`): Goniometer motor settings in radians. Starts at zero.

    """

    def __init__(self):
        """
        Initialize DFXM goniometer.
        """
        self.phi = self.chi = self.mu = self.omega = 0

        self._xhat_lab = np.array([1., 0., 0.])
        self._yhat_lab = np.array([0., 1., 0.])
        self._zhat_lab = np.array([0., 0., 1.])

    def relative_move(self, dphi=None, dchi=None, domega=None, dmu=None):
        if dphi is not None:
            self.phi += dphi
        if dchi is not None:
            self.chi += dchi
        if domega is not None:
            self.omega += domega
        if dmu is not None:
            self.mu += dmu

    def goto(self, phi=None, chi=None, omega=None, mu=None, rotation=None):
        """Move the goniometer to specified angles.

        Either directly give angles phi, chi, omega, mu or, alternatively,
        a scipy.spatial.transform.Rotation or rotation matrix from which
        angles (phi, chi, omega, mu) will be computed such that phi=0.

        Args:
            phi (:obj:`float`): Phi angle in radians (top rock - y rotation).
            chi (:obj:`float`): Chi angle in radians (top roll - x rotation).
            omega (:obj:`float`): Omega angle in radians (around Q-vector - z rotation).
            mu (:obj:`float`): Mu angle in radians (bottom roll - x rotation).
            rotation (:obj:`scipy.spatial.transform.Rotation` or :obj:`numpy array`): set
                the goniometer angles to correspond to an element of SO3 (a rotation).
                WIther a rotaiton object or a 3x3 rotation matrix.

        """
        if rotation is not None:
            self.phi, self.chi, self.omega, self.mu = self._rotation_to_motor_angles(rotation)
        elif None not in (phi, chi, omega, mu):
            self.phi, self.chi, self.omega, self.mu = phi, chi, omega, mu
        else:
            raise ValueError('Either pass a scipy.spatial.transform.Rotation or 4 floats: phi, chi, omega, mu.')

    def _rotation_to_motor_angles(self, rotation):
        """Convert a rotation element into goniometer motor angles.

        Args:
            rotation (:obj:`scipy.spatial.transform.Rotation` or :obj:`numpy array`): set
                the goniometer angles to correspond to an element of SO3 (a rotation).
                WIther a rotaiton object or a 3x3 rotation matrix.

        Returns:
            :obj:`tuple` of :obj:`float`: Motor setting, phi, chi, omega, mu.
        """
        if isinstance(rotation, np.ndarray):
            rotation = Rotation.from_matrix(rotation)
        axes = np.array([self._xhat_lab, self._zhat_lab, self._yhat_lab]) # Phi then Chi then Mu
        chi, omega, mu = Rotation.as_davenport(rotation, axes, order='extrinsic')
        return 0, chi, omega, mu

    @property
    def R(self):
        """Rotation matrix for the current goniometer angles.

        This matrix will take a vector in sample system and bring it
        to the lab frame, such that R @ sample = lab. I.e R is the
        Cartesian basis of the sample system as described in lab
        coordinates.

        Returns:
            (:obj:`numpy array`): Rotation matrix. shape=(3,3).
        """
        Ry_phi = Rotation.from_rotvec(self._yhat_lab * self.phi)
        Rx_chi = Rotation.from_rotvec(self._xhat_lab * self.chi)
        Rz_omega = Rotation.from_rotvec(self._zhat_lab * self.omega)
        Ry_mu = Rotation.from_rotvec(self._yhat_lab * self.mu)
        return (Ry_mu * Rz_omega * Rx_chi * Ry_phi).as_matrix()

    @property
    def info(self):
        print('\n')
        print('---------------------------------------------------------------------')
        print('Goniometer is at angles (degrees) : ')
        print('---------------------------------------------------------------------')
        for key in self.motors:
            print(key.ljust(7), str(np.round(self.motors[key], 6)))
        print('---------------------------------------------------------------------')
        print('\n')

    @property
    def motors(self):
        """Return the current goniometer motor angles in units of degrees, as a dictionary.

        Returns:
            dict: Goniometer angles with keys 'phi', 'chi', 'omega', and 'mu', (units of degrees)
        """
        return {'phi': np.degrees(self.phi),
                'chi': np.degrees(self.chi),
                'omega': np.degrees(self.omega),
                'mu': np.degrees(self.mu)}