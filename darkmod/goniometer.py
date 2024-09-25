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
        self._imaging_system_0 = np.eye(3, 3)
        self.phi = self.chi = self.mu = self.omega = 0
        self.eta = self.theta = None

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

        Either give angles phi, chi, omega, mu or a scipy.spatial.transform.Rotation
        from which angles will be computed such that omega=0.

        Args:
            phi (:obj:`float`): Phi angle in radians (top rock - y rotation).
            chi (:obj:`float`): Chi angle in radians (top roll - x rotation).
            omega (:obj:`float`): Omega angle in radians (around Q-vector - z rotation).
            mu (:obj:`float`): Mu angle in radians (bottom roll - x rotation).
            rotation (:obj:`scipy.spatial.transform.Rotation`): set the goniometer angles to
                correspond to an element of SO3 (a rotation).

        """
        if rotation is not None:
            axes = np.array([self._yhat_lab, self._xhat_lab, self._yhat_lab]) # Phi then Chi then Mu
            phi, chi, mu = Rotation.as_davenport( rotation, axes, order='extrinsic')
            self.phi, self.chi, self.omega, self.mu = phi, chi, 0, mu
        elif None not in (phi, chi, omega, mu):
            self.phi, self.chi, self.omega, self.mu = phi, chi, omega, mu
        else:
            raise ValueError('Either pass a scipy.spatial.transform.Rotation or 4 floats: phi, chi, omega, mu.')

    def from_(self, phi, chi, omega, mu):
        """Move the goniometer to specified angles.

        Args:
            phi (float): Phi angle in radians (top rock - y rotation).
            chi (float): Chi angle in radians (top roll - x rotation).
            omega (float): Omega angle in radians (around Q-vector - z rotation).
            mu (float): Mu angle in radians (bottom roll - x rotation).

        """
        self.phi, self.chi, self.omega, self.mu = phi, chi, omega, mu

    @property
    def R(self):
        """Rotation matrix for the current goniometer angles.

        Returns:
            (:obj:`numpy array`): Rotation matrix. shape=(3,3).
        """
        Ry_phi = Rotation.from_rotvec(self._yhat_lab * self.phi)
        Rx_chi = Rotation.from_rotvec(self._xhat_lab * self.chi)
        Rz_omega = Rotation.from_rotvec(self._zhat_lab * self.omega)
        Ry_mu = Rotation.from_rotvec(self._yhat_lab * self.mu)
        return (Ry_phi * Rx_chi * Rz_omega * Ry_mu).as_matrix()

    @property
    def optical_axis(self):
        """optical axis for the current goniometer angles (as given in lab coordinates).

        Returns:
            (:obj:`numpy array`): optical axis. shape=(3,).
        """
        return self.imaging_system[:, 0]

    @property
    def imaging_system(self):
        """imaging coordinate system for the current goniometer angles (as given in lab coordinates).

        Returns:
            (:obj:`numpy array`): imaging coordinate system. shape=(3,3).
        """
        rotation_th = Rotation.from_rotvec(self._yhat_lab*(-2*self.theta)).as_matrix()
        rotation_eta = Rotation.from_rotvec(self._xhat_lab*(self.eta)).as_matrix()
        self._imaging_system = rotation_eta @ rotation_th @ self._imaging_system_0
        return self._imaging_system[:, :]

    @property
    def info(self):
        print('\n')
        print('---------------------------------------------------------------------')
        print('Goniometer is at angles (degrees) : ')
        print('---------------------------------------------------------------------')
        for key in self.motors:
            print(key.ljust(7), str(np.round(self.motors[key], 6)))
        print('---------------------------------------------------------------------')
        print('Bragg conditions (degrees) : ')
        print('---------------------------------------------------------------------')
        if self.theta is not None:
            print('theta ', np.degrees(self.theta))
            print('eta ', np.degrees(self.eta))
        else:
            print('Not yet set to any Bragg conditions')
        print('---------------------------------------------------------------------')
        print('The optical axis is  : ')
        print('---------------------------------------------------------------------')
        print(self.optical_axis)

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