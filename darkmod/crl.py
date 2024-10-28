import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation


"""
Based on Simons 2016:
Simulating and optimizing compound refractive lens-based X-ray microscopes
J. Synchrotron Rad. (2017). 24, 392–401
https://doi.org/10.1107/S160057751602049X J.
"""


class CompundRefractiveLens(object):

    def __init__(
        self,
        number_of_lenses,
        lens_space,
        lens_radius,
        refractive_decrement,
        magnification,
    ):
        self.number_of_lenses = number_of_lenses
        self.lens_space = lens_space
        self.lens_radius = lens_radius
        self.refractive_decrement = refractive_decrement
        self.magnification = magnification

        self._xhat_lab = np.array([1.0, 0.0, 0.0])
        self._yhat_lab = np.array([0.0, 1.0, 0.0])
        self._zhat_lab = np.array([0.0, 0.0, 1.0])

        self._imaging_system_0 = np.eye(3, 3)

        self.theta = None
        self.eta = None

    def goto(self, theta, eta):
        self.theta = theta
        self.eta = eta

    def refract(self, x):
        """Map lab coordinates (upstream of crl) to the image plane through refraction.

        It is assumed that all coordinates `x` lie at distance d1 from the crl. I.e
        assumed to lie in the object plane. When this is not true, there exist an error
        in the mapped coordinates that scales with d1-dtrue.

        Args:
            x (:obj:`numpy array`): 3D lab coordinates downstreams of crl lens. shape=(3,n).

        Returns:
            (:obj:`numpy array`): Mapped coordinates. shape=(3,n).
        """
        # TODO: add spatial origin considerations.
        oa = self.optical_axis
        projection_matrix = np.eye(3, 3) - np.outer(oa, oa)  # maps to object plane.
        x_mapped = self.imaging_system.T @ (projection_matrix @ x)
        x_mapped = -self.magnification * x_mapped
        x_mapped[0, :] = self.source_to_detector_distance
        return self.imaging_system @ x_mapped

    @property
    def optical_axis(self):
        """optical axis for the current angles (as given in lab coordinates).

        Returns:
            (:obj:`numpy array`): optical axis. shape=(3,).
        """
        return self.imaging_system[:, 0]

    @property
    def imaging_system(self):
        """imaging coordinate system for the current angles (as given in lab coordinates).

        Returns:
            (:obj:`numpy array`): imaging coordinate system. shape=(3,3).
        """
        rotation_th = Rotation.from_rotvec(self._yhat_lab * (-2 * self.theta))
        rotation_eta = Rotation.from_rotvec(self._xhat_lab * (self.eta))
        rot = (rotation_eta * rotation_th).as_matrix()
        return rot @ self._imaging_system_0

    @property
    def T(self):
        return self.lens_space

    @property
    def R(self):
        return self.lens_radius

    @property
    def N(self):
        return self.number_of_lenses

    @property
    def delta(self):
        return self.refractive_decrement

    @property
    def f(self):
        return self.R / (2 * self.delta)

    @property
    def f_N(self):
        return self.f * self.phi * (1.0 / np.tan(self.N * self.phi))

    @property
    def phi(self):
        t2 = 1 - (self.T / (2 * self.f))
        t1 = np.sqrt(1 - (t2) ** 2)
        return np.arctan(t1 / t2)

    @property
    def M_N(self):
        Nc = np.cos(self.N * self.phi)
        Ns = np.sin(self.N * self.phi)
        s = np.sin(self.phi)
        return np.array([[Nc, self.f * Ns * s], [-Ns / (s * self.f), Nc]])

    @property
    def K(self):
        M = self.M_N
        M11, M12 = M[0]
        M21, M22 = M[1]
        d1, d2 = self.d1, self.d2

        K11 = M11 + d2 * M21
        K12 = M12 + d1 * (M11 + d2 * M21) + d2 * M22
        K21 = M21
        K22 = d1 * M21 + M22

        return np.array([[K11, K12], [K21, K22]])

    @property
    def lens_focal_length(self):
        return self.f

    @property
    def crl_focal_length(self):
        return self.f_N

    @property
    def d2(self):
        M = self.M_N
        return -(self.magnification + M[0, 0]) / M[1, 0]

    @property
    def d1(self):
        M = self.M_N
        d2 = self.d2
        return -(d2 * M[1, 1] + M[0, 1]) / (M[0, 0] + d2 * M[1, 0])

    @property
    def source_to_detector_distance(self):
        return self.L

    @property
    def L(self):
        return self.d1 + self.d2 + self.N * self.T

    @property
    def info(self):
        print("------------------------------------------------------------")
        print("CRL information in units of [m]")
        print("------------------------------------------------------------")
        print("Sample to crl distance        : ", self.d1)
        print("CRL to detector distance      : ", self.d2)
        print("CRL focal length              : ", self.crl_focal_length)
        print("Source to detector distance   : ", self.source_to_detector_distance)
        print("Lens spacing                  : ", self.T)
        print("Number of lenses              : ", self.N)
        print("Lens radius                   : ", self.R)
        print("Refractive Decrement          : ", self.refractive_decrement)
        print("------------------------------------------------------------")
