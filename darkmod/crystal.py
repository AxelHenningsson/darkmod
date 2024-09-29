import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from darkmod.goniometer import Goniometer
from darkmod import laue

# TODO: implement non-zero eta diffraction alignments.



class Crystal(object):

    def __init__(self, X, Y, Z, unit_cell, orientation, defgrad):
        """A spatilally extended crystal.

        Each point (x,y,z) in the crystal is associated with a deformation gradient tensor (F).

        At instatiation the crystal internally mounts a goniometer which defines the lab
        frame. The crystal can be accessed through the `goniometer` attribute. Please see
        `darkmod.goniometer.Goniometer` for documentation on the goniometer interface.

        All input are given in sample coordinates such that

            Q_sample = `orientation` @ Q_cystal,

        and `defgrad` acts to deform sample space quanteties.

        Args:
            X, Y, Z (:obj:`numpy array`): Coordinate arrays, these are points in the crystal, shape=(m,n).
            unit_cell (:obj:`iterable`): Reference unit cell parameters (undeformed state). unit_cell=[a,b,c,alpha,beta,gamma].
            orientation (:obj:`numpy array`): Reference orientation matrix  (undeformed state), shape=(3,3).
            defgrad (:obj:`numpy array`): Per point defomration gradient tensor (F), shape=(m,n,3,3).
        """
        self.goniometer = Goniometer()

        self.U = orientation
        self.unit_cell = unit_cell
        self.B = laue.get_b_matrix(unit_cell)

        self._grid_scalar_shape = X.shape
        self._grid_vector_shape = (*X.shape, 3)
        self._grid_tensor_shape = (*X.shape, 3, 3)

        self._flat_scalar_shape = (np.prod(X.shape), )
        self._flat_vector_shape = (np.prod(X.shape), 3)
        self._flat_tensor_shape = (np.prod(X.shape), 3, 3)

        self._x = np.array([X.flatten(), Y.flatten(), Z.flatten()])
        self._F = defgrad.reshape( self._flat_tensor_shape )
        self._FiT = np.transpose(np.linalg.inv(self._F), axes=(0, 2, 1) )

    @property
    def X(self):
        return self._x[0, :].reshape( self._grid_scalar_shape )

    @X.setter
    def X(self, value):
        self._x[0, :] = value.reshape(self._flat_scalar_shape)

    @property
    def Y(self):
        return self._x[1, :].reshape(self._grid_scalar_shape)

    @Y.setter
    def Y(self, value):
        self._x[1, :] = value.reshape(self._flat_scalar_shape)

    @property
    def Z(self):
        return self._x[2, :].reshape(self._grid_scalar_shape)

    @Z.setter
    def Z(self, value):
        self._x[2, :] = value.reshape(self._flat_scalar_shape)

    @property
    def defgrad(self):
        return self._F.reshape(self._grid_tensor_shape)

    @defgrad.setter
    def defgrad(self, value):
        self._F = value.reshape(self._flat_tensor_shape)

    def get_Q_crystal(self, hkl):
        """Diffraction vectors in crystal coordinates.

        This method will return a Q-vector per spatial point in the crystal. i.e
        one Q vector per point in X,Y,Z. This means that the array:

            Q = get_Q_crystal(hkl)

        holds the Q-components, Qx = Q[i,j,k,0], Qy = Q[i,j,k,1], Qz = Q[i,j,k,2]
        at a point x,y,z = (X[i,j,k], Y[i,j,k], Z[i,j,k]).

        NOTE: Crystal coordinates are different from Q-system. The notation is NOT
            the same as Poulsen 2017. Here we define: Q_crystal = U @ B @ hkl.

        Args:
            hkl (:obj:`numpy array`): Miller indices [h, k, l], shape=(3,).

        Returns:
            :obj:`numpy array`: A field of Q-vectors, shape=(n,m,o,3).
        """
        Q_crystal_flat = self.U.T @ self._get_Q_sample_flat(hkl).T
        return Q_crystal_flat.T.reshape(self._grid_vector_shape)

    def get_Q_sample(self, hkl):
        """Diffraction vectors in sample coordinates.

        This method will return a Q-vector per spatial point in the crystal. i.e
        one Q vector per point in X,Y,Z. This means that the array:

            Q = get_Q_sample(hkl)

        holds the Q-components, Qx = Q[i,j,k,0], Qy = Q[i,j,k,1], Qz = Q[i,j,k,2]
        at a point x,y,z = (X[i,j,k], Y[i,j,k], Z[i,j,k]).

        Args:
            hkl (:obj:`numpy array`): Miller indices [h, k, l], shape=(3,).

        Returns:
            :obj:`numpy array`: A field of Q-vectors, shape=(n,m,o,3).
        """
        Q_sample_flat = self._get_Q_sample_flat(hkl)
        return Q_sample_flat.reshape(self._grid_vector_shape)

    def get_Q_lab(self, hkl):
        """Diffraction vectors in lab coordinates.

        This method will return a Q-vector per spatial point in the crystal. i.e
        one Q vector per point in X,Y,Z. This means that the array:

            Q = get_Q_lab(hkl)

        holds the Q-components, Qx = Q[i,j,k,0], Qy = Q[i,j,k,1], Qz = Q[i,j,k,2]
        at a point x,y,z = (X[i,j,k], Y[i,j,k], Z[i,j,k]).

        Args:
            hkl (:obj:`numpy array`): Miller indices [h, k, l], shape=(3,).

        Returns:
            :obj:`numpy array`: A field of Q-vectors, shape=(n,m,o,3).
        """
        Q_lab_flat = self.goniometer.R @ self._get_Q_sample_flat(hkl).T
        return Q_lab_flat.T.reshape(self._grid_vector_shape)

    def _get_Q_sample_flat(self, hkl):
        return self._FiT @ self._get_Q_0_sample_flat(hkl)

    def _get_Q_0_sample_flat(self, hkl):
        return self.U @ self.B @ hkl

    def align(self, hkl, axis):
        """
        Align the crystal orientation matrix (U) such that the G-vector of the provided Miller indices (hkl)
        is parallel to a given real space axis.

        Args:
            hkl (:obj:`numpy array`): Miller indices to align with, ``shape=(3,)``.
            axis (:obj:`numpy array`): Vector to align with, ``shape=(3,)``.

        """
        G = laue.get_G(self.U, self.B, hkl)
        nhat = G / np.linalg.norm(G)
        axis = axis / np.linalg.norm(axis)
        primary_rotation_vector = np.cross(nhat, axis)
        primary_rotation_vector /= np.linalg.norm(primary_rotation_vector)
        angle = np.arccos(nhat@axis)

        rotation = Rotation.from_rotvec(primary_rotation_vector*angle)
        self.goniometer.goto(rotation=rotation)

    def bring_to_bragg(self, hkl, energy):
        """
        Align the crystal orientation matrix (U) such that the provided Miller indices (hkl) is in
        diffraction conditions forming a Bragg angle to the incident beam (assumed to propagate along x-lab).

        The alignment will follow the protocol:
            (1) Align the hkl with the z-axis.
            (2) Rotate the base cradle (mu) with Bragg angle (theta).

        Args:
            hkl (:obj:`numpy array`): Miller indices to align with, ``shape=(3,)``.
            energy (:obj:`float`): X-ray energy in keV.

        """
        self.align(hkl, axis=np.array([0, 0, 1])) # align with z-axis first
        G = laue.get_G(self.U, self.B, hkl)
        theta = laue.get_bragg_angle(G, energy)
        self.goniometer.relative_move(dmu=-theta) # align with the bragg condition
        eta = 0
        return theta, eta

    def inspect(self, hkl, energy, rotation_axis=np.array([0, 0, 1])):
        """
        Inspect the angular settings required at diffraction for the Miller indices (hkl).

        Args:
            hkl (:obj:`numpy array`): Array of Miller indices with shape `(3, n)`.
            energy (:obj:`float`): X-ray energy in keV.
            rotation_axis (:obj:`numpy array`): Axis of rotation ``shape=(3,)``. Defaults to
                zhat=[0,0,1].

        Returns:
            df (:obj:`pandas.DataFrame`): DataFrame with Miller with columns: 'h', 'k', 'l' , 'omega', 'theta', 'eta'.
        """
        refl_labels = [f"reflection {i}" for i in range(hkl.shape[1])]
        df = pd.DataFrame(index=refl_labels, columns=['h', 'k', 'l', 'omega_1', 'omega_2', 'eta_1', 'eta_2', 'theta', '2 theta'])

        G = laue.get_G(self.U, self.B, hkl)
        df.h, df.k, df.l = hkl
        omega = laue.get_omega(self.U, self.unit_cell, hkl, energy, rotation_axis)
        df.omega_1, df.omega_2 = np.degrees(omega)
        df.theta =  np.degrees( laue.get_bragg_angle(G, energy) )
        df['2 theta'] = 2*df.theta
        df.eta_1, df.eta_2 = np.degrees( laue.get_eta_angle(G, omega, energy, rotation_axis) )

        return df

    def diffract(self, resolution_function, crl, detector):
        # 1. compute the Q_lab - crystal + goni
        # 2. compute the p_Q(Q_lab) - resolution_function
        # 3. compute detector cooridnates - crl + detector
        # 4. render image - detector
        pass
