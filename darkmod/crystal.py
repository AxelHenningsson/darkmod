import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from darkmod.goniometer import Goniometer
from darkmod import laue

# TODO: implement non-zero eta diffraction alignments.



class Crystal(object):

    def __init__(self, unit_cell, orientation):
        """A spatilally extended crystal.

        To associate a point (x,y,z) in the crystal with a deformation gradient tensor (F)
        `discretize()` is to be called after instantiation.

        At instatiation the crystal internally mounts a goniometer which defines the lab
        frame. The crystal can be accessed through the `goniometer` attribute. Please see
        `darkmod.goniometer.Goniometer` for documentation on the goniometer interface.

        All input are given in sample coordinates such that

            Q_sample = `orientation` @ Q_cystal,

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

        self._x = None
        self._F = None

    def discretize(self, X, Y, Z, defgrad):
        """Discretize the crystal over a unifrom grid.

        Associates points (x,y,z) in the crystal with deformation gradient tensors (F).
        All inputs are given in sample coordinates and `defgrad` acts to deform sample
        space quanteties, such that the laboratory diffraction vector can be found as

            Q_lab = R @ inv(F.T) @ U @ B @ hkl

        where hkl are MIller indices and R the goniometer rotation matrix.

        Args:
            X, Y, Z (:obj:`numpy array`): Coordinate arrays, these are points in the crystal, shape=(m,n).
                the voxel dimension must be the same in all dimensions.
            defgrad (:obj:`numpy array`): Per point defomration gradient tensor (F), shape=(m,n,3,3).
        """
        self._grid_scalar_shape = X.shape
        self._grid_vector_shape = (*X.shape, 3)
        self._grid_tensor_shape = (*X.shape, 3, 3)

        self._flat_scalar_shape = (np.prod(X.shape), )
        self._flat_vector_shape = (np.prod(X.shape), 3)
        self._flat_tensor_shape = (np.prod(X.shape), 3, 3)

        self._x = np.array([X.flatten(), Y.flatten(), Z.flatten()])
        self._F = defgrad.reshape(self._flat_tensor_shape)
        self._FiT = np.transpose(np.linalg.inv(self._F), axes=(0, 2, 1))

        dx = X[1,0,0] - X[0,0,0]
        dy = Y[0,1,0] - Y[0,1,0]
        dz = Z[0,0,1] - Z[0,0,1]
        assert dx==dy and dx==dz, "voxels must be cubic"
        self.voxel_size = dx

    @property
    def X(self):
        if self._x is None:
            raise ValueError('Please use discretize() to instantiate the field.')
        else:
            return self._x[0, :].reshape(self._grid_scalar_shape)

    @X.setter
    def X(self, value):
        if self._x is None:
            raise ValueError('Please use discretize() to instantiate the field.')
        else:
            self._x[0, :] = value.reshape(self._flat_scalar_shape)

    @property
    def Y(self):
        if self._x is None:
            raise ValueError('Please use discretize() to instantiate the field.')
        else:
            return self._x[1, :].reshape(self._grid_scalar_shape)

    @Y.setter
    def Y(self, value):
        if self._x is None:
            raise ValueError('Please use discretize() to instantiate the field.')
        else:
            self._x[1, :] = value.reshape(self._flat_scalar_shape)

    @property
    def Z(self):
        if self._x is None:
            raise ValueError('Please use discretize() to instantiate the field.')
        else:
            return self._x[2, :].reshape(self._grid_scalar_shape)

    @Z.setter
    def Z(self, value):
        if self._x is None:
            raise ValueError('Please use discretize() to instantiate the field.')
        else:
            self._x[2, :] = value.reshape(self._flat_scalar_shape)

    @property
    def defgrad(self):
        if self._F is None:
            raise ValueError('Please use discretize() to instantiate the field.')
        else:
            return self._F.reshape(self._grid_tensor_shape)

    @defgrad.setter
    def defgrad(self, value):
        if self._F is None:
            raise ValueError('Please use discretize() to instantiate the field.')
        else:
            self._F = value.reshape(self._flat_tensor_shape)


    @property
    def UB_0(self):
        """Crystal UB matrix in (undeformed) reference state.

        Given in sample coordinates.

        Returns:
            :obj:`numpy array`: UB matrix, shape=(3,3).
        """
        return self.U @ self.B

    @property
    def UBi_0(self):
        """Crystal UB inverse matrix in (undeformed) reference state

        Given in sample coordinates.

        Returns:
            :obj:`numpy array`: UB inverse matrix, shape=(3,3).
        """
        return np.linalg.inv(self.U @ self.B)

    @property
    def C_0(self):
        """Crystal unit cell vectors in (undeformed) reference state

        Given in sample coordinates.

        Each column is a vector of the real space unit cell
        parallelpiped as: [a, b, c].

        Returns:
            :obj:`numpy array`: Cell matrix, shape=(3,3).
        """
        return np.linalg.inv(self.U @ self.B).T

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
        if self._F is None:
            raise ValueError('Please use discretize() to instantiate the field.')
        else:
            return self._FiT @ self._get_Q_0_sample_flat(hkl)

    def _get_Q_0_sample_flat(self, hkl):
        return self.U @ (self.B @ hkl)

    def align(self, hkl, axis):
        """
        Align the crystal orientation matrix (U) such that the G-vector of the provided Miller indices (hkl)
        is parallel to a given real space axis.

        Args:
            hkl (:obj:`numpy array`): Miller indices to align with, ``shape=(3,)``.
            axis (:obj:`numpy array`): Vector to align with, ``shape=(3,)``.

        """
        Q_0 = self._get_Q_0_sample_flat(hkl)
        nhat = Q_0 / np.linalg.norm(Q_0)
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
        Q_0 = self._get_Q_0_sample_flat(hkl)
        theta = laue.get_bragg_angle(Q_0, energy)
        self.goniometer.relative_move(dmu=-theta) # align with the bragg condition
        eta = 0
        return theta, eta

    def remount(self):
        """Remount the crystal on the goniometer and zero all motors.

        This amounts to changing the crystal orientation such that the current lab-frame
        crystal orientation becomes aligned with the crystal frame orientation, i.e
            U <-- R @ U
        where R is the goniometer rotation. All goniomter motors are then put to zero (R=I).
        """
        self.U = self.goniometer.R @ self.U
        self.goniometer.goto(0, 0, 0, 0)

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
        df.theta = np.degrees( laue.get_bragg_angle(G, energy) )
        df['2 theta'] = 2*df.theta
        df.eta_1, df.eta_2 = np.degrees( laue.get_eta_angle(G, omega, energy, rotation_axis) )

        return df

    def diffract(self,
                 hkl,
                 resolution_function,
                 crl,
                 detector,
                 beam):

        Q_lab = self.get_Q_lab(hkl)
        Q_lab_flat = Q_lab.reshape(self._flat_vector_shape)


        p_Q = resolution_function(Q_lab_flat.T)

        qnorm = np.linalg.norm(Q_lab, axis=-1)[:, :, 0]
        d = (2*np.pi)/qnorm
        d0 = (2*np.pi)/np.linalg.norm(resolution_function.Q)
        th0 = np.degrees( np.arcsin( 0.71 / (2*d0)) )
        th = np.degrees( np.arcsin( 0.71 / (2*d)) )

        th_current = np.degrees(np.arccos( Q_lab[:, :, 0, 2] / qnorm )[0,0])

        # print(np.max(p_Q))
        # plt.figure(figsize=(10,10))                                                 
        # plt.title('Bragg condition in sample')
        # plt.imshow( th - th_current )

        # plt.figure(figsize=(10,10))
        # plt.title('p_Q in sample')
        # plt.imshow( p_Q.reshape(self._grid_scalar_shape) )
        # plt.show()

        x_lab_at_goni = self.goniometer.R @ self._x

        plt.figure(figsize=(8,6))
        plt.scatter(x_lab_at_goni[0], x_lab_at_goni[2])
        plt.axis('equal')
        plt.grid(True)
        plt.show()

        print(np.max(x_lab_at_goni,axis=1))
        print(np.min(x_lab_at_goni,axis=1))

        w = beam(x_lab_at_goni)

        #x_lab_at_detector = crl.refract(x_lab_at_goni)
        #x_im_at_detector = crl.imaging_system.T @ x_lab_at_detector

        voxel_volume = p_Q*w
        image = detector.render(voxel_volume, self.voxel_size, crl)

        # TODO: We need to add the shifts to the resolution function
        # due to points not being at the imaging origin.
        # dq = D @ x_lab_at_goni
        # p_Q = resolution_function(Q_lab_flat.T + dq)
        # the D matrix will here be a 3x3 transformation dependent on theta
        # and gamma, it should likely be implemented in the crl at least
        # the gamma. Then perhaps the shift should be here.


        # plt.figure(figsize=(8,6))
        # plt.scatter(x_im_at_detector[1], x_im_at_detector[2],c = p_Q)
        # plt.grid(True)
        # plt.show()


        return image
    

if __name__=='__main__':


    from darkmod.beam import GaussianBeam
    from darkmod.detector import Detector
    from darkmod.resolution import DualKentGauss
    from darkmod.crystal import Crystal
    from darkmod.crl import CompundRefractiveLens
    from darkmod.laue import keV_to_angstrom
    from darkmod import laue

    def linear_y_gradient_field(shape):
        # Linear strain gradient in zz-component moving across y.
        F = unity_field(shape)
        deformation_range = np.linspace(-0.003, 0.003, shape[1])
        for i in range(len(deformation_range)):
            F[:, i, :, 2, 2] += deformation_range[i]
        return F

    def unity_field(shape):
        field = np.zeros((*shape, 3, 3))
        for i in range(3): field[:, :, :, i, i] = 1
        return field

    def simple_shear(shape, magnitude=0.02):
        F = unity_field(shape)
        F[:, :, :, 0, 1] = magnitude
        return F
    



    number_of_lenses = 50
    lens_space = 2 * 1e-3
    lens_radius = 50 * 1e-6
    refractive_decrement = 1.65 * 1e-6
    magnification = 10
    crl = CompundRefractiveLens(number_of_lenses,
                                lens_space,
                                lens_radius,
                                refractive_decrement,
                                magnification)
    hkl = np.array([0, 0, 2])
    lambda_0 = 0.71
    energy = laue.angstrom_to_keV(lambda_0)

    # Instantiate an AL crystal
    unit_cell = [4.0493, 4.0493, 4.0493, 90., 90., 90.]
    orientation = np.eye(3, 3)
    crystal = Crystal(unit_cell, orientation)

    # remount the crystal to align Q with z-axis
    crystal.align(hkl, axis=np.array([0, 0, 1]))
    crystal.remount() # this updates U.

    # Find the reflection with goniometer motors.
    theta, eta = crystal.bring_to_bragg(hkl, energy)

    # Bring the CRL to diffracted beam.
    crl.goto(theta, eta)

    # Discretize the crystal
    xg = np.linspace(-150*0.02, 150*0.02, 128)
    yg = np.linspace(-150*0.02, 150*0.02, 128)
    zg = np.linspace(-150*0.01, 150*0.01, 64)
    X, Y, Z = np.meshgrid(xg, yg, zg, indexing='ij')
    defgrad = linear_y_gradient_field(X.shape)
    crystal.discretize(X, Y, Z, defgrad)

    # import vtk
    # import meshio

    # def save_as_vtk_particles(file, coordinates, vector):
    #     """Save numpy arrays with particle information to paraview readable format.

    #     Args:
    #         file (:obj:`string`): Absolute path ending with desired filename. 
    #         coordinates (:obj:`numpy array`): Coordinates of particle ensemble, shape=(N,3).
    #         vector (:obj:`numpy array`): Velocities of particle ensemble, shape=(N,m).

    #     """
    #     cells = [("vertex", np.array([[i] for i in range(coordinates.shape[0])]) )]
    #     if len(file.split("."))==1: 
    #         filename = file + ".vtk"
    #     else:
    #         filename = file
    #     meshio.Mesh(
    #         coordinates,
    #         cells,
    #         point_data={"vector": vector},
    #         ).write(filename)
    # m,n,o = Z.shape
    # vecs = defgrad.reshape(m*n*o,9)
    # save_as_vtk_particles('field', np.array([X.flatten(),Y.flatten(),Z.flatten()]).T, vecs)

    # Q_lab and gamma_C corresponds to:
    # hkl = 002 and cubic AL, a = 4.0493 with U=I
    # After bringing these to Bragg conditions..

    # -5.44137060e-01 -1.90025011e-16  3.05526733e+00]
    #Q_lab = np.array([-5.44137060e-01 ,-1.90025011e-16 , 3.05526733e+00])
    Q_lab = crystal.goniometer.R @ crystal.UB_0 @ hkl

    # Beam divergence params
    gamma_N = np.eye(3, 3)
    desired_FWHM_N = 0.53*1e-3
    kappa_N = np.log(2)/(1-np.cos((desired_FWHM_N)/2.))
    beta_N  = 0

    # Beam wavelength params
    sigma_e = 1.4*1e-4
    epsilon = np.random.normal(0, sigma_e, size=(20000,))
    random_energy = energy + epsilon*energy
    sigma_lambda = laue.keV_to_angstrom(random_energy).std()
    mu_lambda = lambda_0

    # CRL acceptance params
    gamma_C = crl.imaging_system
    desired_FWHM_C = 0.731*1e-3
    kappa_C = np.log(2)/(1-np.cos((desired_FWHM_C)/2.))
    beta_C  = 0

    resolution_function = DualKentGauss(
                        gamma_C,
                        kappa_C,
                        beta_C,
                        gamma_N,
                        kappa_N,
                        beta_N,
                        mu_lambda,
                        sigma_lambda,
                        )

    resolution_function.compile(Q_lab,
                                resolution=(8*1e-4, 8*1e-4, 8*1e-4),
                                ranges=(3.5, 3.5, 3.5),
                                number_of_samples=5000)

    print(resolution_function.p_Q.max())

    pixel_y_size = pixel_z_size = 1
    npix_y = npix_z = 88
    detector = Detector(pixel_y_size, pixel_z_size, npix_y, npix_z)

    #crystal.goniometer.relative_move(dchi = np.radians(0.01))
    #crystal.goniometer.relative_move(dphi = np.radians(0.07))

    #crystal.goniometer.relative_move(dphi = -np.radians(0.029))

    beam = GaussianBeam(y_std=1e8, z_std=0.125, energy=energy)

    import cProfile
    import pstats
    import time
    pr = cProfile.Profile()
    pr.enable()
    t1 = time.perf_counter()
    im = crystal.diffract(hkl,
                        resolution_function,
                        crl,
                        detector,
                        beam
                    )
    t2 = time.perf_counter()
    pr.disable()
    pr.dump_stats('tmp_profile_dump')
    ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
    ps.print_stats(15)
    print('\n\nCPU time is : ', t2-t1, 's')

    plt.figure()
    plt.imshow(im, cmap='gray')
    plt.show()
