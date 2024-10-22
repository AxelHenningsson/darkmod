import unittest
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
import numpy as np

from darkmod.beam import GaussianBeam
from darkmod.detector import Detector
from darkmod.resolution import DualKentGauss
from darkmod.crystal import Crystal
from darkmod.crl import CompundRefractiveLens
from darkmod.laue import keV_to_angstrom
from darkmod import laue


class TestCrystal(unittest.TestCase):

    def setUp(self):
        self.unit_cell = [4.0493, 4.0493, 4.0493, 90, 90, 90]
        self.orientation = Rotation.from_rotvec((np.pi/4)*np.array([1, 1, 1])/np.sqrt(3.)).as_matrix()
        self.crystal = Crystal( self.unit_cell, self.orientation )
        phi, chi, omega, mu = (np.random.rand(4,)-0.5)*np.pi/18.
        self.crystal.goniometer.goto(phi, chi, omega, mu)

        xg = np.linspace(-1, 1, 5)
        self.X, self.Y, self.Z = np.meshgrid(xg, xg, xg, indexing='ij')
        self.defgrad = self.simple_shear(self.X.shape)
        self.crystal.discretize(self.X, self.Y, self.Z, self.defgrad)

    def test_field_shapes(self):
        # Test that the coordinate and tesnor arrays can be accessed in grid format.
        np.testing.assert_equal( self.X.shape, self.crystal.X.shape )
        np.testing.assert_equal( self.Y.shape, self.crystal.Y.shape )
        np.testing.assert_equal( self.Z.shape, self.crystal.Z.shape )
        np.testing.assert_equal( self.defgrad.shape, self.crystal.defgrad.shape )

    def test_field_values(self):
        # Test that the coordinate and tesnor arrays have proper values.
        np.testing.assert_allclose( self.X, self.crystal.X )
        np.testing.assert_allclose( self.Y, self.crystal.Y )
        np.testing.assert_allclose( self.Z, self.crystal.Z )
        np.testing.assert_allclose( self.defgrad, self.crystal.defgrad )

    def test_field_setters(self):
        # Test that the setters and getters will reshape arrays correctly.

        # swap values to be used in setting.
        xg = np.linspace(-1.0345438, 1.2375, self.Z.shape[0])
        X, Y, Z = np.meshgrid(xg, xg, xg, indexing='ij')
        defgrad = self.simple_shear(self.X.shape, magnitude=0.02837424)

        # set the attrubutes, interal reshaping will take place.
        self.crystal.X = X
        self.crystal.Y = Y
        self.crystal.Z = Z
        self.crystal.defgrad = defgrad

        # verify that the setting was done.
        np.testing.assert_allclose( X, self.crystal.X )
        np.testing.assert_allclose( Y, self.crystal.Y )
        np.testing.assert_allclose( Z, self.crystal.Z )
        np.testing.assert_allclose( defgrad, self.crystal.defgrad )

        # re-set the attributes.
        self.crystal.X = self.X
        self.crystal.Y = self.Y
        self.crystal.Z = self.Z
        self.crystal.defgrad = self.defgrad

        # verify that the re-setting worked as expected.
        np.testing.assert_allclose( self.X, self.crystal.X )
        np.testing.assert_allclose( self.Y, self.crystal.Y )
        np.testing.assert_allclose( self.Z, self.crystal.Z )
        np.testing.assert_allclose( self.defgrad, self.crystal.defgrad )

    def test_get_Q_crystal(self):
        # Test that the crystal Q-vector is correctly computed.

        # Test that the shape is ok
        hkl = np.array([-1, 2, 1])
        Qc = self.crystal.get_Q_crystal(hkl)
        np.testing.assert_equal(Qc.shape, (*self.X.shape, 3))

        # Assert that the value is ok
        Q0_sample = self.orientation @ self.crystal.B @ hkl
        Q_sample = np.linalg.inv(self.defgrad[0, 0, 0]).T @ Q0_sample
        expected_Q_crystal = self.orientation.T @ Q_sample
        np.testing.assert_allclose(expected_Q_crystal[0], Qc[:, :, :, 0])
        np.testing.assert_allclose(expected_Q_crystal[1], Qc[:, :, :, 1])
        np.testing.assert_allclose(expected_Q_crystal[2], Qc[:, :, :, 2])

    def test_get_Q_sample(self):
        # Test that the sample Q-vector is correctly computed.

        # Test that the shape is ok
        hkl = np.array([-1, 2, 1])
        Qs = self.crystal.get_Q_sample(hkl)
        np.testing.assert_equal(Qs.shape, (*self.X.shape, 3))

        # Assert that the value is ok
        Q0_sample = self.orientation @ self.crystal.B @ hkl
        expected_Q_sample = np.linalg.inv(self.defgrad[0, 0, 0]).T @ Q0_sample
        np.testing.assert_allclose(expected_Q_sample[0], Qs[:, :, :, 0])
        np.testing.assert_allclose(expected_Q_sample[1], Qs[:, :, :, 1])
        np.testing.assert_allclose(expected_Q_sample[2], Qs[:, :, :, 2])

    def test_get_Q_lab(self):
        # Test that the lab Q-vector is correctly computed.

        # Test that the shape is ok
        hkl = np.array([-1, 2, 1])
        Ql = self.crystal.get_Q_lab(hkl)
        np.testing.assert_equal(Ql.shape, (*self.X.shape, 3))

        # Assert that the value is ok
        Q0_sample = self.orientation @ self.crystal.B @ hkl
        expected_Q_sample = np.linalg.inv(self.defgrad[0, 0, 0]).T @ Q0_sample
        expected_Q_lab = self.crystal.goniometer.R @ expected_Q_sample
        np.testing.assert_allclose(expected_Q_lab[0], Ql[:, :, :, 0])
        np.testing.assert_allclose(expected_Q_lab[1], Ql[:, :, :, 1])
        np.testing.assert_allclose(expected_Q_lab[2], Ql[:, :, :, 2])

    def test_Q_consistency(self):
        # Test that the crystal, lab and sample Q-vectors are consistent with
        # respect to crystal orientation and goniometer settings.
        hkl = np.array([-1, 2, 4])
        Qc = self.crystal.get_Q_crystal(hkl)
        Qs = self.crystal.get_Q_sample(hkl)
        Ql = self.crystal.get_Q_lab(hkl)
        for i in range(self.X.shape[0]):
            for j in range(self.Y.shape[1]):
                for k in range(self.Z.shape[2]):
                    expected_Qs = self.orientation @ Qc[i, j, k]
                    expected_Ql = self.crystal.goniometer.R @ expected_Qs
                    np.testing.assert_allclose(Qs[i, j, k], expected_Qs)
                    np.testing.assert_allclose(Ql[i, j, k], expected_Ql)

    def test_align(self):
        # Test aligment of lattice plane normal with lab axis.
        hkl = np.array([-1, 1, -3])
        axis_to_align_with = np.array([1., 2., 4.]) # in lab frame.
        axis_to_align_with = axis_to_align_with / np.linalg.norm(axis_to_align_with)
        self.crystal.align(hkl, axis=axis_to_align_with)
        Q_lab = self.crystal.goniometer.R @ self.crystal.U @ self.crystal.B @ hkl
        Q_norm_lab = Q_lab / np.linalg.norm(Q_lab)
        np.testing.assert_allclose(Q_norm_lab, axis_to_align_with, atol=1e-14)

        # The deformed field of Q vectors should be almost aligned with the axis now
        Ql = self.crystal.get_Q_lab(hkl)
        Ql /= np.linalg.norm(Ql, axis=-1)[:, :, :, np.newaxis]
        np.testing.assert_almost_equal(Ql[:, :, :, 0], axis_to_align_with[0], decimal=2)
        np.testing.assert_almost_equal(Ql[:, :, :, 1], axis_to_align_with[1], decimal=2)
        np.testing.assert_almost_equal(Ql[:, :, :, 2], axis_to_align_with[2], decimal=2)

    def test_bring_to_bragg(self):
        # Test aligment of lattice plane normal Laue conditions.
        hkl = np.array([1, -1, 1])
        energy = 17.1
        theta, eta =self.crystal.bring_to_bragg(hkl, energy)
        Q_lab = self.crystal.goniometer.R @ self.crystal.U @ self.crystal.B @ hkl # sample and lab is now aligned.
        Q_lab_norm = Q_lab / np.linalg.norm(Q_lab)
        angle = np.arccos( Q_lab_norm @ np.array([1, 0, 0]) ) - (np.pi/2.)

        self.assertAlmostEqual(angle, theta)
        self.assertAlmostEqual(Q_lab_norm[1], 0)
        self.assertLess(Q_lab_norm[0], 0)
        self.assertGreater(Q_lab_norm[2], 0)

        # Verify that the computed Bragg angle complies with the Bragg condition.
        d = self.unit_cell[0] / np.sqrt(3)
        self.assertAlmostEqual( 2*d*np.sin(theta), keV_to_angstrom(energy) )

        # The deformed field of Q vectors should be almost in Laue conditions.
        # They should all be at least within a degree of the Bragg condition.
        Ql = self.crystal.get_Q_lab(hkl)
        Ql /= np.linalg.norm(Ql, axis=-1)[:, :, :, np.newaxis]
        for i in range(self.X.shape[0]):
            for j in range(self.Y.shape[1]):
                for k in range(self.Z.shape[2]):
                    angle = np.arccos( Ql[i, j, k] @ np.array([1, 0, 0]) ) - (np.pi/2.)
                    self.assertLess(np.abs(angle-theta), np.radians(1))

    def test_inspect(self):
        # Test the inspect method.
        # TODO: need additional testing. Especially laue.get_omega().
        hkl = np.array([[1, -1, 1],[0, -1, 1],[2, 1, 1]]).T
        energy = 17.1
        rotation_axis = np.array([0, 0, 1])
        _ = self.crystal.inspect(hkl, energy, rotation_axis)

    def test_diffract(self):
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
        xg = np.linspace(-150*0.02, 150*0.02, 601)
        yg = np.linspace(-150*0.02, 150*0.02, 601)
        zg = np.linspace(-0.05, 0.05, 1)
        zg = zg*0
        X, Y, Z = np.meshgrid(xg, yg, zg, indexing='ij')
        defgrad = self.linear_y_gradient_field(X.shape)
        crystal.discretize(X, Y, Z, defgrad)

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

        crystal.goniometer.relative_move(dphi = -np.radians(0.029))

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

    def linear_y_gradient_field(self, shape):
        # Linear strain gradient in zz-component moving across y.
        F = self.unity_field(shape)
        deformation_range = np.linspace(-0.003, 0.003, shape[1])
        for i in range(len(deformation_range)):
            F[:, i, :, 2, 2] += deformation_range[i]
        return F

    def unity_field(self, shape):
        field = np.zeros((*shape, 3, 3))
        for i in range(3): field[:, :, :, i, i] = 1
        return field

    def simple_shear(self, shape, magnitude=0.02):
        F = self.unity_field(shape)
        F[:, :, :, 0, 1] = magnitude
        return F

if __name__ == '__main__':
    unittest.main()