import unittest
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
import numpy as np

from darkmod.crystal import Crystal
from darkmod.laue import keV_to_angstrom


class TestCrystal(unittest.TestCase):

    def setUp(self):
        xg = np.linspace(-1, 1, 5)
        self.X, self.Y, self.Z = np.meshgrid(xg, xg, xg, indexing='ij')
        self.unit_cell = [4.0493, 4.0493, 4.0493, 90, 90, 90]
        self.orientation = Rotation.from_rotvec((np.pi/4)*np.array([1, 1, 1])/np.sqrt(3.)).as_matrix()
        self.defgrad = self.simple_shear(self.X.shape)
        self.crystal = Crystal(self.X,
                            self.Y,
                            self.Z,
                            self.unit_cell,
                            self.orientation,
                            self.defgrad)
        phi, chi, omega, mu = (np.random.rand(4,)-0.5)*np.pi/18.
        self.crystal.goniometer.goto(phi, chi, omega, mu)

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
        pandas_table = self.crystal.inspect(hkl, energy, rotation_axis)

    def test_diffract(self):
        pass

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