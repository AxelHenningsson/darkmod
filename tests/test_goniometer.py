import unittest
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
import numpy as np

from darkmod.goniometer import Goniometer


class TestGoniometer(unittest.TestCase):

    def setUp(self):
        self.goni = Goniometer()

    def test_relative_move(self):
        # test that relative moves increment motors
        dphi, dchi, domega, dmu = np.random.rand(4,)-0.5
        self.goni.goto(0.1, 0.2, 0.3, 0.4)
        self.goni.relative_move(dphi, dchi, domega, dmu)
        self.assertAlmostEqual(dphi+0.1, self.goni.phi)
        self.assertAlmostEqual(dchi+0.2, self.goni.chi)
        self.assertAlmostEqual(domega+0.3, self.goni.omega)
        self.assertAlmostEqual(dmu+0.4, self.goni.mu)

    def test_goto(self):
        # Test that goto moves the motors.
        dphi, dchi, domega, dmu = np.random.rand(4,)-0.5
        self.goni.relative_move(dphi, dchi, domega, dmu)

        phi, chi, omega, mu = np.random.rand(4,)-0.5
        self.goni.goto(phi, chi, omega, mu)
        self.assertAlmostEqual(phi, self.goni.phi)
        self.assertAlmostEqual(chi, self.goni.chi)
        self.assertAlmostEqual(omega, self.goni.omega)
        self.assertAlmostEqual(mu, self.goni.mu)

        self.goni.goto(0, 0, 0, 0)
        self.goni.goto(phi, chi, 0, mu)
        rot = Rotation.from_matrix(self.goni.R)
        self.goni.goto(rotation=rot)
        np.testing.assert_almost_equal(rot.as_matrix(), self.goni.R, decimal=3)

    def test_small_rot(self):
        # Test that for small rotations, the motor angles are small.
        rot1 = Rotation.from_rotvec((0.5*np.pi/180)*np.array([0,1,0]))
        rot2 = Rotation.from_rotvec((-0.85*np.pi/180)*np.array([1,1,1]/np.sqrt(3)))
        rot = rot1*rot2
        self.goni.goto(rotation=rot)
        self.goni.info
        np.testing.assert_almost_equal(rot.as_matrix(), self.goni.R, decimal=3)
        self.assertLess(self.goni.phi, np.radians(1))
        self.assertLess(self.goni.chi, np.radians(1))
        self.assertLess(self.goni.omega, np.radians(1))
        self.assertLess(self.goni.mu, np.radians(1))

    def test_R(self):
        # Test that the rotation matrix is correct
        rot1 = Rotation.from_rotvec((10*np.pi/180)*np.array([0,1,0]))
        rot2 = Rotation.from_rotvec((np.pi/180)*np.array([1,1,1]/np.sqrt(3)))
        rot = rot1*rot2
        self.goni.goto(rotation=rot)
        np.testing.assert_almost_equal(rot.as_matrix(), self.goni.R, decimal=3)

    def test_numpy(self):
        # Test that numpy rotation matrix works
        rot1 = Rotation.from_rotvec((10*np.pi/180)*np.array([0,1,0]))
        rot2 = Rotation.from_rotvec((np.pi/180)*np.array([1,1,1]/np.sqrt(3)))
        rot = rot1*rot2
        rot = rot.as_matrix()
        self.goni.goto(rotation=rot)
        np.testing.assert_almost_equal(rot, self.goni.R, decimal=3)

    def test_optical_axis(self):
        # Test that the optical axis is correct.
        rot1 = Rotation.from_rotvec((10*np.pi/180)*np.array([0,1,0]))
        rot2 = Rotation.from_rotvec((np.pi/180)*np.array([1,1,1]/np.sqrt(3)))
        rot = rot1*rot2
        self.goni.goto(rotation=rot)
        np.testing.assert_almost_equal(rot.as_matrix(), self.goni.R, decimal=3)

if __name__ == '__main__':
    unittest.main()