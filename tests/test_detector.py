import unittest

import matplotlib.pyplot as plt
import numpy as np

from darkmod import laue
from darkmod.beam import GaussianLineBeam
from darkmod.crl import CompundRefractiveLens
from darkmod.crystal import Crystal
from darkmod.detector import Detector
from darkmod.resolution import PentaGauss


class TestDetector(unittest.TestCase):
    def setUp(self):
        deformation_gradient_field = np.zeros((16, 16, 16, 3, 3))
        for i in range(3):
            deformation_gradient_field[..., i, i] += 1
        voxel_size = 0.1204845  # microns

        dx, dy, dz = np.array(deformation_gradient_field.shape[0:3]) * voxel_size
        xg = np.arange(0, dx, voxel_size)
        yg = np.arange(0, dy, voxel_size)
        zg = np.arange(0, dz, voxel_size)
        xg -= np.median(xg)
        yg -= np.median(yg)
        zg -= np.median(zg)
        Xgrid, Ygrid, Zgrid = np.meshgrid(xg, yg, zg, indexing="ij")

        energy = 17.1  # keV
        eta = np.radians(0)
        unit_cell = [4.0493, 4.0493, 4.0493, 90.0, 90.0, 90.0]  # angstrom

        self.hkl = np.array([1, 1, 1])
        self.crystal = Crystal(unit_cell, np.eye(3, 3))
        self.crystal.align(self.hkl, axis=np.array([0, 0, 1]))
        self.crystal.remount()

        number_of_lenses = 88
        lens_space = 1600  # microns
        lens_radius = 50  # microns
        magnification = 18.4742
        Z = 4  # atomic number, berillium
        rho = 1.845  # density, berillium, g/cm^3
        A = 9.0121831  # atomic mass number, berillium, g/mol

        z_std = 0.25  # microns

        # Beam divergence params
        beam_FWHM_vertical = 0.027 * 1e-3
        beam_FWHM_horizontal = 1e-9

        # Beam wavelength broadening
        sigma_e = (6 * 1e-5) / (2 * np.sqrt(2 * np.log(2)))

        # crl acceptance
        FWHM_CRL_vertical = 0.556 * 1e-3
        FWHM_CRL_horizontal = FWHM_CRL_vertical

        # Detector size
        det_row_count = 2048
        det_col_count = 2048
        pixel_size = 0.65
        super_sampling = 2
        dynamic_range = 2**16 - 1
        exposure = 35000
        psf_width = 1
        noise = True

        # np.linalg.inv(crystal.U @ crystal.B) @ np.array([0,0,1])
        delta = laue.refractive_decrement(Z, rho, A, energy)
        self.crl = CompundRefractiveLens(
            number_of_lenses, lens_space, lens_radius, delta, magnification
        )

        self.crystal.discretize(
            Xgrid,
            Ygrid,
            Zgrid,
            deformation_gradient_field,
        )

        lambda_0 = laue.keV_to_angstrom(energy)
        self.beam = GaussianLineBeam(z_std=z_std, energy=energy)  # 100 nm = 0.1 microns

        d = np.pi * 2 / np.linalg.norm(self.crystal.U @ self.crystal.B @ self.hkl)
        self.theta = np.arcsin(laue.keV_to_angstrom(self.beam.energy) / (2 * d))
        self.crl.goto(self.theta, eta)

        epsilon = np.random.normal(0, sigma_e, size=(20000,))
        random_energy = energy + epsilon * energy
        sigma_lambda = laue.keV_to_angstrom(random_energy).std()

        self.resolution_function = PentaGauss(
            self.crl.optical_axis,
            beam_FWHM_horizontal / (2 * np.sqrt(2 * np.log(2))),
            # desired_FWHM_N / (2 * np.sqrt(2 * np.log(2))),
            beam_FWHM_vertical / (2 * np.sqrt(2 * np.log(2))),
            FWHM_CRL_horizontal / (2 * np.sqrt(2 * np.log(2))),
            FWHM_CRL_vertical / (2 * np.sqrt(2 * np.log(2))),
            lambda_0,
            sigma_lambda,
        )
        self.resolution_function.compile()

        self.detector = Detector.wall_mount(
            self.crl,
            pixel_size,
            det_row_count,
            det_col_count,
            super_sampling=super_sampling,
            exposure=exposure,
            dynamic_range=dynamic_range,
            psf_width=psf_width,
            noise=noise,
        )

    def test_simple_frame(self):
        image = self.crystal.diffract(
            self.hkl,
            self.resolution_function,
            self.crl,
            self.detector,
            self.beam,
        )

        plt.style.use("dark_background")
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        im = ax.imshow(image)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()

        self.crystal.goniometer.mu += self.theta

        image = self.crystal.diffract(
            self.hkl,
            self.resolution_function,
            self.crl,
            self.detector,
            self.beam,
        )

        plt.style.use("dark_background")
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        im = ax.imshow(image)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    unittest.main()
