import cProfile
import pstats
import time

import matplotlib.pyplot as plt
import numpy as np

from darkmod import laue
from darkmod.beam import GaussianLineBeam
from darkmod.crl import CompundRefractiveLens
from darkmod.crystal import Crystal
from darkmod.deformation import linear_gradient, unity_field
from darkmod.detector import Detector
from darkmod.laue import keV_to_angstrom
from darkmod.resolution import DualKentGauss, PentaGauss, TruncatedPentaGauss
from darkmod.properties import moments

if __name__ == "__main__":

    number_of_lenses = 69
    lens_space = 1600  # microns
    lens_radius = 50  # microns
    refractive_decrement = (2.359 / 2.0) * 1e-6
    magnification = 15.1
    crl = CompundRefractiveLens(
        number_of_lenses, lens_space, lens_radius, refractive_decrement, magnification
    )
    hkl = np.array([1, -1, 1])
    energy = 17  # keV
    lambda_0 = laue.keV_to_angstrom(energy)

    # Instantiate a cubic diamond crystal (Fd3m space group (space group 227)
    unit_cell = [3.56, 3.56, 3.56, 90.0, 90.0, 90.0]

    orientation = np.eye(3, 3)
    crystal = Crystal(unit_cell, orientation)

    # remount the crystal to align Q with z-axis
    crystal.align(hkl, axis=np.array([0, 0, 1]))
    crystal.remount()  # this updates U.

    # Find the reflection with goniometer motors.
    theta, eta = crystal.bring_to_bragg(hkl, energy)

    # Bring the CRL to diffracted beam.
    crl.goto(theta, eta)

    # Discretize the crystal
    xg = np.linspace(-1, 1, 32)  # microns
    yg = np.linspace(-1, 1, 32)  # microns
    zg = np.linspace(-1, 1, 32)  # microns
    dx = xg[1] - xg[0]
    X, Y, Z = np.meshgrid(xg, yg, zg, indexing="ij")

    defgrad = unity_field(X.shape)
    # defgrad = linear_gradient(
    #    X.shape,
    #    component=(2, 2),
    #    axis=1,
    #    magnitude=0.0002,
    #    )
    crystal.discretize(X, Y, Z, defgrad)
    # crystal.write("test")

    Q_lab = crystal.goniometer.R @ crystal.UB_0 @ hkl

    # Beam divergence params
    desired_FWHM_N = 0.027 * 1e-3

    # Beam wavelength params
    sigma_e = (6 * 1e-5) / (2 * np.sqrt(2 * np.log(2)))
    epsilon = np.random.normal(0, sigma_e, size=(20000,))
    random_energy = energy + epsilon * energy
    sigma_lambda = laue.keV_to_angstrom(random_energy).std()
    mu_lambda = lambda_0

    FWHM_CRL_vertical = 0.556 * 1e-3
    angular_tilt = 0.73 * 1e-3  # perhaps this is what happened in Poulsen 2017?
    # the idea is that a slight horizontal titlt of the CRL will cause the
    # NA in the horixontal plane to decrease which would explain the rolling curve
    # discrepancies.
    dh = (crl.length * np.sin(angular_tilt)) * 1e-6
    FWHM_CRL_horizontal = FWHM_CRL_vertical - 2 * dh

    # # TODO: truncation wont help
    resolution_function = PentaGauss(
        crl.optical_axis,
        1e-9 / (2 * np.sqrt(2 * np.log(2))),
        # desired_FWHM_N / (2 * np.sqrt(2 * np.log(2))),
        desired_FWHM_N / (2 * np.sqrt(2 * np.log(2))),
        FWHM_CRL_horizontal / (2 * np.sqrt(2 * np.log(2))),
        FWHM_CRL_vertical / (2 * np.sqrt(2 * np.log(2))),
        mu_lambda,
        sigma_lambda,
    )
    resolution_function.compile()

    # Detector size
    det_row_count = 512
    det_col_count = 512
    pixel_size = crl.magnification * dx * 0.1 # this will split the phi chi over seevral pixels....

    print("pixel_size", pixel_size)

    detector = Detector.wall_mount(
        crl, pixel_size, det_row_count, det_col_count, super_sampling=1
    )

    beam = GaussianLineBeam(z_std=0.2, energy=energy)

    npoints = 21
    phi_values = np.linspace(-0.05, 0.05, npoints) * 1e-3
    print(phi_values[1]-phi_values[0])
    chi_values = np.linspace(-2, 2, npoints) * 1e-3

    def mosa_scan(
        hkl,
        phi_values,
        chi_values,
        crystal,
        crl,
        detector,
        beam,
        resolution_function,
    ):
        """Simulate a mosaicity scan in phi and chi."""

        phi0 = crystal.goniometer.phi
        chi0 = crystal.goniometer.chi

        mosa = np.zeros(
            (
                detector.det_row_count,
                detector.det_col_count,
                len(phi_values),
                len(chi_values),
            )
        )

        for i in range(npoints):
            for j in range(npoints):

                crystal.goniometer.phi = phi_values[i]
                crystal.goniometer.chi = chi_values[j]

                mosa[:, :, i, j] = crystal.diffract(
                    hkl,
                    resolution_function,
                    crl,
                    detector,
                    beam,
                )

        crystal.goniometer.phi = phi0
        crystal.goniometer.chi = chi0

        #mosa /= np.max(mosa)
        #mosa *= 64000
        #mosa = mosa.round().astype(np.uint16)

        return mosa


    pr = cProfile.Profile()
    pr.enable()
    t1 = time.perf_counter()
    mosa = mosa_scan(
        hkl,
        phi_values,
        chi_values,
        crystal,
        crl,
        detector,
        beam,
        resolution_function,
    )
    t2 = time.perf_counter()
    pr.disable()
    pr.dump_stats('tmp_profile_dump')
    ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
    ps.print_stats(15)
    print('\n\nCPU time is : ', t2-t1, 's')

    if 1:
        mu, cov = moments( mosa.astype(np.float32), coordinates=(phi_values, chi_values))

        plt.style.use('dark_background')
        fig, ax = plt.subplots(1, 2, figsize=(9,6))
        for i in range(2):
            im = ax[i].imshow(mu[:,:,i])
            fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
        plt.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    im = ax.imshow(mosa[459, 218, :, :])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    im = ax.imshow(mosa[458, 218, :, :])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    im = ax.imshow(mosa[457, 218, :, :])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()