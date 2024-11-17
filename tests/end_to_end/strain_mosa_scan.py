import cProfile
import pstats
import time

import darling
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, find_objects, label

from darkmod import laue
from darkmod.beam import GaussianLineBeam
from darkmod.crl import CompundRefractiveLens
from darkmod.crystal import Crystal
from darkmod.deformation import linear_gradient, rotation_gradient, unity_field
from darkmod.detector import Detector
from darkmod.laue import keV_to_angstrom
from darkmod.resolution import DualKentGauss, PentaGauss, TruncatedPentaGauss

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

    # defgrad = unity_field(X.shape)
    defgrad = linear_gradient(
        X.shape,
        component=(2, 2),
        axis=1,
        magnitude=0.0001, # 0.01 % strain
    )
    # defgrad = rotation_gradient(
    # X.shape,
    # rotation_axis=np.array([1, 1, 1]),
    # axis=1,
    # magnitude=1e-4,
    # )
    crystal.discretize(X, Y, Z, defgrad)
    crystal.write("strain_gradient")

    Q_lab = crystal.goniometer.R @ crystal.UB_0 @ hkl
    d_0 = (2 * np.pi) / np.linalg.norm(Q_lab)

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
    det_row_count = 256
    det_col_count = 256
    pixel_size = (
        crl.magnification * dx * 0.17
    )  # this will split the phi chi over seevral pixels....

    print("pixel_size", pixel_size)

    detector = Detector.wall_mount(
        crl, pixel_size, det_row_count, det_col_count, super_sampling=1
    )

    beam = GaussianLineBeam(z_std=0.1, energy=energy)

    dth = 0.4
    theta_values = np.arange(-1, 1+ dth/2., dth) * 1e-3
    dphi = 0.02
    phi_values = np.arange(-0.05, 0.05 + dphi/2., dphi) * 1e-3
    dchi = 0.6
    chi_values = np.arange(-2, 2+ dchi/2., dchi) * 1e-3

    print( 'Number of scan points is: ', (len( theta_values ) * len( phi_values ) * len( chi_values )) )

    PHI, CHI, THETA = np.meshgrid(
        phi_values, chi_values, crl.theta + theta_values, indexing="ij"
    )

    def strain_mosa_scan(
        hkl,
        theta_values,
        phi_values,
        chi_values,
        crystal,
        crl,
        detector,
        beam,
        resolution_function,
        signal_to_noise=1000,
        verbose=False,
    ):
        """Simulate a strain-mosaicity scan in theta, phi and chi."""

        phi0 = crystal.goniometer.phi
        chi0 = crystal.goniometer.chi
        th0 = crl.theta

        strain_mosa = np.zeros(
            (
                detector.det_row_count,
                detector.det_col_count,
                len(theta_values),
                len(phi_values),
                len(chi_values),
            )
        )

        for i in range(len(theta_values)):

            crl.goto(theta=th0 + theta_values[i], eta=crl.eta)
            detector.remount_to_crl(crl)
            resolution_function.theta_shift(th0 + theta_values[i])

            for j in range(len(phi_values)):
                for k in range(len(chi_values)):

                    if verbose:
                        print(theta_values[i], phi_values[j], chi_values[k])
                    crystal.goniometer.phi = phi_values[j]
                    crystal.goniometer.chi = chi_values[k]

                    strain_mosa[:, :, i, j, k] = crystal.diffract(
                        hkl,
                        resolution_function,
                        crl,
                        detector,
                        beam,
                    )

        crl.goto(theta=th0, eta=crl.eta)
        detector.remount_to_crl(crl)
        resolution_function.theta_shift(th0)
        crystal.goniometer.phi = phi0
        crystal.goniometer.chi = chi0

        noise_level = np.max(strain_mosa) / signal_to_noise
        strain_mosa += np.abs(np.random.normal(0, noise_level, size=strain_mosa.shape))
        strain_mosa /= np.max(strain_mosa)
        strain_mosa *= 64000
        strain_mosa = strain_mosa.round().astype(np.uint16)

        return strain_mosa

    pr = cProfile.Profile()
    pr.enable()
    t1 = time.perf_counter()
    strain_mosa = strain_mosa_scan(
        hkl,
        theta_values,
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
    pr.dump_stats("tmp_profile_dump")
    ps = pstats.Stats("tmp_profile_dump").strip_dirs().sort_stats("cumtime")
    ps.print_stats(15)
    print("\n\nCPU time is : ", t2 - t1, "s")

    if 1:
    


        # a,b,m,n,o = strain_mosa.shape
        # ss = strain_mosa[a // 2, b // 2, m // 2, :, :]
        # fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        # im = ax.imshow(ss, cmap="magma")
        # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # cbar.ax.tick_params(labelsize=32)
        # ax.tick_params(axis="both", which="major", labelsize=26)
        # ax.set_xlabel("$\phi$ [mrad]", fontsize=32)
        # ax.set_ylabel("$\chi$ [mrad]", fontsize=32)
        # thstr = str(
        #     np.degrees(crl.theta + theta_values[len(theta_values) // 2]).round(2)
        # )
        # ax.set_title(
        #     "$\phi$-$\chi$ distribution in central pixel [counts], $\\theta$=" + thstr,
        #     fontsize=32,
        # )
        # plt.tight_layout()

        # ss = strain_mosa[a // 2, b // 2, :, n // 2, :]
        # fig, ax = plt.subplots(1, 1, figsize=(12, 10))
        # im = ax.imshow(ss, cmap="magma")
        # cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # cbar.ax.tick_params(labelsize=32)
        # ax.tick_params(axis="both", which="major", labelsize=26)
        # ax.set_xlabel("$\\theta$ [mrad]", fontsize=32)
        # ax.set_ylabel("$\chi$ [mrad]", fontsize=32)
        # phistr = str(np.degrees(phi_values[len(phi_values) // 2]).round(2))
        # ax.set_title(
        #     "$\\theta$-$\chi$ distribution in central pixel [counts], $\phi$=" + phistr,
        #     fontsize=32,
        # )
        # plt.tight_layout()

        mu, cov = darling.properties.moments(
            strain_mosa, coordinates=(theta_values, phi_values, chi_values)
        )


        fig, ax = plt.subplots(1, 3, figsize=(16, 6))
        for i in range(3):
            if i == 0:
                im = ax[i].imshow(mu[:, :, i] * 1e3, cmap="jet", vmin=-0.05, vmax=0.05)
            if i == 1:
                im = ax[i].imshow(mu[:, :, i] * 1e3, cmap="jet", vmin=-0.05, vmax=0.05)
            if i == 2:
                im = ax[i].imshow(mu[:, :, i] * 1e3, cmap="jet", vmin=-0.03, vmax=0.03)

            cbar = fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
            ax[i].set_title(
                r"Mean " + [r"$\theta$", r"$\phi$", r"$\chi$"][i] + " [mrad]", fontsize=32
            )
            ax[i].set_xlabel("y [pixels]", fontsize=16)
            if i == 0:
                ax[i].set_ylabel("z [pixels]", fontsize=16)
            ax[i].tick_params(axis="both", which="major", labelsize=16)
            cbar.ax.tick_params(labelsize=16)
        plt.tight_layout()


        # The central layer of the voxel volume flipped and normalised
        # this represents what would be seen in terms of Q lab if one
        # had the proper degrees of freedom.
        Q_true = crystal.get_Q_lab(hkl)
        Q_true = np.flip( Q_true[:, :, len(zg) // 2, :], axis=1)

        # The reconstructed Q directions are found as
        # Qrec_lab = R @ Q0_sample
        # i.e given that Q0_sample gives diffraction we must have that
        # Qrec_lab is at R @ Q0_sample
        phi0 = crystal.goniometer.phi
        chi0 = crystal.goniometer.chi

        Q_rec = np.zeros((mu.shape[0], mu.shape[1], 3))
        Q_sample_0 = crystal.UB_0 @ hkl
        Q_sample_0 = Q_sample_0 / np.linalg.norm(Q_sample_0)
        for i in range(mu.shape[0]):
            for j in range(mu.shape[1]):
                crystal.goniometer.phi = mu[i, j, 1]
                crystal.goniometer.chi = mu[i, j, 2]
                Q_rec[i, j] = crystal.goniometer.R @ Q_sample_0
                d_rec = lambda_0 / (2*np.sin(crl.theta + mu[i, j, 0]))
                Q_rec[i, j] *= ( 2 * np.pi / d_rec )
        d_field_rec = (2*np.pi) / np.linalg.norm(Q_rec, axis=-1)
        d_field_true = (2*np.pi) / np.linalg.norm(Q_true, axis=-1)
        hkl_strain_rec = (d_field_rec - d_0) / d_0
        hkl_strain_true =  (d_field_true - d_0) / d_0

        crystal.goniometer.phi = phi0
        crystal.goniometer.chi = chi0

        fig, ax = plt.subplots(2, 3, figsize=(16, 12))
        fig.suptitle(
            r"True and reconstructed Q vectors (in lab)",
            fontsize=32,
        )
        for j in range(2):
            Qs = [Q_true, Q_rec]
            _Q = Qs[j]
            title = [r"True ", r"Estimated "][j]
            for i in range(3):
                im = ax[j, i].imshow(_Q[:, :, i])
                cbar = fig.colorbar(im, ax=ax[j, i], fraction=0.046, pad=0.04)
                ax[j, i].set_title(
                    title
                    + [r"$Q_x$", r"$Q_y$", r"$Q_z$"][i],
                    fontsize=32,
                )
                if j == 1:
                    ax[j, i].set_xlabel("y [pixels]", fontsize=16)
                ax[j, i].set_ylabel("z [pixels]", fontsize=16)
                ax[j, i].tick_params(axis="both", which="major", labelsize=16)
                cbar.ax.tick_params(labelsize=16)
        plt.tight_layout()


        fig, ax = plt.subplots(1, 2, figsize=(14,7))
        fig.suptitle(
            r"True and reconstructed strain in hkl direction: ($d_0$ - d) / $d_0$ )",
            fontsize=32,
        )
        strains = [hkl_strain_true, hkl_strain_rec]
        cmaps = ['viridis','viridis']
        titles = ['True', 'Recon']
        for i in range(2):
            im = ax[i].imshow(strains[i], cmap=cmaps[i])
            fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
            ax[i].set_title(titles[i])
        ax[0].set_xlabel("y [pixels]", fontsize=16)
        ax[1].set_xlabel("y [pixels]", fontsize=16)
        ax[0].set_ylabel("z [pixels]", fontsize=16)
        plt.tight_layout()
    
        plt.show()


