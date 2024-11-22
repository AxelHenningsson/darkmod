import cProfile
import pstats
import time

import matplotlib.pyplot as plt
import numpy as np

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

    #defgrad = unity_field(X.shape)
    # defgrad = linear_gradient(
    #     X.shape,
    #     component=(2, 2),
    #     axis=1,
    #     magnitude=0.0002,
    # )
    defgrad = rotation_gradient(
    X.shape,
    rotation_axis=np.array([0, 1, 0]),
    axis=1,
    magnitude=1e-4,
    )
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

    npoints = 11
    phi_values = np.linspace(-0.05, 0.05, npoints) * 1e-3
    print(phi_values[1] - phi_values[0])
    chi_values = np.linspace(-2, 2, npoints) * 1e-3

    PHI,CHI = np.meshgrid(phi_values, chi_values, indexing='ij')

    def mosa_scan(
        hkl,
        phi_values,
        chi_values,
        crystal,
        crl,
        detector,
        beam,
        resolution_function,
        signal_to_noise=100,
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

        for i in range(len(phi_values)):
            for j in range(len(chi_values)):

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

        noise_level = np.max(mosa) / signal_to_noise
        mosa += np.abs(np.random.normal(0, noise_level, size=mosa.shape))
        mosa /= np.max(mosa)
        mosa *= 64000
        mosa = mosa.round().astype(np.uint16)

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
    pr.dump_stats("tmp_profile_dump")
    ps = pstats.Stats("tmp_profile_dump").strip_dirs().sort_stats("cumtime")
    ps.print_stats(15)
    print("\n\nCPU time is : ", t2 - t1, "s")

    if 1:

        import darling
        from scipy.ndimage import (binary_dilation, binary_fill_holes,
                                   find_objects, label)

        plt.style.use('dark_background')
        fig, ax = plt.subplots(1, 1, figsize=(16, 16))
        im = ax.imshow(mosa[:,:, 9, 3], cmap='magma')
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xlabel("y [pixels]", fontsize=32)
        ax.set_ylabel("z [pixels]", fontsize=32)
        ax.tick_params(axis="both", which="major", labelsize=26)
        phistr = str(np.round(phi_values[9]*1e3,2))
        plt.tight_layout()

        mask = mosa.sum(axis=(2, 3))
        mask = mask > np.max(mask) * 0.1
        labeled_array, num_features = label(mask)
        region_slices = find_objects(labeled_array)
        region_sizes = [
            np.sum(labeled_array[sl] == (i + 1)) for i, sl in enumerate(region_slices)
        ]
        largest_region_index = np.argmax(region_sizes) + 1
        largest_region_mask = labeled_array == largest_region_index
        mask = largest_region_mask
        mask = binary_fill_holes(mask)
        mask = binary_dilation(mask, iterations=2)

        def crop(array, mask):
            non_zero_indices = np.argwhere(mask)
            top_left = non_zero_indices.min(axis=0)
            bottom_right = non_zero_indices.max(axis=0) + 1
            return array[top_left[0] : bottom_right[0], top_left[1] : bottom_right[1]]

        plt.style.use("dark_background")
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))
        im = ax.imshow(mask)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

        mu, cov = darling.properties.moments(mosa, coordinates=(phi_values, chi_values))

        plt.style.use("dark_background")
        fig, ax = plt.subplots(1, 2, figsize=(22, 14))
        for i in range(2):
            if i==0:
                im = ax[i].imshow(mu[:, :, i] * 1e3, cmap="jet", vmin=-0.05, vmax=0.05)
            if i==1:
                im = ax[i].imshow(mu[:, :, i] * 1e3, cmap="jet", vmin=-0.07, vmax=0.07)

            cbar = fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
            ax[i].set_title(
                r"Mean " + [r"$\phi$", r"$\chi$"][i] + " [mrad]", fontsize=32
            )
            ax[i].set_xlabel("y [pixels]", fontsize=32)
            if i == 0:
                ax[i].set_ylabel("z [pixels]", fontsize=32)
            ax[i].tick_params(
                axis="both", which="major", labelsize=26
            )
            cbar.ax.tick_params(labelsize=32)
        plt.tight_layout()

        # We expect to be able to approximate the mean Q vector along the ray paths.
        Q_lab_vol = crystal.get_Q_lab(hkl)

        Qx = detector.render(Q_lab_vol[:,:,:,0], crystal.voxel_size, crl, crystal.goniometer.R)
        Qy = detector.render(Q_lab_vol[:,:,:,1], crystal.voxel_size, crl, crystal.goniometer.R)
        Qz = detector.render(Q_lab_vol[:,:,:,2], crystal.voxel_size, crl, crystal.goniometer.R)

        sample_density = np.ones((Q_lab_vol.shape[0], Q_lab_vol.shape[1], Q_lab_vol.shape[2])) # TODO: should be beam density?
        sample = detector.render(sample_density, crystal.voxel_size, crl, crystal.goniometer.R)
        Q_true = np.stack((Qx, Qy, Qz), axis=-1) / sample[:, :, np.newaxis]
        Q_true /= np.linalg.norm(Q_true, axis=-1)[:, :, np.newaxis]

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
                crystal.goniometer.phi = mu[i, j, 0]
                crystal.goniometer.chi = mu[i, j, 1]
                Q_rec[i, j] = crystal.goniometer.R @ Q_sample_0

        crystal.goniometer.phi = phi0
        crystal.goniometer.chi = chi0

        plt.style.use("dark_background")
        q_lab = Q_lab / np.linalg.norm(Q_lab)
        fig, ax = plt.subplots(2, 3, figsize=(28, 16))
        fig.suptitle(r'True and reconstructed $\Delta$ Q/||Q|| vectors (in lab) [1e-5]', fontsize=32)
        for j in range(2):
            Qs = [Q_true*1e5 - q_lab*1e5, Q_rec*1e5 - q_lab*1e5]
            _Q = Qs[j]
            title = [r"True ", r"Estimated "][j]
            for i in range(3):
                vmin = -4.9
                vmax = 4.9
                if i==2:
                    vmin = -1
                    vmax = 1 
                im = ax[j, i].imshow(_Q[:, :, i], vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(im, ax=ax[j, i], fraction=0.046, pad=0.04)
                ax[j, i].set_title(
                    title + [r"$\Delta$ q$_x$", r"$\Delta$ q$_y$", r"$\Delta$ q$_z$"][i], fontsize=32
                )
                if j==1: ax[j, i].set_xlabel("y [pixels]", fontsize=32)
                ax[j, i].set_ylabel("z [pixels]", fontsize=32)
                ax[j, i].tick_params(axis="both", which="major", labelsize=26)
                cbar.ax.tick_params(labelsize=32)
        plt.tight_layout()

    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    im = ax.pcolormesh(PHI*1e3, CHI*1e3, mosa[mosa.shape[0]//2, mosa.shape[1]//2, :, :], cmap='magma')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=32)
    ax.tick_params(axis="both", which="major", labelsize=26)
    ax.set_xlabel("$\phi$ [mrad]", fontsize=32)
    ax.set_ylabel("$\chi$ [mrad]", fontsize=32)
    ax.set_title("$\phi$-$\chi$ distirbution in central pixel [counts]", fontsize=32)
    plt.tight_layout()



    fig, ax = plt.subplots(1, 1, figsize=(16, 14))
    im = ax.pcolormesh(PHI*1e3, CHI*1e3, mosa[mosa.shape[0]//2, 1, :, :], cmap='magma')
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=32)
    ax.tick_params(axis="both", which="major", labelsize=26)
    ax.set_xlabel("$\phi$ [mrad]", fontsize=32)
    ax.set_ylabel("$\chi$ [mrad]", fontsize=32)
    ax.set_title("$\phi$-$\chi$ distirbution in central pixel [counts]", fontsize=32)
    plt.tight_layout()
    plt.show()

