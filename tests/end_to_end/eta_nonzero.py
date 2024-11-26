import cProfile
import pstats
import time

import darling
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, find_objects, label
from scipy.spatial.transform import Rotation

from darkmod import laue
from darkmod.beam import GaussianLineBeam
from darkmod import scan
from darkmod.crl import CompundRefractiveLens
from darkmod.crystal import Crystal
from darkmod.deformation import linear_gradient, rotation_gradient, unity_field
from darkmod.detector import Detector
from darkmod.laue import keV_to_angstrom
from darkmod.resolution import DualKentGauss, PentaGauss, TruncatedPentaGauss
from darkmod.utils import crop

plt.style.use("dark_background")
fontsize = 16  # General font size for all text
ticksize = 16  # tick size
plt.rcParams["font.size"] = fontsize
plt.rcParams["xtick.labelsize"] = ticksize
plt.rcParams["ytick.labelsize"] = ticksize

if __name__ == "__main__":

    number_of_lenses = 69
    lens_space = 1600  # microns
    lens_radius = 50  # microns
    refractive_decrement = (2.359 / 2.0) * 1e-6
    magnification = 15.1
    crl = CompundRefractiveLens(
        number_of_lenses, lens_space, lens_radius, refractive_decrement, magnification
    )

    energy = 19.1  # keV
    lambda_0 = laue.keV_to_angstrom(energy)

    # Instantiate a cubic AL crystal (space group 225)
    unit_cell = [4.0493, 4.0493, 4.0493, 90.0, 90.0, 90.0]

    orientation = Rotation.random().as_matrix()
    crystal = Crystal(unit_cell, orientation)

    # remount the crystal to align Q with z-axis
    symmetry_axis = np.array([0, 0, 1])
    crystal.align(symmetry_axis, axis=np.array([0, 0, 1]))
    crystal.align(
        np.array([0, 1, 0]), axis=np.array([0, 1, 0]), transformation_hkl=symmetry_axis
    )
    crystal.align(
        np.array([1, 0, 0]), axis=np.array([1, 0, 0]), transformation_hkl=symmetry_axis
    )
    crystal.remount()  # this updates U and zeros the goniometer.

    # Find the reflection with goniometer motors.
    # theta, eta = crystal.bring_to_bragg(hkl, energy)

    # TODO add some funcitonaliyt to the crystal to figure this
    # out automagically

    # NOTE: The hkl we are probing is NOT alignd with sample-z at this point.
    # this makes thinking about the projection of rotation gradients a bit harder.
    hkl = np.array([-1, -1, 3])
    crystal.goniometer.omega = np.radians(6.431585)
    eta = np.radians(20.232593)
    theta = np.radians(15.416837)
    # Bring the CRL to diffracted beam.
    crl.goto(theta, eta)

    # Discretize the crystal
    xg = np.linspace(-1, 1, 32)  # microns
    yg = np.linspace(-1, 1, 32)  # microns
    zg = np.linspace(-1, 1, 32)  # microns
    dx = xg[1] - xg[0]
    X, Y, Z = np.meshgrid(xg, yg, zg, indexing="ij")

    defgrad = rotation_gradient(
        X.shape,
        rotation_axis=np.array([0, 1, 0]),
        axis=0,
        magnitude = 0.03 * 1e-3,
    )

    #defgrad = unity_field(X.shape)

    spatial_artefact = False
    detector_noise = False
    chimax = 2.5
    phimax = 0.22
    nphi = 11
    nchi = 11

    crystal.discretize(X, Y, Z, defgrad)
    # crystal.write("strain_gradient")

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
    pixel_size = crl.magnification * dx * 0.2

    print("pixel_size", pixel_size)

    detector = Detector.orthogonal_mount(
        crl, pixel_size, det_row_count, det_col_count, super_sampling=1
    )

    beam = GaussianLineBeam(z_std=0.1, energy=energy)

    chimin = -np.abs(chimax)
    phimin = -np.abs(phimax)
    phi_values = np.linspace(phimin, phimax, nphi) * 1e-3
    chi_values = np.linspace(chimin, chimax, nchi) * 1e-3
    print(phi_values)
    print(chi_values)

    for angarr in (phi_values, chi_values):
        assert np.abs(np.median(angarr)) < 1e-8

    dphi = phi_values[1] - phi_values[0]
    dchi = chi_values[1] - chi_values[0]

    pr = cProfile.Profile()
    pr.enable()
    t1 = time.perf_counter()
    mosa = scan.phi_chi(
        hkl,
        phi_values,
        chi_values,
        crystal,
        crl,
        detector,
        beam,
        resolution_function,
        spatial_artefact,
        detector_noise,
    )
    t2 = time.perf_counter()
    pr.disable()
    pr.dump_stats("tmp_profile_dump")
    ps = pstats.Stats("tmp_profile_dump").strip_dirs().sort_stats("cumtime")
    ps.print_stats(15)
    print("\n\nCPU time is : ", t2 - t1, "s")

    print("Subtracting background...")
    background = np.median(mosa[:, 0:5, :, :]).astype(np.uint16)
    print("background", background)
    mosa.clip(background, out=mosa)
    mosa -= background

    mask = np.sum(mosa, axis=(-1, -2))
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    im = ax.imshow(mask)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    mask = mask > np.max(mask) * 0.15
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    im = ax.imshow(mask)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    mu, cov = darling.properties.moments(mosa, coordinates=(phi_values, chi_values))

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    for i in range(2):
        if i == 0:
            im = ax[i].imshow(mu[:, :, i] * 1e3, cmap="jet", vmin=-0.001, vmax=0.001)
        if i == 1:
            im = ax[i].imshow(mu[:, :, i] * 1e3, cmap="jet", vmin=-0.006, vmax=0.006)

        cbar = fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
        ax[i].set_title(r"Mean " + [r"$\phi$", r"$\chi$"][i] + " [mrad]", fontsize=16)
        ax[i].set_xlabel("y [pixels]", fontsize=16)
        if i == 0:
            ax[i].set_ylabel("z [pixels]", fontsize=16)
        ax[i].tick_params(axis="both", which="major", labelsize=16)
        cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()

    phi_mesh, chi_mesh = np.meshgrid(phi_values, chi_values, indexing="ij")
    support = np.sum(mosa, axis=(0, 1))
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    ax1.set_title("Scan Sparsity Pattern")
    im = ax1.pcolormesh(
        chi_mesh * 1e3,
        phi_mesh * 1e3,
        np.log(support.clip(1)),
        cmap="plasma",
        edgecolors="black",
    )
    ax1.set_xlabel("$\chi$ [mrad]")
    ax1.set_ylabel("$\phi$ [mrad]")
    fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if 1:
        # We expect to be able to approximate the mean Q vector along the ray paths.
        def expected_Q(crystal, beam, detector):
            Q_sample_vol = crystal.get_Q_sample(hkl)
            x_lab = crystal.goniometer.R @ crystal._x
            sample_beam_density = beam(x_lab).reshape(crystal._grid_scalar_shape)
            Qlw = Q_sample_vol * sample_beam_density[..., np.newaxis]
            Q_true = np.zeros((mu.shape[0], mu.shape[1], 3))
            for i in range(3):
                Q_true[:, :, i] = detector.render(
                    Qlw[:, :, :, i], crystal.voxel_size, crl, crystal.goniometer.R
                )
            w = detector.render(
                sample_beam_density, crystal.voxel_size, crl, crystal.goniometer.R
            )
            return Q_true / w[:, :, np.newaxis]

        Q_true = expected_Q(crystal, beam, detector)

        mask = np.sum(mosa, axis=(-1, -2))
        mask = mask > np.max(mask) * 0.15

        # sample space recon
        # R @ Q_sample = Q_lab_0 => Q_sample = R_ij.T @ Q_lab_0
        Q_rec = np.zeros((mu.shape[0], mu.shape[1], 3))
        Q_sample_0 = crystal.UB_0 @ hkl
        for i in range(mu.shape[0]):
            for j in range(mu.shape[1]):
                R_s = crystal.goniometer.get_R_top(mu[i, j, 0], mu[i, j, 1])
                Q_rec[i, j] = R_s.T @ Q_sample_0

        Q_true[~mask, :] = np.nan
        Q_rec[~mask, :] = np.nan

        fig, ax = plt.subplots(2, 3, figsize=(14, 9))
        fig.suptitle(
            r"True and reconstructed Q vectors (in lab)",
            fontsize=16,
        )
        for j in range(2):
            Qs = [Q_true, Q_rec]
            _Q = Qs[j]
            title = [r"True ", r"Estimated "][j]
            for i in range(3):
                vmin = np.nanmin(crop(Qs[1][:, :, i], mask))
                vmax = np.nanmax(crop(Qs[1][:, :, i], mask))
                im = ax[j, i].imshow(crop(_Q[:, :, i], mask), vmin=vmin, vmax=vmax)
                cbar = fig.colorbar(im, ax=ax[j, i], fraction=0.046, pad=0.04)
                ax[j, i].set_title(
                    title + [r"Q$_x$", r"Q$_y$", r"Q$_z$"][i],
                    fontsize=16,
                )
                if j == 1:
                    ax[j, i].set_xlabel("y [pixels]", fontsize=16)
                if i == 0:
                    ax[j, i].set_ylabel("z [pixels]", fontsize=16)
                ax[j, i].tick_params(axis="both", which="major", labelsize=16)
                cbar.ax.tick_params(labelsize=16)
        plt.tight_layout()
        plt.show()
