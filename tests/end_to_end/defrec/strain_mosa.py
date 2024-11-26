import cProfile
import pstats
import time

import darling
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, find_objects, label
from scipy.spatial.transform import Rotation

from darkmod import laue, scan
from darkmod.beam import GaussianLineBeam
from darkmod.crl import CompundRefractiveLens
from darkmod.crystal import Crystal
from darkmod.deformation import (
    linear_gradient,
    multi_gradient,
    rotation_gradient,
    straight_edge_dislocation,
    unity_field,
)
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

    #
    # Reflection set of rank : 3
    # condition number       : 3.0
    # Diffracting at eta     : 20.2326
    # Symmetry axis          :  0, 0, 1
    #                 h	  k	  l	omega_1	omega_2	eta_1	eta_2	theta	2 theta
    # reflection 40	-1.0	-1.0	3.0	28.00	285.14	20.23	-20.23	15.42	30.83
    # reflection 45	1.0	-1.0	3.0	298.00	195.14	20.23	-20.23	15.42	30.83
    # reflection 46	-1.0	1.0	3.0	118.00	15.14	20.23	-20.23	15.42	30.83
    # reflection 49	1.0	1.0	3.0	105.14	208.00	-20.23	20.23	15.42	30.83
    #
    #

    # NOTE: The hkl we are probing is NOT alignd with sample-z at this point.
    # this makes thinking about the projection of rotation gradients a bit harder.

    spatial_artefact = False
    detector_noise = False
    reflection = 1

    if reflection == 1:  # Reflection #1
        hkl = np.array([-1, -1, 3])
        crystal.goniometer.omega = np.radians(6.431585)
        thmax = 0.6
        phimax = 1.4
        chimax = 2.4
        ntheta = 11
        nphi = 61
        nchi = 61
    elif reflection == 2:  # Reflection #2
        hkl = np.array([1.0, -1.0, 3.0])
        crystal.goniometer.omega = np.radians(276.431585)
        thmax = 0.6
        phimax = 1.4
        chimax = 2.4
        ntheta = 11
        nphi = 61
        nchi = 21
    elif reflection == 3:  # Reflection #3
        hkl = np.array([-1.0, 1.0, 3.0])
        crystal.goniometer.omega = np.radians(96.431585)
        thmax = 0.6
        phimax = 1.4
        chimax = 2.4
        ntheta = 11
        nphi = 61
        nchi = 21

    eta = np.radians(20.232593)
    theta = np.radians(15.416837)
    # Bring the CRL to diffracted beam.
    crl.goto(theta, eta)

    # Discretize the crystal
    xg = np.linspace(-1, 1, 33)  # microns
    yg = np.linspace(-1, 1, 33)  # microns
    zg = np.linspace(-1, 1, 33)  # microns
    dx = xg[1] - xg[0]
    zg = zg[len(zg)//2  - 1: len(zg)//2 + 2]
    X, Y, Z = np.meshgrid(xg, yg, zg, indexing="ij")

    # OK !
    # defgrad = unity_field(X.shape)

    # OK !
    # defgrad = linear_gradient(
    #     X.shape,
    #     component=(2, 2),
    #     axis=1,
    #     magnitude=0.001,
    # )

    # OK !
    # defgrad = rotation_gradient(
    # X.shape,
    # rotation_axis=np.array([0, 1, 0]),
    # axis=1,
    # magnitude=0.1 * 1e-3,
    # )

    # OK !
    # defgrad = multi_gradient(
    #     X.shape,
    #     component=(2, 2),
    #     rotation_axis=np.array([0, 1, 0]),
    #     axis=1,
    #     rot_magnitude = 0.1 * 1e-3,
    #     strain_magnitude = 10 * 1e-4,
    # )

    # OK !
    # defgrad = multi_gradient(
    #     X.shape,
    #     component=(2, 2),
    #     rotation_axis=np.array([1, 1, 1]),
    #     axis=1,
    #     rot_magnitude = 0.1 * 1e-3,
    #     strain_magnitude = 10 * 1e-4,
    # )
    
    # TODO: fix singularity....
    defgrad = straight_edge_dislocation((X, Y, Z), x0=[[0, 0, 0]])

    theta_values = np.linspace(-np.abs(thmax), thmax, ntheta) * 1e-3
    phi_values = np.linspace(-np.abs(phimax), phimax, nphi) * 1e-3
    chi_values = np.linspace(-np.abs(chimax), chimax, nchi) * 1e-3

    for angarr in (theta_values, phi_values, chi_values):
        assert np.abs(np.median(angarr)) < 1e-8, angarr

    crystal.discretize(X, Y, Z, defgrad)
    crystal.write("straight_edge_dislocation")

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

    detector = Detector.orthogonal_mount(
        crl, pixel_size, det_row_count, det_col_count, super_sampling=1
    )

    beam = GaussianLineBeam(z_std=0.1, energy=energy)

    dth = theta_values[1] - theta_values[0]
    dphi = phi_values[1] - phi_values[0]
    dchi = chi_values[1] - chi_values[0]

    print(
        "Number of scan points is: ",
        (len(theta_values) * len(phi_values) * len(chi_values)),
    )
    print("theta resolution is: ", dth * 1e3, "mrad")
    print("phi resolution is: ", dphi * 1e3, "mrad")
    print("chi resolution is: ", dchi * 1e3, "mrad")

    pr = cProfile.Profile()
    pr.enable()
    t1 = time.perf_counter()
    strain_mosa = scan.theta_phi_chi(
        hkl,
        theta_values,
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
    background = np.median(strain_mosa[:, 0:5, :, :]).astype(np.uint16)
    print("background", background)
    strain_mosa.clip(background, out=strain_mosa)
    strain_mosa -= background

    mask = np.sum(strain_mosa, axis=(-1, -2, -3))
    # fig, ax = plt.subplots(1, 1, figsize=(7,7))
    # im = ax.imshow(mask)
    # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # plt.tight_layout()


    mask = mask > np.max(mask) * 0.05

    mu, cov = darling.properties.moments(
        strain_mosa, coordinates=(theta_values, phi_values, chi_values)
    )

    fig, ax = plt.subplots(1, 3, figsize=(16, 6))
    _mu = crop(mu, mask)
    for i in range(3):
        # im = ax[i].imshow(_mu[:, :, i] * 1e3, cmap="jet")

        if i == 0:
            im = ax[i].imshow(_mu[:, :, i] * 1e3, cmap="jet", vmin=-0.006, vmax=0.006)
        if i == 1:
            im = ax[i].imshow(_mu[:, :, i] * 1e3, cmap="jet", vmin=-0.22, vmax=0.22)
        if i == 2:
            im = ax[i].imshow(_mu[:, :, i] * 1e3, cmap="jet", vmin=-0.22, vmax=0.22)

        cbar = fig.colorbar(im, ax=ax[i], fraction=0.046 / 2.0, pad=0.04)
        ax[i].set_title(
            r"Mean " + [r"$\Delta \theta$", r"$\phi$", r"$\chi$"][i] + " [mrad]"
        )
        ax[i].set_xlabel("y [pixels]")
        if i == 0:
            ax[i].set_ylabel("z [pixels]")
        ax[i].tick_params(axis="both", which="major", labelsize=16)
        cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()

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

    # sample space recon
    # R @ Q_sample = Q_lab_0 => Q_sample = R_ij.T @ Q_lab_0
    Q_rec = np.zeros((mu.shape[0], mu.shape[1], 3))
    q_sample_0 = crystal.UB_0 @ hkl / np.linalg.norm(crystal.UB_0 @ hkl)

    # Rom = crystal.goniometer.get_R_omega(crystal.goniometer.omega)
    # k_sample_hat = Rom @ np.array([1, 0, 0])
    # rotvec = np.cross(k_sample_hat, q_sample_0)
    # rotvec /= np.linalg.norm(rotvec)
    M = resolution_function._get_M()

    # sample space recon
    # R @ Q_sample = Q_lab_0 => Q_sample = R_ij.T @ Q_lab_0
    Q_rec = np.zeros((mu.shape[0], mu.shape[1], 3))
    q_sample_0 = crystal.UB_0 @ hkl / np.linalg.norm(crystal.UB_0 @ hkl)
    R_omega = crystal.goniometer.get_R_omega(crystal.goniometer.omega)
    print(crystal.goniometer.info)
    rotation_eta = Rotation.from_rotvec(np.array([1, 0, 0]) * (crl.eta))

    for i in range(mu.shape[0]):
        for j in range(mu.shape[1]):
            R_s = crystal.goniometer.get_R_top(mu[i, j, 1], mu[i, j, 2])
            d_rec = lambda_0 / (2 * np.sin(crl.theta + mu[i, j, 0]))
            Q_norm = 2 * np.pi / d_rec

            # TODO:
            # approximate:
            # _x = np.array([1, 0, 0, 0, 2 * mu[i, j, 0]])
            # Q_sample_0 = R_omega.T @ (M @ _x)

            # exact:
            rotation_th = Rotation.from_rotvec(
                np.array([0, 1, 0]) * (-2 * (crl.theta + mu[i, j, 0]))
            )
            rot = (rotation_eta * rotation_th).as_matrix()
            _Q = rot @ np.array([1, 0, 0]) - np.array([1, 0, 0])
            _Q = _Q / np.linalg.norm(_Q)
            Q_sample_0 = R_omega.T @ _Q * Q_norm

            # print(np.abs(Q_sample_0 - crystal.UB_0 @ hkl))

            # angle = mu[i, j, 0]
            # Ry_dth = Rotation.from_rotvec(rotvec * angle).as_matrix()
            # s,c = np.sin(mu[i, j, 0]), np.cos(mu[i, j, 0])
            # Ry_dth = np.array([[c,0,s],[0,1,0],[-s,0,c]])
            # Q_rec[i, j] = R_s.T @ (Ry_dth @.T q_sample_0 * Q_norm)
            Q_rec[i, j] = R_s.T @ Q_sample_0

    # Q_true[~mask,:]= np.nan
    # Q_rec[~mask,:]= np.nan

    d_field_rec = (2 * np.pi) / np.linalg.norm(Q_rec, axis=-1)
    d_field_true = (2 * np.pi) / np.linalg.norm(Q_true, axis=-1)
    hkl_strain_rec = (d_field_rec - d_0) / d_0
    hkl_strain_true = (d_field_true - d_0) / d_0

    fig, ax = plt.subplots(2, 3, figsize=(16, 12))
    fig.suptitle(
        r"True and reconstructed Q vectors (in lab)",
    )
    for j in range(2):
        Qs = [Q_true, Q_rec]
        _Q = Qs[j]
        _Q[np.abs(_Q) < 1e-12] = 0
        title = [r"True ", r"Estimated "][j]
        for i in range(3):
            vmin = np.nanmin(crop(Qs[1][:, :, i], mask))
            vmax = np.nanmax(crop(Qs[1][:, :, i], mask))
            im = ax[j, i].imshow(crop(_Q[:, :, i], mask), vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(im, ax=ax[j, i], fraction=0.046 / 2.0, pad=0.04)
            ax[j, i].set_title(
                title + [r"$Q_x$", r"$Q_y$", r"$Q_z$"][i],
            )
            if j == 1:
                ax[j, i].set_xlabel("y [pixels]")
            ax[j, i].set_ylabel("z [pixels]")
            ax[j, i].tick_params(axis="both", which="major", labelsize=16)
            cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()

    fig, ax = plt.subplots(1, 2, figsize=(14, 7))
    fig.suptitle(
        r"True and reconstructed strain in hkl direction: ($d_0$ - d) / $d_0$ )",
    )
    strains = [hkl_strain_true, hkl_strain_rec]
    cmaps = ["RdBu_r", "RdBu_r"]
    titles = ["True", "Recon"]
    for i in range(2):
        im = ax[i].imshow(
            crop(strains[i], mask), cmap=cmaps[i], vmin=-3 * 1e-5, vmax=3 * 1e-5
        )
        fig.colorbar(im, ax=ax[i], fraction=0.046 / 2.0, pad=0.04)
        ax[i].set_title(titles[i])
    ax[0].set_xlabel("y [pixels]")
    ax[1].set_xlabel("y [pixels]")
    ax[0].set_ylabel("z [pixels]")
    plt.tight_layout()

    phi_mesh, chi_mesh = np.meshgrid(phi_values, chi_values, indexing="ij")
    support = np.sum(strain_mosa, axis=(0, 1))[len(theta_values) // 2, :, :]
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    ax1.set_title("$\phi$-$\chi$  Log-Sparsity Scan Pattern")
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

    theta_mesh, chi_mesh = np.meshgrid(theta_values, chi_values, indexing="ij")
    support = np.sum(strain_mosa, axis=(0, 1))[:, len(phi_values) // 2, :]
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 8))
    ax1.set_title("$\\theta$-$\chi$ Log-Sparsity Scan Pattern")
    im = ax1.pcolormesh(
        chi_mesh * 1e3,
        theta_mesh * 1e3,
        np.log(support.clip(1)),
        cmap="plasma",
        edgecolors="black",
    )
    ax1.set_xlabel("$\chi$ [mrad]")
    ax1.set_ylabel("$\\theta$ [mrad]")
    fig1.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
    plt.tight_layout()

    plt.show()
