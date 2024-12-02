import cProfile
import pstats
import time
import os

import darling
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, find_objects, label
from scipy.spatial.transform import Rotation

from darkmod import laue, scan
from darkmod.beam import GaussianLineBeam, HeavysideBeam
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


def main(
    savedir,
    reflection,
    ntheta,
    nphi,
    nchi,
    spatial_artefact=False,
    detector_noise=False,
    plot=False,
):
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

    # NOTE: The hkl we are probing is NOT alignd with sample-z at this point.
    # this makes thinking about the projection of rotation gradients a bit harder.

    if reflection == 1:  # Reflection #1
        hkl = np.array([-1, -1, 3])
        crystal.goniometer.omega = np.radians(6.431585)
        thmax = 0.75
        phimax = 2.0
        chimax = 3.0
    elif reflection == 2:  # Reflection #2
        hkl = np.array([-1.0, 1.0, 3.0])
        crystal.goniometer.omega = np.radians(96.431585)
        thmax = 0.7
        phimax = 3.0
        chimax = 2.0
    elif reflection == 3:  # Reflection #3
        hkl = np.array([1.0, 1.0, 3.0])
        crystal.goniometer.omega = np.radians(186.431585)
        thmax = 0.75
        phimax = 1.2
        chimax = 2.5
    elif reflection == 4:
        hkl = np.array([-1.0, 1.0, 3.0])
        crystal.goniometer.omega = np.radians(96.431585)
        thmax = 0.7
        phimax = 3.0
        chimax = 2.0

    eta = np.radians(20.232593)
    theta = np.radians(15.416837)
    # Bring the CRL to diffracted beam.
    crl.goto(theta, eta)

    # Discretize the crystal
    xg = np.linspace(-1, 1, 33 * 4 + 1)  # microns
    yg = np.linspace(-1, 1, 33 * 4 + 1)  # microns
    # zg = np.linspace(-1, 1, 67)  # microns
    dx = 4 * (xg[1] - xg[0])
    zg = np.array([0])
    X, Y, Z = np.meshgrid(xg, yg, zg, indexing="ij")

    defgrad = straight_edge_dislocation((X, Y, Z), x0=[[0, 0, 0]])

    if ntheta == 1:
        theta_values = np.array([0.0])
    else:
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
    # NA in the horixontal plane to decrease which would explain the rolling curves
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

    detector = Detector.orthogonal_mount(
        crl, pixel_size, det_row_count, det_col_count, super_sampling=1
    )

    # beam = GaussianLineBeam(z_std=0.1, energy=energy)
    beam = HeavysideBeam(y_width=1e8, z_width=1e8, energy=energy)
    if len(theta_values) == 1:
        dth = 0
    else:
        dth = theta_values[1] - theta_values[0]
    dphi = phi_values[1] - phi_values[0]
    dchi = chi_values[1] - chi_values[0]

    print("pixel_size", pixel_size)
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

    # We expect to be able to approximate the mean Q vector along the ray paths.
    def expected_Q(crystal, beam, detector):
        Q_sample_vol = crystal.get_Q_sample(hkl)
        x_lab = crystal.goniometer.R @ crystal._x
        sample_beam_density = beam(x_lab).reshape(crystal._grid_scalar_shape)
        Qlw = Q_sample_vol * sample_beam_density[..., np.newaxis]
        Q_true = np.zeros((mu.shape[0], mu.shape[1], 3))
        for i in range(3):
            Q_true[:, :, i] = detector.render(
                Qlw[:, :, :, i],
                crystal.voxel_size,
                crl.optical_axis,
                crl.magnification,
                crystal.goniometer.R,
            )
        w = detector.render(
            sample_beam_density,
            crystal.voxel_size,
            crl.optical_axis,
            crl.magnification,
            crystal.goniometer.R,
        )
        Q_true = np.divide(
            Q_true,
            w[:, :, np.newaxis],
            out=np.full_like(Q_true, np.nan),
            where=w[:, :, np.newaxis] != 0,
        )
        return Q_true

    Q_true = expected_Q(crystal, beam, detector)

    # sample space recon
    Q_rec = np.zeros((mu.shape[0], mu.shape[1], 3))
    R_omega = crystal.goniometer.get_R_omega(crystal.goniometer.omega)
    rotation_eta = Rotation.from_rotvec(np.array([1, 0, 0]) * (crl.eta))
    for i in range(mu.shape[0]):
        for j in range(mu.shape[1]):
            R_s = crystal.goniometer.get_R_top(mu[i, j, 1], mu[i, j, 2])
            d_rec = lambda_0 / (2 * np.sin(crl.theta + mu[i, j, 0]))
            Q_norm = 2 * np.pi / d_rec
            rotation_th = Rotation.from_rotvec(
                np.array([0, 1, 0]) * (-2 * (crl.theta + mu[i, j, 0]))
            )
            rot = (rotation_eta * rotation_th).as_matrix()
            _Q = rot @ np.array([1, 0, 0]) - np.array([1, 0, 0])
            _Q = _Q / np.linalg.norm(_Q)
            Q_sample_0 = R_omega.T @ _Q * Q_norm
            Q_rec[i, j] = R_s.T @ Q_sample_0

    d_field_rec = (2 * np.pi) / np.linalg.norm(Q_rec, axis=-1)
    d_field_true = (2 * np.pi) / np.linalg.norm(Q_true, axis=-1)
    hkl_strain_rec = (d_field_rec - d_0) / d_0
    hkl_strain_true = (d_field_true - d_0) / d_0

    bp_Q_sample_3D_true = np.zeros((*X.shape, 3))
    for i in range(3):
        bp_Q_sample_3D_true[..., i] = detector.backpropagate(
            Q_true[..., i],
            X.shape,
            crystal.voxel_size,
            crl.optical_axis,
            crl.magnification,
            crystal.goniometer.R,
        )

    bp_Q_sample_3D_rec = np.zeros((*X.shape, 3))
    for i in range(3):
        bp_Q_sample_3D_rec[..., i] = detector.backpropagate(
            Q_rec[..., i],
            X.shape,
            crystal.voxel_size,
            crl.optical_axis,
            crl.magnification,
            crystal.goniometer.R,
        )

    if savedir is not None:
        np.savez(
            os.path.join(savedir, "reflection_" + str(reflection)),
            Q_rec=Q_rec,
            Q_true=Q_true,
            strain_mosa=strain_mosa,
            hkl=hkl,
            U_0=crystal.U,
            B_0=crystal.B,
            eta_0=crl.eta,
            theta_0=crl.theta,
            mu_0=crystal.goniometer.mu,
            omega_0=crystal.goniometer.omega,
            phi_0=crystal.goniometer.phi,
            chi_0=crystal.goniometer.chi,
            delta_theta=theta_values,
            phi=phi_values,
            chi=chi_values,
            X=X,
            Y=Y,
            Z=Z,
            defgrad=defgrad,
            Q_sample_3D_true=crystal.get_Q_sample(hkl),
            optical_axis=crl.optical_axis,
            magnification=crl.magnification,
            detector_corners=detector.detector_corners,
            det_col_count=detector.det_col_count,
            det_row_count=detector.det_row_count,
            pixel_size=detector.pixel_size,
            voxel_size=crystal.voxel_size,
            bp_Q_sample_3D_true=bp_Q_sample_3D_true,
            bp_Q_sample_3D_rec=bp_Q_sample_3D_rec,
        )

    if plot:
        fig, ax = plt.subplots(1, 3, figsize=(16, 6), sharex=True, sharey=True)
        _mu = crop(mu, mask)
        for i in range(3):
            # im = ax[i].imshow(_mu[:, :, i] * 1e3, cmap="jet")

            if i == 0:
                im = ax[i].imshow(
                    _mu[:, :, i] * 1e3, cmap="jet", vmin=-0.006, vmax=0.006
                )
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

        fig, ax = plt.subplots(2, 3, figsize=(16, 12), sharex=True, sharey=True)
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
                if i == 0:
                    ax[j, i].set_ylabel("z [pixels]")
                ax[j, i].tick_params(axis="both", which="major", labelsize=16)
                cbar.ax.tick_params(labelsize=16)
        plt.tight_layout()

        fig, ax = plt.subplots(1, 2, figsize=(14, 7), sharex=True, sharey=True)
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


if __name__ == "__main__":
    for reflection in range(1, 5):
        print("\n\n######## REFLECTION " + str(reflection) + " ##########")
        main(
            savedir="/home/naxhe/workspace/darkmod/tests/end_to_end/defrec/saves/simul_mosa",
            reflection=reflection,
            ntheta=1,
            nphi=81,
            nchi=81,
            spatial_artefact=False,
            detector_noise=False,
            plot=False,
        )
