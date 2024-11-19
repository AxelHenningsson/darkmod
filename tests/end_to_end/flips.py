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

    # defgrad = linear_gradient(
    #     X.shape,
    #     component=(2, 2),
    #     axis=1,
    #     magnitude=0.0002,
    # )

    defgrad = rotation_gradient(
        X.shape,
        rotation_axis=np.array([1, 1, 0]),
        axis=0,
        magnitude=0.01 * 1e-3,
    )

    defgrad = linear_gradient(
      X.shape,
       component=(2, 2),
       axis=2,
       magnitude=0.001,
    )

    #defgrad = unity_field(X.shape)

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
    pixel_size = crl.magnification * dx * 0.17


    print("pixel_size", pixel_size)

    detector = Detector.orthogonal_mount(
        crl, pixel_size, det_row_count, det_col_count, super_sampling=1
    )

    beam = GaussianLineBeam(z_std=0.1, energy=energy)

    F = crystal.defgrad[0, X.shape[1]//2, X.shape[2]//2]
    eps = (F.T @ F - np.eye(3)) / 2.0
    eps_z = eps[2, 2]
    theta_deformed = np.arcsin(lambda_0 / (2 * d_0 * (1 + eps_z)))
    theta_range = 2 * (crl.theta - theta_deformed)
    print("theta range: ", np.round(np.abs(theta_range * 1e3), 3), "mrad")

    pad = np.abs(crl.theta - theta_deformed) * 1e3

    print(pad)

    nphi = 7
    nchi = 5
    phi_values = np.linspace(-0.13 - 2 * pad, 0.13 + 2 * pad, nphi) * 1e-3
    chi_values = np.linspace(-2.5, 2.5, nchi) * 1e-3

    for angarr in (phi_values, chi_values):
        assert np.median(angarr) == 0

    dphi = phi_values[1] - phi_values[0]
    dchi = chi_values[1] - chi_values[0]

    PHI, CHI = np.meshgrid(phi_values, chi_values, indexing="ij")


    # voxel_volume = np.ones_like(X)

    # crystal.goniometer.phi = 0
    # crystal.goniometer.chi = -0.00125

    # #crystal.goniometer.mu = np.radians( -10. )
    # sample_rotation = crystal.goniometer.R
    # P = detector._projector


    # ray_direction = crl.optical_axis

    # ray_direction = sample_rotation.T @ ray_direction
    # P.detector_corners = sample_rotation.T @ P.detector_corners


    # print(P.detector_corners)
    # print(ray_direction)
    # print(crystal.voxel_size*crl.magnification)


    # image = P(voxel_volume, crystal.voxel_size*crl.magnification, ray_direction)

    # print(P._get_astra_vectors( crystal.voxel_size*crl.magnification, ray_direction))
    # image = np.flipud(np.fliplr(image))


    # fontsize = 16 # General font size for all text
    # ticksize= 16 # tick size
    # plt.rcParams['font.size'] = fontsize
    # plt.rcParams['xtick.labelsize'] = ticksize
    # plt.rcParams['ytick.labelsize'] = ticksize

    # plt.style.use('dark_background')
    # fig, ax = plt.subplots(1, 2, figsize=(13,7))
    # im = ax[0].imshow(image[:, 50:200])
    # fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    # im = ax[1].imshow(np.diff(image,axis=1)[:, 50:200])
    # fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    # plt.tight_layout()
    # plt.show()

    def mosa_scan(
        hkl,
        phi_values,
        chi_values,
        crystal,
        crl,
        detector,
        beam,
        resolution_function,
        signal_to_noise=1000000,
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

        # the noise model
        lam = 2.267
        mu = 99.453
        std = 2.317
        shot_noise  = np.random.poisson(lam=lam, size=mosa.shape)
        thermal_noise = np.random.normal(loc=mu, scale=std, size=mosa.shape)
        noise = thermal_noise + shot_noise

        mosa /= np.max(mosa)
        mosa *= (64000 - 200) # we simulate that we use close to the full range of the camera
        mosa += noise

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

    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    strphi = r'$\Delta \phi$ = '+str(np.round(phi_values[3]*1e3,2))+', '
    strchi = r'$\Delta \chi$ = '+str(np.round(chi_values[2]*1e3,2))
    strtitle = 'Detector image\n '+ strphi + strchi + ' mrad'
    ax.set_title(strtitle)
    im = ax.imshow(mosa[:50,:50,3,2],cmap='plasma')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    print('Subtracting background...')
    background = np.median(mosa[:,0:5,:,:]).astype(np.uint16)
    print( 'background', background)
    mosa.clip(background, out=mosa)
    mosa -= background

    mask = np.sum(mosa, axis=(-1, -2))
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    im = ax.imshow(mask)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    mask = mask > np.max(mask)*0.15
    fig, ax = plt.subplots(1, 1, figsize=(7,7))
    im = ax.imshow(mask)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    # print('phi', phi_values[3])
    # print('chi', chi_values[1])


    # fig, ax = plt.subplots(1, 2, figsize=(14, 8), sharex=True, sharey=True)
    # im = ax[0].imshow(np.diff(mosa[:, 50:200, 3, 1].astype(float),axis=1), cmap="jet")
    # fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    # im = ax[1].imshow(np.diff(mosa[:, 50:200, 3, 3].astype(float),axis=1), cmap="jet")
    # fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    # plt.tight_layout()
    # plt.show()

    # row_index = 76
    # col_index = 107
    # m1 = mosa[row_index, 71, :, :]
    # P, C = np.meshgrid(phi_values, chi_values, indexing="ij")
    # chimean1 = np.sum(mosa[row_index, col_index] * C) / np.sum(mosa[85, 71])
    # m2 = mosa[row_index, 71, :, :]
    # chimean2 = np.sum(mosa[row_index, col_index + 1] * C) / np.sum(mosa[85, 71])

    # im1 = mosa[row_index, col_index, :, :].astype(float)
    # im2 = mosa[row_index, col_index + 1, :, :].astype(float)

    # fig, ax = plt.subplots(1, 3, figsize=(14, 5))
    # fig.suptitle(str(chimean1) + ",  " + str(chimean2))
    # im = ax[0].imshow(im1, cmap="jet")
    # fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    # im = ax[1].imshow(im2, cmap="jet")
    # fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    # im = ax[2].imshow(im2 - im1, cmap="RdBu_r")
    # fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    # plt.tight_layout()
    # plt.show()

    mu, cov = darling.properties.moments(mosa, coordinates=(phi_values, chi_values))

    fig, ax = plt.subplots(1, 2, figsize=(16, 6))
    for i in range(2):
        if i == 0:
            im = ax[i].imshow(mu[:, :, i] * 1e3, cmap="jet", vmin=-0.0006, vmax=0.0006  )
        if i == 1:
            im = ax[i].imshow(mu[:, :, i] * 1e3, cmap="jet", vmin=-0.007, vmax=0.007)

        cbar = fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
        ax[i].set_title(r"Mean " + [r"$\phi$", r"$\chi$"][i] + " [mrad]", fontsize=16)
        ax[i].set_xlabel("y [pixels]", fontsize=16)
        if i == 0:
            ax[i].set_ylabel("z [pixels]", fontsize=16)
        ax[i].tick_params(axis="both", which="major", labelsize=16)
        cbar.ax.tick_params(labelsize=16)
    plt.tight_layout()
    plt.show()
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
                im = ax[j, i].imshow(_Q[:, :, i], cmap="plasma")
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
