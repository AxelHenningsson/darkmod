import os

import matplotlib.pyplot as plt
import numpy as np

from darkmod.transforms import beta_to_stress, curl, divergence
from darkmod.utils import crop

plt.style.use("dark_background")
fontsize = 24  # General font size for all text
ticksize = 24  # tick size
plt.rcParams["font.size"] = fontsize
plt.rcParams["xtick.labelsize"] = ticksize
plt.rcParams["ytick.labelsize"] = ticksize


def _lsq(Q, G_0, mask):
    """Solve many independent LSQ problems over a 2D field.

    Least squares by pseudoinverse computed though an SVD,
    the procedure is that as y.T = USV i.e y.T* = V.T Si U.T
    the solution is solution = y.T* @ G_0. This allows for
    batch solving over a field, currently not supported in
    np.linalg.lstsq.

    NOTE: This is equivalent to the following loop-style code:
        F = np.zeros((m, n, 3, 3))
        for i in range(m):
            for j in range(n):
                if mask[i, j]:
                    F[i, j] = np.linalg.lstsq(Q[i, j, ...].T, G_0.T, rcond=None)[0]

    Args:
        Q (_type_): _description_
        G_0 (_type_): _description_
        mask (_type_): _description_

    Returns:
        _type_: _description_
    """
    m, n, _, k = Q.shape
    y_T = Q.reshape(m * n, 3, k).transpose(0, 2, 1)[mask.flatten()]
    U, S, Vt = np.linalg.svd(y_T, full_matrices=False)
    _numerical_zeros = S < 1e-8
    S[~_numerical_zeros] = 1.0 / S[~_numerical_zeros]
    S[_numerical_zeros] = 0
    s = Vt.transpose(0, 2, 1) @ (S[..., np.newaxis] * (U.transpose(0, 2, 1) @ G_0.T))
    solution = np.zeros((m, n, 3, 3))
    solution[mask] = s
    if 1:
        F_sample_check = np.zeros((m, n, 3, 3))
        for i in range(m):
            for j in range(n):
                if mask[i, j]:
                    F_sample_check[i, j] = np.linalg.lstsq(
                        Q[i, j, ...].T, G_0.T, rcond=None
                    )[0]
        assert np.allclose(solution, F_sample_check)
    return solution


def deformation(diffraction_vectors, hkl, omega, UB_reference):
    # The measured Q-vectors as a m,n,3,3 array where Q[i,j,:,k] is a diffraction vectord
    Q = np.transpose(np.stack(diffraction_vectors, axis=2), axes=(0, 1, 3, 2))

    m, n, _, k = Q.shape
    mask = ~np.any(np.isnan(Q.reshape(m * n, 3 * k)), axis=1).reshape(m, n)

    if 0:
        fig, ax = plt.subplots(3, 3, figsize=(22, 16), sharex=True, sharey=True)
        fig.suptitle(
            r"Backpropagated Reflections (sample coord)",
        )
        for j in range(3):
            rt = [r"#1", r"#2", r"#3"][j]
            for i in range(3):
                im = ax[j, i].imshow(crop(Q[:, :, i, j], mask))
                cbar = fig.colorbar(im, ax=ax[j, i], fraction=0.046 / 2.0, pad=0.04)
                if j == 0:
                    ax[j, i].set_title([r"$Q_x$", r"$Q_y$", r"$Q_z$"][i])
                if j == 2:
                    ax[j, i].set_xlabel("y [voxels]")
                if i == 0:
                    om = np.round(np.degrees(omega[j]), 1)
                    ax[j, i].set_ylabel(
                        "Reflection "
                        + rt
                        + "\n$\omega$="
                        + str(om)
                        + "$^o$"
                        + "\nz [voxels]"
                    )

        for a in ax.flatten():
            for spine in a.spines.values():
                spine.set_visible(False)
        plt.tight_layout()
        plt.show()
    # least squares reconstruction
    # Reference diffraction vectors: G_0 = U_0 @ B_0 @ Hmatrix
    # The measurement are a stack of Q: y = (U @ B) @ Hmatrix
    # y.T @ F = G_0.T such that
    # F = np.linalg.inv(y @ y.T) @ y @ G_0.T such that
    Hmatrix = np.array(hkl).T
    assert (
        np.linalg.matrix_rank(Hmatrix) == 3
    ), "The hkl reflection set is not fully ranked"
    assert np.linalg.cond(Hmatrix) < 1e4, "illconditioned set of hkl reflections"
    G_0 = UB_reference @ Hmatrix
    F_sample = _lsq(Q, G_0, mask)

    return F_sample


if __name__ == "__main__":
    # Path to the directory in which reflections are stored
    savedir = "/home/naxhe/workspace/darkmod/tests/end_to_end/defrec/saves/simul1"

    # we store the 3 reflections in an array
    data = np.empty((4,), dtype=object)

    for reflection in range(1, 5):
        file = os.path.join(savedir, "reflection_" + str(reflection) + ".npz")

        # Each reflection is associated to multiple data fields.
        data[reflection - 1] = np.load(file)

    # we can access the different fields like so:
    # print(data[2]['strain_mosa'])
    # print(data[0]['Q_rec'])
    # etc...

    # unpack the true diffraction vectors to test our reconstructor
    diffraction_vectors = [reflection["Q_true"] for reflection in data]
    hkl = [reflection["hkl"] for reflection in data]
    omega = [reflection["omega_0"] for reflection in data]
    UB_reference = data[0]["U_0"] @ data[0]["B_0"]

    # this works as expected.
    # diffraction_vectors = [
    #    reflection["Q_sample_3D_true"][:, :, 0, :] for reflection in data
    # ]

    diffraction_vectors = [
        reflection["bp_Q_sample_3D_rec"][:, :, 0, :] for reflection in data
    ]

    # Run the reconstructor to get back the deformation gradient tensor field
    defgrad = deformation(
        diffraction_vectors,
        hkl,
        omega,
        UB_reference,
    )

    mask = ~(np.sum(defgrad, axis=(-1, -2)) == 0)
    beta = defgrad.copy()

    beta_true_3D = data[0]["defgrad"]
    for i in range(3):
        beta[..., i, i] -= 1
        beta_true_3D[..., i, i] -= 1

    _b = np.concatenate((beta_true_3D, beta_true_3D, beta_true_3D), axis=2)
    dx = dy = dz = data[0]["voxel_size"]
    curl_beta_true_3D = curl(_b, (dx, dy, dz))

    elasticity_matrix = np.array(
        [
            [104, 73, 73, 0, 0, 0],
            [73, 104, 73, 0, 0, 0],
            [73, 73, 104, 0, 0, 0],
            [0, 0, 0, 32, 0, 0],
            [0, 0, 0, 0, 32, 0],
            [0, 0, 0, 0, 0, 32],
        ]
    )  # units of GPa

    B_0 = data[0]["B_0"]
    U_0 = data[0]["U_0"]
    C_c = np.linalg.inv(B_0).T
    E = np.column_stack((C_c[:, 0], np.cross(C_c[:, 2], C_c[:, 0]), C_c[:, 2]))
    E /= np.linalg.norm(E, axis=0)

    R = U_0 @ E

    stress = beta_to_stress(beta[..., np.newaxis, :, :], elasticity_matrix, rotation=R)

    _b = np.concatenate((stress, stress, stress), axis=2)
    residual = divergence(_b, (dx, dy, dz))

    plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 3, figsize=(22, 16), sharex=True, sharey=True)
    fig.suptitle("Stress residual ($\\Delta r$)")
    for i in range(3):
        _s = crop(residual[..., 1, i], mask)

        vmax = np.nanmax(np.abs(stress[..., 0, i, :]))
        vmin = -vmax
        im = ax[i].imshow(
            _s,
            vmin=vmin,
            vmax=vmax,
            cmap="seismic",
        )
        ax[i].annotate(
            r"$\boldsymbol{\Delta r}_{" + str(i + 1) + r"}$",
            (15, 25),
            c="black",
        )
        fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
        ax[i].set_xlabel("y [pixels]")
        if i == 0:
            ax[i].set_ylabel("x [pixels]")
    plt.tight_layout()
    for a in ax.flatten():
        for spine in a.spines.values():
            spine.set_visible(False)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(3, 3, figsize=(22, 16), sharex=True, sharey=True)
    fig.suptitle("Reconstructed stress ($\\sigma$)")
    for i in range(3):
        for j in range(3):
            _s = crop(stress[:, :, 0, i, j], mask)

            vmax = np.nanmax(np.abs(stress[:, :, 0, i, j])) * 0.5
            vmin = -vmax
            im = ax[i, j].imshow(
                _s,
                vmin=vmin,
                vmax=vmax,
                cmap="seismic",
            )
            ax[i, j].annotate(
                r"$\boldsymbol{\sigma}_{" + str(i + 1) + str(j + 1) + r"}$",
                (15, 25),
                c="black",
            )
            fig.colorbar(im, ax=ax[i, j], fraction=0.046, pad=0.04)
            if i == 2:
                ax[i, j].set_xlabel("y [pixels]")
            if j == 0:
                ax[i, j].set_ylabel("x [pixels]")
    plt.tight_layout()
    for a in ax.flatten():
        for spine in a.spines.values():
            spine.set_visible(False)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(3, 3, figsize=(22, 16), sharex=True, sharey=True)
    fig.suptitle("Reconstructed elastic distortion ($\\beta$) field around dislocation")
    for i in range(3):
        for j in range(3):
            _s = crop(beta[:, :, i, j], mask)

            vmax = np.nanmax(np.abs(beta_true_3D[:, :, 0, i, j])) * 0.25
            vmin = -vmax
            im = ax[i, j].imshow(
                _s,
                vmin=vmin,
                vmax=vmax,
                cmap="coolwarm",
            )
            ax[i, j].annotate(
                r"$\boldsymbol{\beta}_{" + str(i + 1) + str(j + 1) + r"}$",
                (15, 25),
                c="black",
            )
            fig.colorbar(im, ax=ax[i, j], fraction=0.046, pad=0.04)
            if i == 2:
                ax[i, j].set_xlabel("y [pixels]")
            if j == 0:
                ax[i, j].set_ylabel("x [pixels]")
    plt.tight_layout()
    for a in ax.flatten():
        for spine in a.spines.values():
            spine.set_visible(False)
    plt.style.use("dark_background")

    fig, ax = plt.subplots(3, 3, figsize=(22, 16), sharex=True, sharey=True)
    fig.suptitle(
        "Real space true elastic distortion ($\\beta$) field around dislocation"
    )
    for i in range(3):
        for j in range(3):
            # _s = crop(beta[:, :, i, j], mask)
            _s = beta_true_3D[:, :, 0, i, j]

            vmax = np.nanmax(np.abs(_s)) * 0.25
            vmin = -vmax
            im = ax[i, j].imshow(
                _s,
                vmin=vmin,
                vmax=vmax,
                cmap="coolwarm",
            )
            ax[i, j].annotate(
                r"$\boldsymbol{\beta}_{" + str(i + 1) + str(j + 1) + r"}$",
                (15, 25),
                c="black",
            )
            fig.colorbar(im, ax=ax[i, j], fraction=0.046, pad=0.04)
            if i == 2:
                ax[i, j].set_xlabel("y [voxels]")
            if j == 0:
                ax[i, j].set_ylabel("x [voxels]")
    plt.tight_layout()
    for a in ax.flatten():
        for spine in a.spines.values():
            spine.set_visible(False)

    fig, ax = plt.subplots(3, 3, figsize=(22, 16), sharex=True, sharey=True)

    fig.suptitle(
        "Real space true curl of elastic distortion ($\\nabla x \\beta$) [x 1e6]"
    )
    for i in range(3):
        for j in range(3):
            # _s = crop(beta[:, :, i, j], mask)
            _s = curl_beta_true_3D[:, :, 0, i, j]

            vmax = np.nanmax(np.abs(_s)) * 0.1
            vmin = -vmax
            im = ax[i, j].imshow(
                _s,
                vmin=vmin,
                vmax=vmax,
                cmap="jet",
            )
            ax[i, j].annotate(
                r"$\boldsymbol{\alpha}_{" + str(i + 1) + str(j + 1) + r"}$",
                (15, 25),
                c="black",
            )
            fig.colorbar(im, ax=ax[i, j], fraction=0.046, pad=0.04)
            if i == 2:
                ax[i, j].set_xlabel("y [voxels]")
            if j == 0:
                ax[i, j].set_ylabel("x [voxels]")
    plt.tight_layout()
    for a in ax.flatten():
        for spine in a.spines.values():
            spine.set_visible(False)

    fig, ax = plt.subplots(3, 3, figsize=(22, 16), sharex=True, sharey=True)
    beta = beta[:, :, np.newaxis, :, :]
    _b = np.concatenate((beta, beta, beta), axis=2)
    dx = dy = dz = data[0]["voxel_size"]
    curl_beta_rec_3D = curl(_b, (dx, dy, dz))

    fig.suptitle(
        "Reconstructed curl of elastic distortion ($\\nabla x \\beta$) [x 1e6]"
    )
    for i in range(3):
        for j in range(3):
            # _s = crop(beta[:, :, i, j], mask)
            _s = curl_beta_rec_3D[:, :, 0, i, j]

            vmax = np.nanmax(np.abs(curl_beta_true_3D[:, :, 0, i, j])) * 0.1
            vmin = -vmax
            im = ax[i, j].imshow(
                _s,
                vmin=vmin,
                vmax=vmax,
                cmap="jet",
            )
            ax[i, j].annotate(
                r"$\boldsymbol{\alpha}_{" + str(i + 1) + str(j + 1) + r"}$",
                (15, 25),
                c="black",
            )
            fig.colorbar(im, ax=ax[i, j], fraction=0.046, pad=0.04)
            if i == 2:
                ax[i, j].set_xlabel("y [voxels]")
            if j == 0:
                ax[i, j].set_ylabel("x [voxels]")
    plt.tight_layout()
    for a in ax.flatten():
        for spine in a.spines.values():
            spine.set_visible(False)

    plt.show()
