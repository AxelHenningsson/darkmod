import os

import matplotlib.pyplot as plt
import numpy as np

from darkmod.utils import crop

plt.style.use("dark_background")
fontsize = 32  # General font size for all text
ticksize = 32  # tick size
plt.rcParams["font.size"] = fontsize
plt.rcParams["xtick.labelsize"] = ticksize
plt.rcParams["ytick.labelsize"] = ticksize


if __name__ == "__main__":

    # Path to the directory in which reflections are stored
    savedir = "/home/naxhe/workspace/darkmod/tests/end_to_end/defrec/saves"

    # we store the 3 reflections in an array
    data = np.empty((4,), dtype=object)

    for reflection in range(1, 5):
        file = os.path.join(savedir, "reflection_" + str(reflection) + ".npz")

        # Each reflection is associated to multiple data fields.
        data[reflection - 1] = np.load(file)

    hkl = [reflection["hkl"] for reflection in data]
    omega = [reflection["omega_0"] for reflection in data]
    UB_reference = data[0]["U_0"] @ data[0]["B_0"]

    ##### Q 3D TRUE ####
    true_diffraction_vectors = [
        reflection["Q_sample_3D_true"][:, :, 0, :] for reflection in data
    ]
    Q_true_3D = np.transpose(
        np.stack(true_diffraction_vectors, axis=2), axes=(0, 1, 3, 2)
    )
    m, n, _, k = Q_true_3D.shape
    mask = ~np.any(np.isnan(Q_true_3D.reshape(m * n, 3 * k)), axis=1).reshape(m, n)

    fig, ax = plt.subplots(4, 3, figsize=(22, 16), sharex=True, sharey=True)
    fig.suptitle(
        r"True 3D volume (sample coord)",
    )
    for j in range(4):
        rt = [r"#1", r"#2", r"#3", r"#4"][j]
        for i in range(3):
            im = ax[j, i].imshow(crop(Q_true_3D[:, :, i, j], mask))
            cbar = fig.colorbar(im, ax=ax[j, i], fraction=0.046 / 2.0, pad=0.04)
            if j == 0:
                ax[j, i].set_title([r"$Q_x$", r"$Q_y$", r"$Q_z$"][i])
            if j == 3:
                ax[j, i].set_xlabel("y [voxels]")
            if i == 0:
                om = np.round(np.degrees(omega[j]), 1)
                ax[j, i].set_ylabel(
                    "Reflection "
                    + rt
                    + "\n$\omega$="
                    + str(om)
                    + "$^o$"
                    + "\nx [voxels]"
                )

    for a in ax.flatten():
        for spine in a.spines.values():
            spine.set_visible(False)
    plt.tight_layout()

    ##### BP TRUE ####
    bp_true_diffraction_vectors = [
        reflection["bp_Q_sample_3D_true"][:, :, 0, :] for reflection in data
    ]
    Q_bp_true = np.transpose(
        np.stack(bp_true_diffraction_vectors, axis=2), axes=(0, 1, 3, 2)
    )
    m, n, _, k = Q_bp_true.shape
    mask = ~np.any(np.isnan(Q_bp_true.reshape(m * n, 3 * k)), axis=1).reshape(m, n)

    fig, ax = plt.subplots(4, 3, figsize=(22, 16), sharex=True, sharey=True)
    fig.suptitle(
        r"Backpropagated Reflections (sample coord)",
    )
    for j in range(4):
        rt = [r"#1", r"#2", r"#3", r"#4"][j]
        for i in range(3):
            im = ax[j, i].imshow(crop(Q_bp_true[:, :, i, j], mask))
            cbar = fig.colorbar(im, ax=ax[j, i], fraction=0.046 / 2.0, pad=0.04)
            if j == 0:
                ax[j, i].set_title([r"$Q_x$", r"$Q_y$", r"$Q_z$"][i])
            if j == 3:
                ax[j, i].set_xlabel("y [voxels]")
            if i == 0:
                om = np.round(np.degrees(omega[j]), 1)
                ax[j, i].set_ylabel(
                    "Reflection "
                    + rt
                    + "\n$\omega$="
                    + str(om)
                    + "$^o$"
                    + "\nx [voxels]"
                )

    for a in ax.flatten():
        for spine in a.spines.values():
            spine.set_visible(False)
    plt.tight_layout()

    fig, ax = plt.subplots(4, 3, figsize=(22, 16), sharex=True, sharey=True)
    fig.suptitle(
        r"Residual Reflections (sample coord)",
    )
    for j in range(4):
        rt = [r"#1", r"#2", r"#3", r"#4"][j]
        for i in range(3):
            im = ax[j, i].imshow(
                crop(Q_bp_true[:, :, i, j] - Q_true_3D[:, :, i, j], mask),
                cmap="RdBu_r",
                vmin=-1e-4,
                vmax=1e-4,
            )
            #cbar = fig.colorbar(im, ax=ax[j, i], fraction=0.046 / 2.0, pad=0.04)
            if j == 0:
                ax[j, i].set_title([r"$Q_x$", r"$Q_y$", r"$Q_z$"][i])
            if j == 3:
                ax[j, i].set_xlabel("y [voxels]")
            if i == 0:
                om = np.round(np.degrees(omega[j]), 1)
                ax[j, i].set_ylabel(
                    "Reflection "
                    + rt
                    + "\n$\omega$="
                    + str(om)
                    + "$^o$"
                    + "\nx [voxels]"
                )

    for a in ax.flatten():
        for spine in a.spines.values():
            spine.set_visible(False)
    plt.tight_layout()

    Q_true = [reflection["Q_true"] for reflection in data]
    Q_true = np.transpose(np.stack(Q_true, axis=2), axes=(0, 1, 3, 2))
    m, n, _, k = Q_true.shape
    mask = ~np.any(np.isnan(Q_true.reshape(m * n, 3 * k)), axis=1).reshape(m, n)

    fig, ax = plt.subplots(4, 3, figsize=(22, 16), sharex=True, sharey=True)
    fig.suptitle(
        r"Reflections (sample coord)",
    )
    for j in range(4):
        rt = [r"#1", r"#2", r"#3", r"#4"][j]
        for i in range(3):
            im = ax[j, i].imshow(crop(Q_true[:, :, i, j], mask))
            cbar = fig.colorbar(im, ax=ax[j, i], fraction=0.046 / 2.0, pad=0.04)
            if j == 0:
                ax[j, i].set_title([r"$Q_x$", r"$Q_y$", r"$Q_z$"][i])
            if j == 3:
                ax[j, i].set_xlabel("y [voxels]")
            if i == 0:
                om = np.round(np.degrees(omega[j]), 1)
                ax[j, i].set_ylabel(
                    "Reflection "
                    + rt
                    + "\n$\omega$="
                    + str(om)
                    + "$^o$"
                    + "\nx [voxels]"
                )

    for a in ax.flatten():
        for spine in a.spines.values():
            spine.set_visible(False)
    plt.tight_layout()

    plt.show()
