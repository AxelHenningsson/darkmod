import numpy as np
from scipy.spatial.transform import Rotation


def straight_edge_dislocation(coord, x0=[[0, 0, 0]], v=0.334, b=2.86 * 1e-4):
    """Calculate the deformation gradient for an edge dislocation.

    Computes the deformation field caused by edge dislocations
    located at a series of points `x0` in the crystal.

    Args:
        X, Y (:obj:`np.ndarray`): 2D coordinate arrays for the crystal points.
        x0 (:obj:`list` of lists): Dislocation positions in the crystal, default is [[0, 0, 0]].
        v (:obj:`float`): Poisson's ratio, default is 0.3.
        b (:obj:`float`): Burgers vector magnitude, default is 2.86e-4.

    Returns:
        :obj:`np.ndarray`: Deformation gradient tensor F, shape=(m, n, 3, 3).
    """

    # dislocation system basis matrix
    U_d = np.array([[1, -1, 0], [1, 1, -1], [1, 1, 2]]).T
    U_d = U_d / np.linalg.norm(U_d, axis=0)

    # dislocation system voxel cooridnates.
    X, Y, Z = U_d.T @ np.array([c.flatten() for c in coord])

    # The deformation gradient tensor in dislocation frame.
    F_d = np.zeros((len(X), 3, 3))
    for i in range(3):
        F_d[:, i, i] = 1

    for i in range(len(x0)):

        # dislocation position in dislocation frame.
        x, y, _ = U_d.T @ np.array(x0[i])

        # compute local F value
        Ys, Xs = (X - x), (Y - y)
        x2, y2 = Ys * Ys, Xs * Xs
        a_1 = b / (4 * np.pi * (1 - v))

        a_2 = x2 + y2
        a_3 = 1 / ((a_2 * a_2) + 1e-8)
        a_4 = -2 * v * a_2
        F_d[:, 0, 0] += (-Ys * (3 * x2 + y2 + a_4)) * a_3 * a_1
        F_d[:, 0, 1] += (Xs * (3 * x2 + y2 + a_4)) * a_3 * a_1
        F_d[:, 1, 0] += (-Xs * (x2 + 3 * y2 + a_4)) * a_3 * a_1
        F_d[:, 1, 1] += (Ys * (x2 - y2 + a_4)) * a_3 * a_1

    # Map F back to sample frame.
    F_g = (U_d @ F_d @ U_d.T).reshape((*coord[0].shape, 3, 3))

    return F_g


def edge_dislocation(X, Y, x0=[[0, 0]], v=0.3, b=2.86 * 1e-4):
    """Calculate the deformation gradient for an edge dislocation.

    Computes the deformation field caused by edge dislocations
    located at a series of points `x0` in the crystal.

    Args:
        X, Y (:obj:`np.ndarray`): 2D coordinate arrays for the crystal points.
        x0 (:obj:`list` of lists): Dislocation positions in the crystal, default is [[0, 0]].
        v (:obj:`float`): Poisson's ratio, default is 0.3.
        b (:obj:`float`): Burgers vector magnitude, default is 2.86e-4.

    Returns:
        :obj:`np.ndarray`: Deformation gradient tensor F, shape=(m, n, 3, 3).
    """
    F = np.zeros((*X.shape, 3, 3))
    a_1 = b / (4 * np.pi * (1 - v))

    for x, y in x0:

        Ys, Xs = (X - x), (Y - y)
        x2, y2 = Ys * Ys, Xs * Xs
        a_2 = x2 + y2
        a_3 = 1 / ((a_2 * a_2))
        a_4 = -2 * v * a_2

        F[:, :, 0, 0] += (-Ys * (3 * x2 + y2 + a_4)) * a_3
        F[:, :, 0, 1] += (Xs * (3 * x2 + y2 + a_4)) * a_3
        F[:, :, 1, 0] += (-Xs * (x2 + 3 * y2 + a_4)) * a_3
        F[:, :, 1, 1] += (Ys * (x2 - y2 + a_4)) * a_3

    F *= a_1

    for i in range(3):
        F[:, :, i, i] += 1

    return F


def linear_gradient(shape, component=(2, 2), axis=1, magnitude=0.003):
    """Linear gradient in x,y or z-component moving across x,y or z.

    Args:
        shape (:obj:`tuple` of int): The 3D spatial array shape (m,n,o) of the field.
        component (:obj:`tuple` of int, optional): The component of the  deformation
            will vary across the field. Defaults to (2,2), i.e the zz-component.
        axis (:obj:`int`, optional): The axis (x,y,z) that the linear gradient varies across.
            Defaults to 2, i.e the y direction.
        magnitude (:obj:`float`, optional): Value of the graident, the gradient will range
             from -magnitude to +magnitude. Defaults to 0.003.

    Returns:
        :obj:`np.ndarray: Deformation graident tensor field of shape=(m,n,o,3,3).
    """
    F = unity_field(shape)
    k, l = component
    deformation_range = np.linspace(-magnitude, magnitude, shape[1])
    for i in range(len(deformation_range)):
        if axis == 0:
            F[i, :, :, k, l] += deformation_range[i]
        elif axis == 1:
            F[:, i, :, k, l] += deformation_range[i]
        elif axis == 2:
            F[:, :, i, k, l] += deformation_range[i]
    return F


def rotation_gradient(shape, rotation_axis, axis=1, magnitude=0.003):
    """Linear gradient in x,y or z-component moving across x,y or z.

    Args:
        shape (:obj:`tuple` of int): The 3D spatial array shape (m,n,o) of the field.
        rot_ax (:obj:`numpy array`): The axis of rotation.
        axis (:obj:`int`, optional): The axis (x,y,z) that the linear gradient varies across.
            Defaults to 2, i.e the y direction.
        magnitude (:obj:`float`, optional): Value of the graident in radians.

    Returns:
        :obj:`np.ndarray: Deformation graident tensor field of shape=(m,n,o,3,3).
    """
    r = rotation_axis / np.linalg.norm(rotation_axis)
    F = unity_field(shape)
    deformation_range = np.linspace(-magnitude, magnitude, shape[1])
    for i in range(len(deformation_range)):
        rotmat = Rotation.from_rotvec(r * deformation_range[i]).as_matrix()
        if axis == 0:
            F[i, :, :, :, :] = rotmat
        elif axis == 1:
            F[:, i, :, :, :] = rotmat
        elif axis == 2:
            F[:, :, i, :, :] = rotmat
    return F


def multi_gradient(
    shape,
    component,
    rotation_axis,
    axis=1,
    rot_magnitude=0.003,
    strain_magnitude=0.003,
):
    """Linear rotation gradient in x,y or z-component moving across x,y or z of.
    and at the same time a linear strain gradient in x,y or z-component moving
    across the same selection of x,y or z.

    Args:
        shape (:obj:`tuple` of int): The 3D spatial array shape (m,n,o) of the field.
        rot_ax (:obj:`numpy array`): The axis of rotation.
        axis (:obj:`int`, optional): The axis (x,y,z) that the linear gradient varies across.
            Defaults to 2, i.e the y direction.
        rot_magnitude (:obj:`float`, optional): Value of the graident in radians.
        strain_magnitude (:obj:`float`, optional): Value of the strain graident.


    Returns:
        :obj:`np.ndarray: Deformation graident tensor field of shape=(m,n,o,3,3).
    """
    r = rotation_axis / np.linalg.norm(rotation_axis)
    F = unity_field(shape)
    rot_deformation_range = np.linspace(-rot_magnitude, rot_magnitude, shape[1])
    strain_deformation_range = np.linspace(
        -strain_magnitude, strain_magnitude, shape[1]
    )
    k, l = component
    for i in range(shape[1]):
        rotmat = Rotation.from_rotvec(r * rot_deformation_range[i]).as_matrix()
        stretch = np.eye(3)
        stretch[k, l] += strain_deformation_range[i]
        if axis == 0:
            F[i, :, :, :, :] = rotmat @ stretch
        elif axis == 1:
            F[:, i, :, :, :] = rotmat @ stretch
        elif axis == 2:
            F[:, :, i, :, :] = rotmat @ stretch
    return F


def unity_field(shape):
    """A field of unity deformation (i.e no deformation)

    Args:
        shape (:obj:`tuple` of int): The 3D spatial array shape (m,n,o) of the field.

    Returns:
        :obj:`np.ndarray: Deformation graident tensor field of shape=(m,n,o,3,3).
    """
    F = np.zeros((*shape, 3, 3))
    for i in range(3):
        F[:, :, :, i, i] = 1
    return F


def simple_shear(shape, shear_magnitude=0.003):
    """A field of uniform simple shear deformation.

    The F[0, 1] component is set to the fixed value of shear_magnitude.

    Args:
        shape (:obj:`tuple` of int): The 3D spatial array shape (m,n,o) of the field.
        shear_magnitude (:obj:`float`, optional): Value of the shear. Defaults to 0.003.

    Returns:
        :obj:`np.ndarray: Deformation graident tensor field of shape=(m,n,o,3,3).
    """
    F = unity_field(shape)
    F[:, :, :, 0, 1] = shear_magnitude
    return F


if __name__ == "__main__":

    xg = np.linspace(-1, 1, 128)
    coord = np.meshgrid(xg, xg, xg, indexing="ij")
    defgrad = straight_edge_dislocation(coord, x0=[[0,0,0]])
    beta = defgrad.copy()
    for i in range(3):
        beta[..., i, i] -= 1

    import matplotlib.pyplot as plt

    fontsize = 16  # General font size for all text
    ticksize = 16  # tick size
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["xtick.labelsize"] = ticksize
    plt.rcParams["ytick.labelsize"] = ticksize

    # from matplotlib.colors import LinearSegmentedColormap

    # color_min    = "#4203ff"
    # color_center = "black"
    # color_max    = "#ff0342"
    # cmap = LinearSegmentedColormap.from_list(
    #     "cmap_name",
    #     [color_min, color_center, color_max]
    # )

    plt.style.use("dark_background")
    fig, ax = plt.subplots(3, 3, figsize=(15, 9), sharex=True, sharey=True)
    fig.suptitle("Elastic distortion ($\\beta$) field around dislocations")
    for i in range(3):
        for j in range(3):
            vmax = np.max(np.abs(beta[:, :, coord[0].shape[2] // 2, i, j]))*0.1
            vmin = -vmax
            im = ax[i, j].imshow(
                beta[:, :, coord[0].shape[2] // 2, i, j],
                vmin=vmin,
                vmax=vmax,
                cmap="coolwarm",
            )
            ax[i, j].annotate(r'$\boldsymbol{F}_{'+str(i+1)+str(j+1)+r'}$', (6,14), c='black')
            fig.colorbar(im, ax=ax[i, j], fraction=0.046, pad=0.04)
            if i == 2 :
                ax[i, j].set_xlabel("y [voxels]")
            if j == 0 :
                ax[i, j].set_ylabel("x [voxels]")
    plt.tight_layout()
    plt.show()
