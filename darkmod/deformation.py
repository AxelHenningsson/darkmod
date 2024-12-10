import numpy as np
from scipy.spatial.transform import Rotation

import darkmod


def _disp_grad(x_d, y_d, bmag, nu):
    """displacement gradient field around a single of edge dislocations"""
    x_d2, y_d2 = x_d * x_d, y_d * y_d
    epsilon = 1e-10  # avoid zero division
    t0 = x_d2 + y_d2 + epsilon
    t1 = 2 * nu * t0
    t2 = 3 * x_d2 + y_d2
    factor = bmag / (4 * np.pi * (1 - nu) * t0**2)
    dux_dx = -y_d * factor * (t2 - t1)
    dux_dy = x_d * factor * (t2 - t1)
    duy_dx = -x_d * factor * (x_d2 + 3 * y_d2 - t1)
    # duy_dy = y_d * factor * (x_d2 - y_d2 - t1)
    duy_dy = y_d * factor * (x_d2 - y_d2 + t1)
    return dux_dx, dux_dy, duy_dx, duy_dy


def straight_edge_dislocation(
    coord,
    x0=[[0, 0, 0]],
    nu=0.334,
    b=2.86 * np.array([1, -1, 0]) * 1e-4,
    n=np.array([1, 1, -1]),
    t=np.array([1, 1, 2]),
    U=np.eye(3),
):
    """Calculate the deformation gradient field around a set of edge dislocations.

    Computes the deformation field caused by edge dislocations
    located at a series of points `x0` in the crystal. Input
    coordinates are in the grain system as specified in the following
    citation:

    citation:
        J. Appl. Cryst. (2021). 54, 1555-1571
        H. F. Poulsen et al.
        Geometrical optics formalism to model contrast in DFXM

    NOTE: this function operates under the fault asssumption of units of microns.

    NOTE: This deformation gradient field originates from Hirthe and Lothe (1992) and
        is derived under the assumption of a straight edge dislocation in an isotropic
        crystal. In reality, the dislocation line may be curved and the crystal is
        always anisotropic.

    Args:
        coord (:obj:`iterable` of :obj:`np.ndarray`): x,y,z coordinate arrays for the crystal points.
        x0 (:obj:`list` of lists): Dislocation line positions in the crystal, defaults to [[0, 0, 0]].
            in grain coordinates. These are the line transaltions from the origin.
        nu (:obj:`float`): Poisson's ratio, defaults to 0.3.
        b (:obj:`np.ndarray`): Burgers vector, shape=(3,), defaults to 2.86e-4*[1, -1, 0].
        n (:obj:`np.ndarray`): Spli plane normal, shape=(3,) defaults to [1, 1, -1].
        t (:obj:`np.ndarray`): Line direction, shape=(3,), defaults to [1, 1, 2].
        U (:obj:`np.ndarray`): Crystal rotation matrix, shape=(3, 3), defaults to np.eye(3).
            Brings crystal frame to sample frame, v_sample = U @ v_crystal.

    Returns:
        :obj:`np.ndarray`: Deformation gradient tensor F, shape=(m, n, 0, 3, 3). x is along rows, y along columns
            and z along axis=2.
    """

    # dislocation system basis matrix
    U_d = np.array([b, n, t]).T
    U_d = U_d / np.linalg.norm(U_d, axis=0)

    bmag = np.linalg.norm(b)

    assert np.allclose(U_d.T @ U_d, np.eye(3)), "b,n and t must form a Cartesian basis."
    assert np.allclose(U_d @ U_d.T, np.eye(3)), "b,n and t must form a Cartesian basis."
    assert np.allclose(np.linalg.det(U_d), 1)
    assert np.allclose(U.T @ U, np.eye(3)), "U must form a Cartesian basis."
    assert np.allclose(U @ U.T, np.eye(3)), "U form a Cartesian basis."
    assert np.allclose(np.linalg.det(U), 1)

    # dislocation system voxel cooridnates.
    X, Y, Z = (U_d.T @ U.T) @ np.array([c.flatten() for c in coord])

    # The deformation gradient tensor in dislocation frame.
    F_d = np.zeros((len(X), 3, 3))

    x0_d = (U_d.T @ U.T) @ np.array(x0).T

    for i in range(x0_d.shape[1]):
        x, y, _ = x0_d[:, i]
        dux_dx, dux_dy, duy_dx, duy_dy = _disp_grad((X - x), (Y - y), bmag, nu)
        F_d[:, 0, 0] += dux_dx
        F_d[:, 0, 1] += dux_dy
        F_d[:, 1, 0] += duy_dx
        F_d[:, 1, 1] += duy_dy

    # Add the identity tensor to the diagonal.
    for i in range(3):
        F_d[:, i, i] += 1

    # Map F back to grain frame.64
    F_g = ((U @ U_d) @ F_d @ (U_d.T @ U.T)).reshape((*coord[0].shape, 3, 3))

    return F_g


def edge_dislocation(X, Y, x0=[[0, 0]], v=0.3, b=2.86 * 1e-4):
    """Calculate the deformation gradient for an edge dislocation.

    Computes the deformation field caused by edge dislocations
    located at a series of points `x0` in the crystal.

    Args:
        X, Y (:obj:`np.ndarray`): 2D coordinate arrays for the crystal points.
        x0 (:obj:`list` of lists): Dislocation positions in the crystal, defaults to [[0, 0]].
        v (:obj:`float`): Poisson's ratio, defaults to 0.3.
        b (:obj:`float`): Burgers vector magnitude, defaults to 2.86e-4.

    Returns:
        :obj:`np.ndarray`: Deformation gradient tensor F, shape=(m, n, 3, 3).
    """
    F = np.zeros((*X.shape, 3, 3))
    a_1 = b / (4 * np.pi * (1 - v))

    for x, y in x0:
        Ys, Xs = (X - x), (Y - y)
        x2, y2 = Ys * Ys, Xs * Xs
        a_2 = x2 + y2
        a_3 = 1 / (a_2 * a_2)
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


def maxwell_stress(coord):
    # A = x**3 * y**2 * z**2
    # B = x**2 * y**2 * z**2
    # C = x**2 * y**2 * z**2

    x, y, z = coord

    dBdx = 2 * y**2 * z**2
    dBdz = 2 * x**2 * y**2

    dAdy = 2 * x**3 * z**2
    dAdz = 2 * x**3 * y**2

    dCdx = 2 * y**2 * z**2
    dCdy = 2 * x**2 * z**2

    dAdyz = 2 * y * 2 * z * x**3
    dBdzx = 2 * x * 2 * z * y * y
    dCdxy = 2 * x * 2 * y * z * z

    sigma = np.zeros((*x.shape, 3, 3))
    sigma[..., 0, 0] = dBdz + dCdy
    sigma[..., 1, 1] = dCdx + dAdz
    sigma[..., 2, 2] = dAdy + dBdx
    sigma[..., 1, 2] = sigma[..., 2, 1] = -dAdyz
    sigma[..., 0, 2] = sigma[..., 2, 0] = -dBdzx
    sigma[..., 0, 1] = sigma[..., 1, 0] = -dCdxy

    return sigma * 1e9


def cantilevered_beam(coord, l, v, E):
    x, y, z = coord

    nu = v

    h = l / 2.0
    t = (6 / 20.0) * l
    Pz = 1000
    Py = 2000
    Iyy = t * h * h * h / 12.0
    Izz = t * h * h * h / 12.0

    eps = np.zeros((*coord[0].shape, 6))
    eps[..., 0] = Py * (l - x) * y / E / Iyy + Pz * (l - x) * z / E / Izz
    eps[..., 1] = (-v * Py * (l - x) * y / E / Iyy) - (v * Pz * (l - x) * z / E / Izz)
    eps[..., 2] = (-v * Py * (l - x) * y / E / Iyy) - (v * Pz * (l - x) * z / E / Izz)
    eps[..., 5] = 2 * (-(1 + v) / E * Py * ((h**2 / 4.0) - y**2) / 2 / Iyy)
    eps[..., 4] = 2 * (-(1 + v) / E * Pz * ((t**2 / 4.0) - z**2) / 2 / Izz)
    eps[..., 3] = 0

    eps = eps.reshape(-1, 6).T

    D = darkmod.transforms.elasticity_matrix(E, v)

    stress_voigt = (D @ eps).T.reshape((*coord[0].shape, 6)).T
    sigma = np.zeros((*x.shape, 3, 3))
    sigma[..., 0, 0] = stress_voigt[0]
    sigma[..., 1, 1] = stress_voigt[1]
    sigma[..., 2, 2] = stress_voigt[2]
    sigma[..., 1, 2] = sigma[..., 2, 1] = stress_voigt[3]
    sigma[..., 0, 2] = sigma[..., 2, 0] = stress_voigt[4]
    sigma[..., 0, 1] = sigma[..., 1, 0] = stress_voigt[5]

    # sigma = np.zeros((*x.shape, 3, 3))
    # sigma[..., 0, 0] = (y + z) ** 2
    # sigma[..., 1, 1] = x + z
    # sigma[..., 2, 2] = np.exp(x + y * 0.2)
    # sigma[..., 1, 2] = sigma[..., 2, 1] = np.sqrt(x + 1e-4) + x * x
    # sigma[..., 0, 2] = sigma[..., 2, 0] = y
    # sigma[..., 0, 1] = sigma[..., 1, 0] = np.log(z) * z

    return sigma


if __name__ == "__main__":
    E, nu = 70, 0.334

    # l = 20 * 1e-3
    s, c = np.sin(np.pi / 3), np.cos(np.pi / 3)
    Rx = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    Rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    U = np.eye(3)  # Rx @ Ry @ Rz

    # xg = np.linspace(1, 1 + 1e-4, 128)
    xg = np.linspace(-1, 1, 128)

    dx = xg[1] - xg[0]
    coord = np.meshgrid(xg, xg, xg, indexing="ij")

    x, y, z = np.eye(3)

    a = 4.0493
    bvec = (a / np.sqrt(2)) * np.array([1, -1, 0]) * 1e-4  # microns
    n = np.array([1, 1, -1])
    t = np.array([1, 1, 2])
    U_d = np.array([bvec, n, t]).T
    U_d = U_d / np.linalg.norm(U_d, axis=0)

    import cProfile
    import pstats
    import time

    pr = cProfile.Profile()
    pr.enable()
    t1 = time.perf_counter()

    defgrad = straight_edge_dislocation(
        coord,
        x0=np.random.rand(100, 3) - 0.5,
        nu=nu,
        b=bvec,
        n=n,
        t=t,
        U=U,
    )

    t2 = time.perf_counter()
    pr.disable()
    pr.dump_stats("tmp_profile_dump")
    ps = pstats.Stats("tmp_profile_dump").strip_dirs().sort_stats("cumtime")
    ps.print_stats(15)
    print("\n\nCPU time is : ", t2 - t1, "s")

    beta = defgrad.copy()
    for i in range(3):
        beta[..., i, i] -= 1

    import matplotlib.pyplot as plt

    from darkmod.transforms import beta_to_stress, curl, divergence, elasticity_matrix

    fontsize = 32  # General font size for all text
    ticksize = 32  # tick size
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["xtick.labelsize"] = ticksize
    plt.rcParams["ytick.labelsize"] = ticksize

    if 0:
        cb = curl(beta)
        print(U_d[:, 2])
        print(U_d[:, 0])
        print(np.outer(U_d[:, 2], U_d[:, 0]))
        plt.style.use("dark_background")
        fig, ax = plt.subplots(3, 3, figsize=(17, 12), sharex=True, sharey=True)
        for i in range(3):
            for j in range(3):
                vmin = np.min(cb[..., cb.shape[2] // 2, i, j]) / 16
                vmax = np.max(cb[..., cb.shape[2] // 2, i, j]) / 16
                im = ax[i, j].imshow(
                    cb[..., cb.shape[2] // 2, i, j], vmin=vmin, vmax=vmax
                )
                fig.colorbar(im, ax=ax[i, j], fraction=0.046, pad=0.04)
        plt.show()
    # This is an isotropic asssumption, the zener ratio should be 1.
    D = elasticity_matrix(E, nu)
    zener_ratio = 2 * D[3, 3] / (D[0, 0] - D[0, 1])
    print("zener_ratio", zener_ratio)
    assert np.isclose(zener_ratio, 1)

    # (due isotropic assumptions the use of U is not required.)
    stress = beta_to_stress(beta, D, rotation=U)

    G = E / (2 * (1 + nu))
    Ds = G * np.linalg.norm(bvec) / (2 * np.pi * (1 - nu))
    x, y, z = coord

    stress_true = np.zeros_like(stress)
    stress_true[..., 0, 0] = -Ds * y * (3 * x**2 + y**2) / (x**2 + y**2) ** 2
    stress_true[..., 1, 1] = Ds * y * (x**2 - y**2) / (x**2 + y**2) ** 2
    stress_true[..., 0, 1] = stress_true[..., 1, 0] = (
        Ds * x * (x**2 - y**2) / (x**2 + y**2) ** 2
    )
    stress_true[..., 2, 2] = nu * (stress_true[..., 0, 0] + stress_true[..., 1, 1])

    ss1 = stress_true[4, 4, 16]
    ss2 = stress[4, 4, 16]
    print(ss1, "\n\n", ss2)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(3, 3, figsize=(18, 16), sharex=True, sharey=True)
    fig.suptitle("Stress in crystal coordinate system [MPa]")
    for i in range(3):
        for j in range(3):
            vmax = np.median(np.abs(stress[:, :, beta.shape[2] // 2, i, j] * 1e3)) * 4
            vmin = -vmax

            im = ax[i, j].imshow(
                stress[:, :, beta.shape[2] // 2, i, j] * 1e3,
                vmax=vmax,
                vmin=vmin,
            )

            fig.colorbar(im, ax=ax[i, j], fraction=0.046, pad=0.04)
            ax[i, j].annotate(
                r"$\boldsymbol{\sigma}_{" + str(i + 1) + str(j + 1) + r"}$",
                (15, 25),
                c="white",
                fontsize=24,
            )
    # plt.style.use("dark_background")
    # fig, ax = plt.subplots(3, 3, figsize=(18, 16), sharex=True, sharey=True)
    # fig.suptitle("Stress in dislocation coordinate system [MPa]")
    # for i in range(3):
    #     for j in range(3):
    #         vmax = (
    #             np.max(np.abs(stress_true[:, :, beta.shape[2] // 2, i, j] * 1e3)) / 10.0
    #         )
    #         vmin = -vmax

    #         im = ax[i, j].imshow(
    #             stress_true[:, :, beta.shape[2] // 2, i, j] * 1e3,
    #             vmax=vmax,
    #             vmin=vmin,
    #         )
    #         fig.colorbar(im, ax=ax[i, j], fraction=0.046, pad=0.04)
    #         ax[i, j].annotate(
    #             r"$\boldsymbol{\sigma}_{" + str(i + 1) + str(j + 1) + r"}$",
    #             (15, 25),
    #             c="white",
    #             fontsize=24,
    #         )

    residual = divergence(stress, (dx, dx, dx))

    plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 3, figsize=(18, 9), sharex=True, sharey=True)
    fig.suptitle("Relative error in stress residual ($\\Delta r$)")
    for i in range(3):
        _s = residual[..., beta.shape[2] // 2, i]
        vmax = np.nanmax(np.abs(stress[..., beta.shape[2] // 2, i, :]))
        vmin = -vmax

        gx = np.gradient(stress[..., 0, i], dx, axis=0)[..., beta.shape[2] // 2]
        gy = np.gradient(stress[..., 1, i], dx, axis=1)[..., beta.shape[2] // 2]
        gz = np.gradient(stress[..., 2, i], dx, axis=2)[..., beta.shape[2] // 2]

        r1 = _s / np.abs(gx)
        r2 = _s / np.abs(gy)
        r3 = _s / np.abs(gz)

        rerr = np.zeros((*r1.shape, 3))
        rerr[..., 0] = r1
        rerr[..., 1] = r2
        rerr[..., 2] = r3

        m = np.abs(rerr) == np.max(np.abs(rerr), axis=-1)[:, :, np.newaxis]
        _s = np.sum(rerr * m, axis=-1)[1:-1, 1:-1]

        im = ax[i].imshow(
            _s,
            cmap="plasma",
            vmax=1e-1,
            vmin=-1e-1,
        )
        ax[i].annotate(
            r"$\boldsymbol{\Delta r}_{" + str(i + 1) + r"}$",
            (15, 25),
            c="white",
            fontsize=24,
        )
        fig.colorbar(im, ax=ax[i], fraction=0.046, pad=0.04)
        ax[i].set_xlabel("y [pixels]")
        if i == 0:
            ax[i].set_ylabel("x [pixels]")
    plt.tight_layout()
    for a in ax.flatten():
        for spine in a.spines.values():
            spine.set_visible(False)
    plt.show()
