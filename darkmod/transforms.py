import matplotlib.pyplot as plt
import numpy as np


# class HighPrecisionRotation(object):
#     """Scipy wrapper to have R @ vector work out of the box without having to call
#     the .apply() method, and to allow for the notation of R.T = R.inv()

#     Args:
#         object (scipy.spatial.transform.Rotation): scipy_rotation
#     """

#     def __init__(self, scipy_rotation):
#         self.scipy_rotation = scipy_rotation

#     def __matmul__(self, vectors):
#         return self.scipy_rotation.apply(vectors.T).T

#     def __mul__(self, other):
#         return HighPrecisionRotation(self.scipy_rotation * other.scipy_rotation)

#     def __rmul__(self, other):
#         return HighPrecisionRotation(other.scipy_rotation * self.scipy_rotation)

#     def as_matrix(self):
#         return self.scipy_rotation.as_matrix()

#     @property
#     def T(self):
#         return HighPrecisionRotation(self.scipy_rotation.inv())


def _lab_to_Q_rot_mat(Q_lab):
    q_ll = Q_lab / np.linalg.norm(Q_lab)
    q_roll = np.array([0, 1, 0])
    q_rock = np.cross(q_roll, q_ll)
    q_rock = q_rock / np.linalg.norm(q_rock)
    return np.array((q_rock, q_roll, q_ll)).T


def lab_to_Q(lab_xyz, Q_lab):
    """Tranform from lab x,y,z coordinates to Q-system (q_rock, q_roll, q_ll)

    Args:
        lab_xyz (:obj:`numpy array`): Lab frame cartesian points. shape=(3,N)
        Q_lab (:obj:`numpy array`): The diffraction vector that will define the
            z-direction in the Q-system. shape=(3,)

    Returns:
        :obj:`numpy array`: Q-cooridnates of the xyz points as (q_rock, q_roll, q_ll)
    """
    tranformation_matrix = _lab_to_Q_rot_mat(Q_lab)
    return tranformation_matrix.T @ lab_xyz


def Q_to_lab(q_system_xyz, Q_lab):
    """Tranform from Q-system to lab x,y,z coordinates

    Args:
        q_system_xyz (:obj:`numpy array`): Q-system points. shape=(3,N)
        Q_lab (:obj:`numpy array`): The diffraction vector that will define the
            z-direction in the Q-system. shape=(3,)

    Returns:
        :obj:`numpy array`: Q-cooridnates of the xyz points as (q_rock, q_roll, q_ll)
    """
    tranformation_matrix = _lab_to_Q_rot_mat(Q_lab)
    return tranformation_matrix @ q_system_xyz


def curl(tensor_field, dx=(1, 1, 1)):
    """
    Compute the curl of a 3D voxelated field of first or second order tensors.

    NOTE: uses second order finite difference scheme to approximate derivatives
    according to numpy.gradient. It is assumed that the spatial coordinates
    increase along the array axis and, that the x,y,z axes are ordered as
    axis=0=x, axis=1=y, and axis=2=z.

    Args:
        tensor_field (:obj:`numpy array`): first or second order tensor field
            of shape=(m,n,0,3) or shape=(m,n,0,3,3).
        dx(:obj:`tuple` of :obj:`float`): dx,dy,dz spacings between voxels in the field.
            Defaults to (1,1,1).

    Returns:
        :obj:`numpy array`: The curl of the input field of same shape as tensor_field.
    """

    e = np.array(  # Levi-Civita symbol
        [
            [[0, 0, 0], [0, 0, 1], [0, -1, 0]],
            [[0, 0, -1], [0, 0, 0], [1, 0, 0]],
            [[0, 1, 0], [-1, 0, 0], [0, 0, 0]],
        ]
    )

    order = len(tensor_field.shape) - 3
    if order == 1:
        l_range = 1
    elif order == 2:
        l_range = 3

    out = np.zeros_like(tensor_field)
    for i in range(3):
        for l in range(l_range):  # to handle both 1rst and 2nd order tensors.
            for j in range(3):
                for k in range(3):
                    if e[i, j, k] != 0:
                        if len(tensor_field.shape) == 4:
                            w_k_j = np.gradient(tensor_field[..., k], dx[j], axis=j)
                            print(e[i, j, k] * w_k_j[32, 32, 32], e[i, j, k], i, j, k)
                            out[..., i] += e[i, j, k] * w_k_j
                        elif len(tensor_field.shape) == 5:
                            T_kl_j = np.gradient(tensor_field[..., k, l], dx[j], axis=j)
                            out[..., i, l] += e[i, j, k] * T_kl_j
    return out


def elasticity_matrix(E, nu):
    """Create a 6x6 elasticity matrix for isotropic material.

    Args:
        E (:obj:`float`): Young's modulus.
        nu (:obj:`float`): Poisson's ratio.

    Returns:
        :obj:`numpy array`: The 6x6 elasticity matrix.
    """
    return (E / ((1 + nu) * (1 - 2 * nu))) * np.array(
        [
            [1 - nu, nu, nu, 0, 0, 0],
            [nu, 1 - nu, nu, 0, 0, 0],
            [nu, nu, 1 - nu, 0, 0, 0],
            [0, 0, 0, (1 - 2 * nu) / 2, 0, 0],
            [0, 0, 0, 0, (1 - 2 * nu) / 2, 0],
            [0, 0, 0, 0, 0, (1 - 2 * nu) / 2],
        ]
    )


def divergence(tensor_field, dx=(1, 1, 1)):
    """
    Compute the divergence of a 3D voxelated field of first or second order tensors.

    NOTE: uses second order finite difference scheme to approximate derivatives
    according to numpy.gradient. It is assumed that the spatial coordinates
    increase along the array axis and, that the x, y, z axes are ordered as
    axis=0=x, axis=1=y, and axis=2=z.

    Args:
        tensor_field (:obj:`numpy array`): first or second order tensor field
            of shape=(m, n, o, 3) or shape=(m, n, o, 3, 3).
        dx(:obj:`tuple` of :obj:`float`): dx, dy, dz spacings between voxels in
            the field. Defaults to (1, 1, 1).

    Returns:
        :obj:`numpy array`: The divergence of the input field of same shape as
            tensor_field.
    """
    order = len(tensor_field.shape) - 3
    out = np.zeros_like(tensor_field[..., 0])

    for i in range(3):
        if order == 1:
            out += np.gradient(tensor_field[..., i], dx[i], axis=i)
        elif order == 2:
            for j in range(3):
                out[..., j] += np.gradient(tensor_field[..., i, j], dx[i], axis=i)

    return out


def defgrad_to_stress(deformation_gradient, elasticity_matrix):
    """Convert deformation gradient to cauchy stress over a voxel field.

    The linear elastic Hook law is used such that at each point:
        beta = F - I
        epsilon = 0.5*(beta + beta.T),
        sigma = elasticity_matrix  @ epsilon
    where sigma is the cauchy stress tensor, epsilon is the small strain tensor,
    and F is the deformation gradient tensor.

    NOTE: it is assumed that the input elasticity matrix is arranged such that
        voigt notation is used for the stress and strain tensors. I.e
            elasticity_matrix @ epsilon_voigt = sigma_voigt
        where the voigt notation is defined as:
            epsilon_voigt = [e11, e22, e33, 2*e23, 2*e13, 2*e12]


    Args:
        deformation_gradient (:obj:`numpy array`): The deformation gradient tensor
            of shape=(m,n,o,3,3).
        elasticity_matrix (:obj:`numpy array`): The elasticty matrix of shape=(6,6).

    Returns:
        :obj:`numpy array`: The cauchy stress tensor of shape=(m,n,o,3,3).

    """
    beta = deformation_gradient.copy()
    for i in range(3):
        beta[..., i, i] -= 1
    sigma = beta_to_stress(beta, elasticity_matrix)
    return sigma


def beta_to_stress(beta, elasticity_matrix, rotation=np.eye(3)):
    """Convert small linear elastic distortion to cauchy stress over a voxel field.

    The linear elastic Hook law is used such that at each point:
        epsilon = 0.5*(beta + beta.T),
        sigma = elasticity_matrix  @ epsilon
    where sigma is the cauchy stress tensor and epsilon is the small strain tensor.

    NOTE: it is assumed that the input elasticity matrix is arranged such that
        voigt notation is used for the stress and strain tensors. I.e
            elasticity_matrix @ epsilon_voigt = sigma_voigt
        where the voigt notation is defined as:
            epsilon_voigt = [e11, e22, e33, 2*e23, 2*e13, 2*e12]

    Args:
        beta (:obj:`numpy array`): The small elastic distortion of shape=(m,n,o,3,3).
        elasticity_matrix (:obj:`numpy array`): The elasticty matrix of shape=(6,6).
        rotation (:obj:`numpy array`): The rotation matrix to change the basis of the
            beta tensor. Defaults to np.eye(3).

    Returns:
        :obj:`numpy array`: The cauchy stress tensor of shape=(m,n,o,3,3).

    """
    beta = change_basis(beta, rotation)

    epsilon = 0.5 * (beta + beta.transpose(0, 1, 2, 4, 3))

    eps_flat = epsilon.reshape(-1, 9).T
    eps_flat_voigt = eps_flat[[0, 4, 8, 5, 2, 1], :]
    eps_flat_voigt[3:, :] *= 2

    stress_flat_voigt = elasticity_matrix @ eps_flat_voigt
    stress_voigt = stress_flat_voigt.reshape(6, *beta.shape[:3])

    sigma = np.zeros((*beta.shape[:3], 3, 3))
    sigma[..., 0, 0] = stress_voigt[0]
    sigma[..., 1, 1] = stress_voigt[1]
    sigma[..., 2, 2] = stress_voigt[2]
    sigma[..., 1, 2] = sigma[..., 2, 1] = stress_voigt[3]
    sigma[..., 0, 2] = sigma[..., 2, 0] = stress_voigt[4]
    sigma[..., 0, 1] = sigma[..., 1, 0] = stress_voigt[5]

    sigma = change_basis(sigma, rotation.T)

    return sigma


def change_basis(field, rotation):
    """Change the basis of a 3D voxelated field of vectors or tensors.

    The corresponding rotation matrix is applied to each vector or tensor in the field
    as
        rotation.T @ field[i,j,k] @ rotation (if field.shape=(m,n,o,3,3))
            or
        rotation.T @ field[i,j,k] (if field.shape=(m,n,o,3,))
    I.e if rotatoin is the bais of system a then the field is brought into the basis
    of a. I.e we put field in the rotation basis.

    Args:
        field (:obj:`numpy array`): 3D voxelated field of vectors or tensors of
            shape=(m,n,o,3) or shape=(m,n,o,3,3).
        rotation (:obj:`numpy array`): 3x3 rotation matrix.

    Returns:
        :obj:`numpy array`: The field in the new basis of same shape as field.
    """
    if len(field.shape) == 5:
        return (rotation.T @ field.reshape(-1, 3, 3) @ rotation).reshape(field.shape)
    elif len(field.shape) == 4:
        return (rotation.T @ field.reshape(-1, 3)).reshape(field.shape)


if __name__ == "__main__":
    field = np.random.rand(10, 10, 10, 3, 3)
    from scipy.spatial.transform import Rotation

    R = Rotation.random().as_matrix()

    fc = change_basis(field, R)

    beta_to_stress(field, np.eye(6))
    print(fc[5, 5, 5] - R @ field[5, 5, 5] @ R.T)
    raise

    fontsize = 18  # General font size for all text
    ticksize = 18  # tick size
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["xtick.labelsize"] = ticksize
    plt.rcParams["ytick.labelsize"] = ticksize
    plt.style.use("dark_background")

    n = 64
    x = np.linspace(-1, 1, n)
    y = np.linspace(-1, 1, n)
    z = np.linspace(-1, 1, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    U = -Y
    V = X
    W = np.zeros_like(X)

    F = np.zeros((*X.shape, 3))
    F[..., 0] = -Y
    F[..., 1] = X
    F[..., 2] = 0

    plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    im = ax.imshow(div_F[:, :, n // 2])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
    raise
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]

    c = curl(F, (dx, dy, dz))

    plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    im = ax.imshow(c[:, :, n // 2, 2])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    X, Y, U, V = (
        X[:, :, len(z) // 2],
        Y[:, :, len(z) // 2],
        U[:, :, len(z) // 2],
        V[:, :, len(z) // 2],
    )
    magnitude = np.sqrt(U**2 + V**2)

    # Create a figure
    fig, ax = plt.subplots(figsize=(12, 9))

    # Plot the magnitude as an image (heatmap)
    cax = ax.imshow(magnitude, extent=[-1, 1, -1, 1], origin="lower", cmap="viridis")

    # Overlay the quiver plot (arrows in black)
    ax.quiver(X, Y, U, V, color="white", scale=50, alpha=1)

    # Add colorbar for the heatmap
    fig.colorbar(cax, ax=ax, label="Magnitude", fraction=0.046, pad=0.04)

    # Formatting the plot
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("Curl of Vector Field, f(x,y,z)=[-y,x,0]")

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    # Show the plot
    plt.show()
