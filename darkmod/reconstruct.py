import numpy as np
from scipy.spatial.transform import Rotation


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


def deformation(diffraction_vectors, hkl, UB_reference):
    # The measured Q-vectors as a m,n,3,3 array where Q[i,j,:,k] is a diffraction vectord
    Q = np.transpose(np.stack(diffraction_vectors, axis=2), axes=(0, 1, 3, 2))

    m, n, _, k = Q.shape
    mask = ~np.any(np.isnan(Q.reshape(m * n, 3 * k)), axis=1).reshape(m, n)

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


def diffraction_vectors(
    moment_map,
    crystal,
    lambda_0,
    crl,
):
    """Reconstruct the sample-Q vectors from the mean angular values in phi, chi and theta.

    This deploys the associated rotatations to bring the Q-reference vector the Bragg condition.

    Args:
        moment_map (:obj:`numpy.ndarray`): The mean angular values in phi, chi and theta. shape=(*detector.shape, 3)
        crystal (:obj:`Crystal`): The crystal object.
        lambda_0 (:obj:`float`): The wavelength of the beam.
        crl (:obj:`CRL`): The compound refractive lens object.

    Returns:
        (:obj:`numpy.ndarray`): The reconstructed Q vectors in sample space. shape=(*detector.shape, 3)
    """

    # TODO: implement this with an interface that does not require the crystal and crl
    # objects This should be part of the darling module.

    # NOTE: tested against slow poke loop code and passed. 11 Dec.

    muf = moment_map.reshape(-1, 3)

    x, y, z = np.eye(3)

    R_omega = crystal.goniometer.get_R_omega(crystal.goniometer.omega)
    rotation_eta = Rotation.from_rotvec(x * (crl.eta))

    R_s = crystal.goniometer.get_R_top(muf[:, np.newaxis, 1], muf[:, np.newaxis, 2])
    d_rec = lambda_0 / (2 * np.sin(crl.theta + muf[:, 0]))
    Q_norm = 2 * np.pi / d_rec
    rotation_th = Rotation.from_rotvec(
        (y.reshape(3, 1) * (-2 * (crl.theta + muf[:, 0]))).T
    ).as_matrix()

    ko = rotation_eta.as_matrix() @ (rotation_th @ x).T
    ki = x.reshape(3, 1)
    _Q = ko - ki
    _Q = _Q / np.linalg.norm(_Q, axis=0)

    Q_sample_0 = R_omega.T @ _Q * Q_norm

    Q_rec = np.zeros_like(muf)

    R_s_T = R_s.transpose(0, 2, 1)
    Q_rec[:, 0] = np.sum(R_s_T[:, 0, :] * Q_sample_0.T, axis=1)
    Q_rec[:, 1] = np.sum(R_s_T[:, 1, :] * Q_sample_0.T, axis=1)
    Q_rec[:, 2] = np.sum(R_s_T[:, 2, :] * Q_sample_0.T, axis=1)

    return Q_rec.reshape(moment_map.shape)
