import numpy as np


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
