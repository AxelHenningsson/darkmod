import numpy as np


def edge_dislocation(X, Y, x0=[ [0, 0] ], v=0.3, b=2.86*1e-4):
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
    a_1 = b / ( 4 * np.pi* ( 1 - v ) )

    for x, y in x0:

        Ys, Xs = (X-x), (Y-y)
        x2, y2 = Ys*Ys, Xs*Xs
        a_2 = (x2+y2)
        a_3 = 1 / ( (a_2*a_2) )
        a_4 = -2*v*a_2

        F[:, :, 0, 0] += ( -Ys * (3*x2 +   y2 + a_4))*a_3
        F[:, :, 0, 1] += (  Xs * (3*x2 +   y2 + a_4))*a_3
        F[:, :, 1, 0] += ( -Xs * (  x2 + 3*y2 + a_4))*a_3
        F[:, :, 1, 1] += (  Ys * (  x2 -   y2 + a_4))*a_3

    F *= a_1

    for i in range(3): F[:, :, i, i] += 1

    return F