"""Functions module for computation of data features over 4D fields. I.e computation of moments of mosa-scans and the like.

As an example, in a DFXM mosaicity-scan setting, using random arrays, the moments can be retrieved as:

.. code-block:: python

    import numpy as np
    import darling

    # create coordinate arrays
    phi = np.linspace(-1, 1, 8)
    chi = np.linspace(-1, 1, 16)

    # create a random data array
    detector_dim = (128, 128)
    data = np.random.rand(len(phi), len(chi), *detector_dim)
    data = data.astype(np.uint16)

    # compute the first and second moments
    mean, covariance = darling.properties.moments(data, coordinates=(phi, chi))


"""

import numba
import numpy as np


def moments(data, coordinates):
    """Compute the sample mean and covariance of a 4D DFXM data-set.

    The data-set represents a DFXM scan with 2 degrees of freedom. These could be phi and chi or phi and energy, etc.

    NOTE: Computation is done in parallel using shared memory with numba just in time compiling. For this reason
        the data array must be of type numpy uint16.

    Example in a DFXM mosaicity-scan setting using random arrays:

    .. code-block:: python

        import numpy as np
        import darling

        # create coordinate arrays
        phi = np.linspace(-1, 1, 8)
        chi = np.linspace(-1, 1, 16)

        # create a random data array
        detector_dim = (128, 128)
        data = np.random.rand(len(phi), len(chi), *detector_dim)
        data = data.astype(np.uint16)

        # compute the first and second moments
        mean, covariance = darling.properties.moments(data, coordinates=(phi, chi))

    Args:
        data (:obj:`numpy array`):  Array of shape=(n, m, a, b) where the maps over which the mean will be calculated are
            of shape=(n, m) and the field is of shape=(a, b). Must be numpy uint16. I.e the detector roi is of shape=(a,b)
            while the scan dimensions are of shape=(n, m) such that data[:,:,i,j] is a distirbution for pixel i,j.
        coordinates (:obj:`tuple` of :obj:`numpy array`): Tuple of len=2 continaning numpy 1d arrays specifying the coordinates
            in the n and m dimensions respectively. I.e, as an example, these could be the
            phi and chi angular cooridnates.

    Returns:
        :obj:`tuple` of :obj:`numpy array` : The mean map of shape=(k,l) and the covariance map of shape=(k,l,2,2).
    """
    mu = mean(data, coordinates)
    cov = covariance(data, coordinates, first_moments=mu)
    return mu, cov


def mean(data, coordinates):
    """Compute the sample mean of a 2D data map in parallel using shared memory.

    The data-set represents a DFXM scan with 2 degrees of freedom. These could be phi and chi or phi and energy, etc.

    NOTE: Computation is done in parallel using shared memory with numba just in time compiling. For this reason
        the data array must be of type numpy uint16.

    Example in a DFXM mosaicity-scan setting using random arrays:

    .. code-block:: python

        import numpy as np
        import darling

        # create coordinate arrays
        phi = np.linspace(-1, 1, 8)
        chi = np.linspace(-1, 1, 16)

        # create a random data array
        detector_dim = (128, 128)
        data = np.random.rand(len(phi), len(chi), *detector_dim)
        data = data.astype(np.uint16)

        # compute the first moments
        first_moment = darling.properties.mean(data, coordinates=(phi, chi))

    Args:
        data (:obj:`numpy array`):  Array of shape=(n, m, a, b) where the maps over which the mean will be calculated are
            of shape=(n, m) and the field is of shape=(a, b). Must be numpy uint16. I.e the detector roi is of shape=(a,b)
            while the scan dimensions are of shape=(n, m) such that data[:,:,i,j] is a distirbution for pixel i,j.
        coordinates (:obj:`tuple` of :obj:`numpy array`): Tuple of len=2 continaning numpy 1d arrays specifying the coordinates
            in the n and m dimensions respectively. I.e, as an example, these could be the
            phi and chi angular cooridnates.

    Returns:
        :obj:`numpy array` : The mean map of shape=(k,l).
    """
    _check_data(data, coordinates)
    dum = np.arange(2).astype(np.float32)
    res = np.zeros((data.shape[0], data.shape[1], 2), dtype=np.float32)
    m1_mesh, m2_mesh = _get_grid_mesh(coordinates)
    _first_moments(data, m1_mesh, m2_mesh, dum, res)
    return res


def covariance(data, coordinates, first_moments=None):
    """Compute the sample covariance of a 2D data map in parallel using shared memory.

    NOTE: Computation is done in parallel using shared memory with numba just in time compiling. For this reason
        the data array must be of type numpy uint16.

    Example in a DFXM mosaicity-scan setting using random arrays:

    .. code-block:: python

        import numpy as np
        import darling

        # create coordinate arrays
        phi = np.linspace(-1, 1, 8)
        chi = np.linspace(-1, 1, 16)

        # create a random data array
        detector_dim = (128, 128)
        data = np.random.rand(len(phi), len(chi), *detector_dim)
        data = data.astype(np.uint16)

        # compute the first moments
        first_moment = darling.properties.mean(data, coordinates=(phi, chi))

        # compute the second moments
        covariance = darling.properties.covariance(data, (phi, chi), first_moments=first_moment)

    Args:
        data (:obj:`numpy array`):  Array of shape=(n, m, a, b) where the maps over which the covariance will be calculated are
            of shape=(n, m) and the field is of shape=(a, b). Must be numpy uint16. I.e the detector roi is of shape=(a,b)
            while the scan dimensions are of shape=(n, m) such that data[:,:,i,j] is a distirbution for pixel i,j.
        coordinates (:obj:`tuple` of :obj:`numpy array`): Tuple of len=2 continaning numpy 1d arrays specifying the coordinates
            in the n and m dimensions respectively. I.e, as an example, these could be the
            phi and chi angular cooridnates.
        first_moments (:obj:`numpy array`):  Array of shape=(a, b) of the first moments. Defaults to None, in which case the first moments
            are recomputed on the fly.

    Returns:
        :obj:`numpy array` : The covariance map of shape=(k,l,2,2).
    """
    _check_data(data, coordinates)
    dum = np.arange(2).astype(np.float32)
    res = np.zeros((data.shape[0], data.shape[1], 2, 2), dtype=np.float32)
    points = _get_point_mesh(coordinates)
    if first_moments is None:
        first_moments = mean(data, coordinates)
    _second_moments(data, first_moments, points, dum, res)
    return res

def _check_data(data, coordinates):
    assert len(data.shape) == 4, "data array must be of shape=(n, m, a, b)"
    assert data.dtype == np.float32, "data must be of type uint16"
    assert len(coordinates) == 2, "coordinates tuple must be of len=2"

def _get_grid_mesh(coordinates):
    m1_mesh, m2_mesh = np.meshgrid(*coordinates, indexing="ij")
    m1_mesh, m2_mesh = m1_mesh.astype(np.float32), m2_mesh.astype(np.float32)
    return m1_mesh, m2_mesh

def _get_point_mesh(coordinates):
    m1_mesh, m2_mesh = _get_grid_mesh(coordinates)
    return np.array([m1_mesh.flatten(), m2_mesh.flatten()])

@numba.guvectorize(
    [
        (
            numba.float32[:, :],
            numba.float32[:, :],
            numba.float32[:, :],
            numba.float32[:],
            numba.float32[:],
        )
    ],
    "(n,m),(n,m),(n,m),(p)->(p)",
    nopython=True,
    target="parallel",
)
def _first_moments(chi_phi, chis, phis, dum, res):
    """Compute the sample mean of a 2D map.

    Args:
        chi_phi (:obj:`numpy array`): a 2d data map to proccess.
        chis (:obj:`numpy array`): the first coordinate mesh 2d arrays indexing ij.
        phis (:obj:`numpy array`): the second coordinate mesh 2d arrays indexing ij.
        dum (:obj:`numpy array`): dummpy variable for numba shapes.
        res (:obj:`numpy array`): array in which to store output.
    """
    I = np.sum(chi_phi)
    if I == 0:
        res[...] = np.zeros((2,))
    else:
        cchi = np.sum(chi_phi * chis)
        cphi = np.sum(chi_phi * phis)
        res[...] = [cchi / I, cphi / I]


@numba.guvectorize(
    [
        (
            numba.float32[:, :],
            numba.float32[:],
            numba.float32[:, :],
            numba.float32[:],
            numba.float32[:, :],
        )
    ],
    "(n,m),(p),(k,q),(p)->(p,p)",
    nopython=True,
    target="parallel",
)
def _second_moments(chi_phi, first_moments, points, dum, res):
    """Compute the sample covariance of a 2D map.

    Args:
        chi_phi (:obj:`numpy array`): a 2d data map to proccess.
        first_moments (:obj:`numpy array`): the first moment of the 2d data map to proccess.
        points (:obj:`numpy array`): 2,n flattened array of the coordinates.
        dum (:obj:`numpy array`): dummpy variable for numba shapes.
        res (:obj:`numpy array`): array in which to store output.
    """
    I = np.sum(chi_phi)
    if I == 0:
        res[...] = np.zeros((2, 2))
    else:
        # Equivalent of the numpy.cov function setting the chi_phi intesnity as aweights,
        # see also https://numpy.org/doc/stable/reference/generated/numpy.cov.html
        m = points.copy()
        m[0] -= first_moments[0]
        m[1] -= first_moments[1]
        a = chi_phi.flatten()
        m -= np.sum(m * a) / I
        cov = np.dot(m * a, m.T) / I
        res[...] = cov
