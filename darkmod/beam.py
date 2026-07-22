import numba as nb
import numpy as np

from darkmod.distribution import Normal


class HeavysideBeam(object):
    """Represents a Heavyside beam with specified width and energy.

    The beam as uniform intensity and a rectangualr cross section in
    y-z-lab. The beam propagates along x-lab.

    Args:
        y_width (:obj:`float`): Width of the beam in the y-dimension.
        z_width (:obj:`float`): Width of the beam in the z-dimension.
        energy (:obj:`float`): Energy of the beam.
    """

    def __init__(self, y_width, z_width, energy):
        """Initialize the Heavyside beam.

        Args:
            y_width (:obj:`float`): Width of the beam in the y-dimension.
            z_width (:obj:`float`): Width of the beam in the z-dimension.
            energy (:obj:`float`): Energy of the beam.
        """
        self.y_width = y_width
        self.z_width = z_width
        self.energy = energy

    def __call__(self, x):
        """Calculate beam intensity weights based on the input positions.

        Args:
            x (:obj:`numpy.ndarray`): Lab coordinates, shape=(3,N).

        Returns:
            :obj:`numpy.ndarray`: Intensity weight for the given positions.
        """
        weights = (np.abs(x[1]) < self.y_width / 2.0) & (
            np.abs(x[2]) < self.z_width / 2.0
        )
        return weights.astype(float)


class GaussianBeam(object):
    """Represents a Gaussian beam with specified standard deviations and energy.

    The beam as Gaussian intensity cross section profile in y-z-lab.
    The beam propagates along x-lab.

    Args:
        y_std (:obj:`float`): Standard deviation of the beam in y-lab.
        z_std (:obj:`float`): Standard deviation of the beam in z-lab.
        energy (:obj:`float`): Energy of the beam.
    """

    def __init__(self, y_std, z_std, energy):
        """Initialize the Gaussian beam.

        Args:
            y_std (:obj:`float`): Standard deviation of the beam in y-lab.
            z_std (:obj:`float`): Standard deviation of the beam in z-lab.
            energy (:obj:`float`): Energy of the beam.
        """
        self.y_std = y_std
        self.z_std = z_std
        self._intensity_y = Normal(0, y_std)
        self._intensity_z = Normal(0, z_std)
        self.energy = energy

    def __call__(self, x):
        """Calculate beam intensity weights based on the input positions.

        Args:
            x (:obj:`numpy.ndarray`): Lab coordinates, shape=(3,N).

        Returns:
            :obj:`numpy.ndarray`: Intensity weight for the given positions.
        """
        return self._intensity_y(x[1]) * self._intensity_z(x[2])


@nb.njit(parallel=True, fastmath=True)
def _gaussian_line_beam_kernel(z, inv_2var, prefactor, out):
    n = z.shape[0]

    for i in nb.prange(n):
        out[i] = prefactor * np.exp(-z[i] * z[i] * inv_2var)


class GaussianLineBeam:
    """
    Gaussian line beam.

    Beam propagates along x_lab.
    Uniform along y_lab.
    Gaussian along z_lab.
    """

    def __init__(self, z_std, energy):
        self.energy = energy
        self.z_std = z_std

    @property
    def z_std(self):
        return self._z_std

    @z_std.setter
    def z_std(self, value):
        self._z_std = float(value)
        self._inv_2var = 0.5 / (self._z_std * self._z_std)
        self._prefactor = 1.0 / (np.sqrt(2.0 * np.pi) * self._z_std)

    def __call__(self, x, out=None):
        x = np.asarray(x)

        if x.ndim == 1:
            z = x
        elif x.ndim == 2:
            if x.shape[0] != 3:
                raise ValueError("x must have shape (3, n)")
            z = x[2]
        else:
            raise ValueError("x must have shape (n,) or (3, n)")

        if out is None:
            out = np.empty(z.shape[0], dtype=z.dtype)

        _gaussian_line_beam_kernel(
            z,
            np.asarray(self._inv_2var, dtype=z.dtype),
            np.asarray(self._prefactor, dtype=z.dtype),
            out,
        )

        return out


if __name__ == "__main__":
    pass
