import numpy as np
import matplotlib.pyplot as plt
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
        weights = (np.abs(x[1]) < self.y_width // 2.0) & (
            np.abs(x[2]) < self.z_width // 2.0
        )
        return weights


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
        return self._intensity_y(x[1])*self._intensity_z(x[2])


if __name__ == "__main__":
    pass
