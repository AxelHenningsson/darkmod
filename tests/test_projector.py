import unittest
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
import numpy as np

from darkmod.projector import GpuProjector


class TestGpuProjector(unittest.TestCase):

    def setUp(self):

        self.debug=False

        self.det_row_count = 256
        self.det_col_count = 256
        self.voxel_size = 0.165
        self.pixel_size = 1.2543
        self.theta = np.radians(10)
        
        s, c = np.sin(2 * self.theta), np.cos(2 * self.theta)
        Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
        self.optical_axis = Ry.T @ np.array([1, 0, 0])
        self.ray_direction = self.optical_axis

        y, z = np.array([0, 1, 0]), np.array([0, 0, 1])
        zim = Ry.T @ z
        dr = self.pixel_size * self.det_row_count / 2.0
        dc = self.pixel_size * self.det_col_count / 2.0
        d0 = self.optical_axis - y * dc - zim * dr
        d1 = d0 + y * self.det_col_count * self.pixel_size
        d2 = d0 + zim * self.det_row_count * self.pixel_size

        detector_corners = np.array([d0, d1, d2]).T

        self.projector = GpuProjector(
            detector_corners, self.pixel_size, self.det_row_count, self.det_col_count, super_sampling=2
        )

    def test_project(self):
        data = self._phantom()
        image = self.projector(data, self.voxel_size*10, self.ray_direction)

        if self.debug:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(1, 1, figsize=(7,7))
            im = ax.imshow(image)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            plt.title('y is to the left and z is up')
            plt.tight_layout()
            plt.show()

    def test_slice(self):
        data = self._phantom_xy()
        image = self.projector(data, self.voxel_size*20, self.ray_direction)

        if 1:
            plt.style.use('dark_background')
            fig, ax = plt.subplots(1, 2, figsize=(7,7))
            im = ax[0].imshow(image)
            fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
            im = ax[1].imshow(data[:,:,64])
            fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
            plt.title('y is to the left and z is up')
            plt.tight_layout()
            plt.show()
        
    def _phantom_xy(self):
        data = np.zeros((128, 128, 128), dtype=np.float32)

        a = data.shape[2] // 2

        data[32:-32,32:-32,a-10:a+10] = 1
        data[64-1:64+2,32:-32-16,a-10:a+10] = 2

        data[10+32:10+-32, 10+64-1:10+64+2,a-10:a+10] = 3
        data[10+-32:10+-28, 10+64-1:10+64+2,a-10:a+10] = 4

        return data

    def _phantom(self):
        data = np.zeros((128, 128, 128), dtype=np.float32)
        pn, wn = 15, 4

        data[
            data.shape[0] // 2, data.shape[1] // 2 - wn : data.shape[1] // 2 + wn, wn:-wn
        ] = np.linspace(
            1, 4, data.shape[2] - 2 * wn
        )  # z-axis
        for i in range(pn):
            data[
                data.shape[0] // 2,
                data.shape[1] // 2 - pn + i : data.shape[1] // 2 + pn - i,
                data.shape[2] - pn + i,
            ] = 4  # z

        data[
            data.shape[0] // 2, wn:-wn, data.shape[2] // 2 - wn : data.shape[2] // 2 + wn
        ] = np.linspace(4, 7, data.shape[1] - 2 * wn)[
            :, np.newaxis
        ]  # y-axis
        for i in range(pn):
            data[
                data.shape[0] // 2,
                data.shape[1] - pn + i,
                data.shape[2] // 2 - pn + i : data.shape[2] // 2 + pn - i,
            ] = 7  # y

        data *= 1
        return data

if __name__ == '__main__':
    unittest.main()