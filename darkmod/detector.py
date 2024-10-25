import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from darkmod.projector import GpuProjector


class Detector(object):

    def __init__(
        self,
        detector_corners,
        pixel_size,
        det_row_count,
        det_col_count,
        super_sampling=1,
    ):
        self.detector_corners = detector_corners
        self.pixel_size = pixel_size
        self.det_row_count = det_row_count
        self.det_col_count = det_col_count
        self.super_sampling = super_sampling

        self._projector = GpuProjector(
            self.detector_corners,
            self.pixel_size,
            self.det_row_count,
            self.det_col_count,
            self.super_sampling,
        )

    @classmethod
    def wall_mount(
        cls,
        crl,
        pixel_size,
        det_row_count,
        det_col_count,
        super_sampling=1,
    ):
        detector_corners = _get_det_corners_wall_mount(
            crl,
            pixel_size,
            det_row_count,
            det_col_count,
        )

        return cls(
            detector_corners,
            pixel_size,
            det_row_count,
            det_col_count,
            super_sampling,
        )

    @classmethod
    def orthogonal_mount(
        cls,
        crl,
        pixel_size,
        det_row_count,
        det_col_count,
        super_sampling=1,
    ):
        detector_corners = _get_det_corners_orthogonal_mount(
            crl,
            pixel_size,
            det_row_count,
            det_col_count,
        )

        return cls(
            detector_corners,
            pixel_size,
            det_row_count,
            det_col_count,
            super_sampling,
        )

    def render(self, voxel_volume, voxel_size, crl):

        image = self._projector(
            voxel_volume,
            voxel_size * crl.magnification,
            crl.optical_axis,
        )

        return np.flipud(np.fliplr(image))


def _get_det_corners_wall_mount(
    crl,
    pixel_size,
    det_row_count,
    det_col_count,
):
    y, z = np.array([0, 1, 0]), np.array([0, 0, 1])
    dr = pixel_size * det_row_count / 2.0
    dc = pixel_size * det_col_count / 2.0
    d0 = crl.optical_axis * crl.L - y * dc - z * dr
    d1 = d0 + y * det_col_count * pixel_size
    d2 = d0 + z * det_row_count * pixel_size
    return np.array([d0, d1, d2]).T


def _get_det_corners_orthogonal_mount(
    crl,
    pixel_size,
    det_row_count,
    det_col_count,
):
    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])
    theta = np.arccos(crl.optical_axis[0]) / 2.
    s, c = np.sin(2*theta), np.cos(2*theta)
    Ry = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    zim = Ry.T @ z

    dr = pixel_size * det_row_count / 2.0
    dc = pixel_size * det_col_count / 2.0
    d0 = crl.optical_axis * crl.L - y * dc - zim * dr
    d1 = d0 + y * det_col_count * pixel_size
    d2 = d0 + zim * det_row_count * pixel_size

    detector_corners = np.array([d0, d1, d2]).T

    return detector_corners


if __name__ == "__main__":

    # data = np.ones((128, 128, 128), dtype=np.float32)


    data = np.zeros((128, 128, 128), dtype=np.float32)
    pn, wn = 15, 4

    data[data.shape[0] // 2 , data.shape[1] // 2 - wn : data.shape[1] // 2 + wn, wn:-wn] = np.linspace(
        1, 4, data.shape[2] - 2*wn
    )  # z-axis
    for i in range(pn):
        data[
            data.shape[0] // 2 ,
            data.shape[1] // 2 - pn + i : data.shape[1] // 2 + pn - i,
            data.shape[2] - pn + i,
        ] = 4  # z

    data[data.shape[0] // 2 , wn:-wn, data.shape[2] // 2 - wn : data.shape[2] // 2 + wn] = np.linspace(
        4, 7, data.shape[1] - 2*wn
    )[
        :, np.newaxis
    ]  # y-axis
    for i in range(pn):
        data[
            data.shape[0] // 2 ,
            data.shape[1] - pn + i,
            data.shape[2] // 2 - pn + i : data.shape[2] // 2 + pn - i,
        ] = 7  # y


    number_of_lenses = 50
    lens_space = 2 * 1e-3
    lens_radius = 50 * 1e-6
    refractive_decrement = 1.65 * 1e-6
    magnification = 10

    from darkmod.crl import CompundRefractiveLens
    crl = CompundRefractiveLens(number_of_lenses,
                                lens_space,
                                lens_radius,
                                refractive_decrement,
                                magnification)
    crl.goto(theta=np.radians(10), eta=0)


    # Detector size
    det_row_count = 256
    det_col_count = 256
    voxel_size = 0.165
    pixel_size = 1.2543

    detector = Detector.orthogonal_mount(
        crl, pixel_size, det_row_count, det_col_count, super_sampling=2
    )

    import cProfile
    import pstats
    import time

    pr = cProfile.Profile()
    pr.enable()
    t1 = time.perf_counter()

    image = detector.render(data, voxel_size, crl)

    t2 = time.perf_counter()
    pr.disable()
    pr.dump_stats("tmp_profile_dump")
    ps = pstats.Stats("tmp_profile_dump").strip_dirs().sort_stats("cumtime")
    ps.print_stats(15)
    print("\n\nCPU time is : ", t2 - t1, "s")

    plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    im = ax.imshow(image)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
