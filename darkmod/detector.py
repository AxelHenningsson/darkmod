import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.interpolate import griddata
from darkmod.projector import GpuProjector


class Detector(object):
    """A flat surface detector with uniform pixel size.

    Attributes:
        detector_corners (:obj:`numpy array`): Detector corners, shape=(3,3).
        pixel_size (:obj:`float`): Pixel size in units of microns.
        det_row_count (:obj:`int`): Number of pixels along detector columns.
        det_col_count (:obj:`int`): Number of pixels along detector rows.
        super_sampling (:obj:`int`): Number of super sampling points in each dimension.
            Defaults to 1, in which case a single ray integral is ompute for each detector
            pixel.

    """

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

        self._wall_mount = False
        self._orthogonal_mount = False
        self._free_mount = True

    def remount_to_crl(self, crl):
        """Set the detector gometry to match the center with the optical CRL axis.

        NOTE: This is only well deifned for wall mount and orthogonal mount geometries.

        Args:
            crl (obj:`darkmod.crl.CompoundRefractiveLens`): The crl.
_
        """
        if self._wall_mount:
            self.detector_corners = _get_det_corners_wall_mount(
                crl,
                self.pixel_size,
                self.det_row_count,
                self.det_col_count,
            )
        elif self._orthogonal_mount:
            self.detector_corners = _get_det_corners_orthogonal_mount(
                crl,
                self.pixel_size,
                self.det_row_count,
                self.det_col_count,
            )
        else:
            raise NotImplementedError(
                "The dtector is free mounted crl mounting not well defined."
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

        detector = cls(
            detector_corners,
            pixel_size,
            det_row_count,
            det_col_count,
            super_sampling,
        )

        detector._wall_mount = True
        detector._free_mount = False

        return detector

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

        detector = cls(
            detector_corners,
            pixel_size,
            det_row_count,
            det_col_count,
            super_sampling,
        )

        detector._orthogonal_mount = True
        detector._free_mount = False

        return detector

    def render(
        self,
        voxel_volume,
        voxel_size,
        crl,
        sample_rotation=np.eye(3),
    ):
        """_summary_

        Args:
            voxel_volume (:obj:`np.ndarray`): The sample voxel volume with the scalar
                diffracted intensity in each voxel. shape=(m,n,o).
            voxel_size (_type_): voxel size in microns.
            crl (obj:`darkmod.crl.CompoundRefractiveLens`): The crl.
            sample_rotation (:obj:`np.ndarray`): The rotation matrix that brings sample vectors
                to lab frame. I.e the goniometer setting. Defaults to np.eye(3) in which case
                sample and lab frames are considered to be aligned. shape=(3,3).

        Returns:
            _type_: The detector image
        """

        # The backrotation of the detector by the goniometer setting simulates the
        # fact that the sample and lab coordinate systems are not aligned. I.e the
        # sample voxel_volume is tilted. We bring the geomtry to the sample frame.
        # This is a compatability requirement for the projector, which does not
        # support rotated volumes. Also, this is faster than rotating all the voxels
        # in the volume, here we require to only rotate the optical axis and the
        # detector corners.

        self._projector.detector_corners = sample_rotation.T @ self.detector_corners
        ray_direction = sample_rotation.T @ crl.optical_axis

        # The CRL will magnify the sample voxel_volume. We simulate this by passing
        # a virtual voxel size to the projector, scaled by the crl magnification.
        magnified_voxel_size = voxel_size * crl.magnification

        # The voxel volume can now be tomographically ray-traced along the
        # diffracted ray_direction resultingin a detector image.
        detector_image = self._projector(
            voxel_volume,
            magnified_voxel_size,
            ray_direction,
        )

        # The resulting image should be inverted due to the CRL lens effect.
        detector_image = np.flipud(np.fliplr(detector_image))

        return detector_image


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
    theta = np.sign(crl.optical_axis[2]) * np.arccos(crl.optical_axis[0]) / 2.0
    s, c = np.sin(2 * theta), np.cos(2 * theta)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
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

    number_of_lenses = 50
    lens_space = 2 * 1e-3
    lens_radius = 50 * 1e-6
    refractive_decrement = 1.65 * 1e-6
    magnification = 10

    from darkmod.crl import CompundRefractiveLens

    # Detector size
    det_row_count = 256 * 2
    det_col_count = 256 * 2
    voxel_size = 0.165
    pixel_size = 1.2543
    crl = CompundRefractiveLens(
        number_of_lenses, lens_space, lens_radius, refractive_decrement, magnification
    )

    ss = []
    thetas = np.linspace(-np.radians(45), np.radians(45), 64)
    for theta in thetas:
        crl.goto(theta=theta, eta=0)

        detector = Detector.orthogonal_mount(
            crl, pixel_size, det_row_count, det_col_count, super_sampling=2
        )

        # import cProfile
        # import pstats
        # import time

        # pr = cProfile.Profile()
        # pr.enable()
        # t1 = time.perf_counter()

        image = detector.render(data, voxel_size, crl)

        # t2 = time.perf_counter()
        # pr.disable()
        # pr.dump_stats("tmp_profile_dump")
        # ps = pstats.Stats("tmp_profile_dump").strip_dirs().sort_stats("cumtime")
        # ps.print_stats(15)
        # print("\n\nCPU time is : ", t2 - t1, "s")

        print("image", image.sum())
        ss.append(image.sum())

    print("noise: ", (np.max(ss) - np.min(ss)) / np.min(ss))
    plt.figure(figsize=(8, 6))
    plt.plot(thetas, ss)
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.hist(np.array(ss) - np.mean(ss))
    plt.show()

    crl.goto(theta=-0.39, eta=0)

    detector = Detector.orthogonal_mount(
        crl, pixel_size, det_row_count, det_col_count, super_sampling=2
    )
    image = detector.render(data, voxel_size, crl)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    im = ax.imshow(image)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    crl.goto(theta=np.radians(-3), eta=0)

    detector = Detector.orthogonal_mount(
        crl, pixel_size, det_row_count, det_col_count, super_sampling=2
    )
    image = detector.render(data, voxel_size, crl)

    plt.style.use("dark_background")
    fig, ax = plt.subplots(1, 1, figsize=(7, 7))
    im = ax.imshow(image)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()
