import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import convolve1d
from scipy.spatial.transform import Rotation

from darkmod.projector import GpuProjector


class Detector(object):
    """A flat surface detector with uniform pixel size.

    Attributes:
        detector_corners (:obj:`numpy array`): Detector corners, shape=(3,3).
        pixel_size (:obj:`float`): Pixel size in units of microns.
        det_row_count (:obj:`int`): Number of pixels along detector columns.
        det_col_count (:obj:`int`): Number of pixels along detector rows.
        super_sampling (:obj:`int`): Number of super sampling points in each dimension.
            Defaults to 1, in which case a single ray integral is ompute for each
            detector pixel.
        exposure (:obj:`float`): Exposure time in arb units of time. This value will be
            a fix scale factor on the computed 2D images. Defaults to 1.
        dynamic_range (:obj:`int`): The dynamic range of the detector in arb units.
            Defaults to 64000, in which case the detector images are of type np.uint16.
            Set to np.inf to generate np.float32 images allowing for floating point
            precision (no rounding to integers).
        psf_width (:obj:`float`): The width of the point spread function (PSF) in units
            of pixels. This is the standard deviation of the gaussian kernel used to
            convolve with the detector images. If set to 0, no PSF is applied.
            Defaults to 1.
        noise (:obj:`bool`): If True, the detector images will be generated with
            noise added. The default model uses thermal + shot noise. Defaults to True.
    """

    def __init__(
        self,
        detector_corners,
        pixel_size,
        det_row_count,
        det_col_count,
        super_sampling=1,
        exposure=1,
        dynamic_range=64000,
        psf_width=1,
        noise=True,
    ):
        self.detector_corners = detector_corners
        self.pixel_size = pixel_size
        self.det_row_count = det_row_count
        self.det_col_count = det_col_count
        self.super_sampling = super_sampling

        self._projector = GpuProjector(
            self.pixel_size,
            self.det_row_count,
            self.det_col_count,
            self.super_sampling,
        )

        self._wall_mount = False
        self._orthogonal_mount = False
        self._free_mount = True

        self.exposure = exposure
        self.dynamic_range = dynamic_range
        self.noise = noise

        self.dtype = np.uint16 if self.dynamic_range <= 64000 else np.float32

        self.psf_width = psf_width
        self._pfs_kernel = self._get_psf_kernel(self.psf_width)

    def remount_to_crl(self, crl):
        """Set the detector gometry to match the center with the optical CRL axis.

            NOTE: This is only well deifned for wall mount and orthogonal mount geometries.

        Args:
            crl (obj:`darkmod.crl.CompoundRefractiveLens`): The crl.
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
        exposure=1,
        dynamic_range=64000,
        psf_width=1,
        noise=True,
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
            exposure,
            dynamic_range,
            psf_width,
            noise,
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
        exposure=1,
        dynamic_range=64000,
        psf_width=1,
        noise=True,
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
            exposure,
            dynamic_range,
            psf_width,
            noise,
        )

        detector._orthogonal_mount = True
        detector._free_mount = False

        return detector

    def readout(
        self,
        voxel_volume,
        voxel_size,
        optical_axis,
        magnification,
        sample_rotation=np.eye(3),
        sample_translation=np.zeros(3),
    ):
        """Readout the detector image from the detector chip. Similar to a render()
        call but with noise, exposure, PSF convolution and dynamic range clipping.

        Args:
            voxel_volume (:obj:`np.ndarray`): The sample voxel volume with the scalar
                diffracted intensity in each voxel. shape=(m,n,o).
            voxel_size (_type_): voxel size in microns.
            optical_axis (::obj:`np.ndarray`): The optical axis (this is the ray path
                at sample_rotation=np.eye(3)), shape=(3,)
            magnification (:obj:`float`): The crl magnification.
            sample_rotation (:obj:`np.ndarray`): The rotation matrix that brings sample vectors
                to lab frame. I.e the goniometer setting. Defaults to np.eye(3) in which case
                sample and lab frames are considered to be aligned. shape=(3,3).
            sample_translation (:obj:`np.ndarray`): The absolute translation of the sample in lab frame
                in units of microns. shape=(3,) (x,y,z). Defaults to np.zeros(3).

        Returns:
            :obj:`np.ndarray`: The detector image of shape=(det_row_count, det_col_count)
        """
        detector_image = self.render(
            voxel_volume,
            voxel_size,
            optical_axis,
            magnification,
            sample_rotation,
            sample_translation,
        )

        # simulate the exposure time
        detector_image *= self.exposure

        # apply the point spread function (PSF)
        if self._pfs_kernel is not None:
            self._point_spread_function(detector_image)

        # add noise
        if self.noise:
            detector_image = self._add_noise(detector_image)

        if self.dtype == np.uint16:
            np.round(detector_image, out=detector_image)
        np.clip(detector_image, 0, self.dynamic_range, out=detector_image)
        detector_image.astype(self.dtype, copy=False)

        return detector_image

    def render(
        self,
        voxel_volume,
        voxel_size,
        optical_axis,
        magnification,
        sample_rotation=np.eye(3),
        sample_translation=np.zeros(3),
    ):
        """Render an image by projecting the voxel_volume unto the detector plane.

        NOTE: this uses tomographic ray-tracing in conjuction with crl image inversion
            and magnification considerations.

        Args:
            voxel_volume (:obj:`np.ndarray`): The sample voxel volume with the scalar
                diffracted intensity in each voxel. shape=(m,n,o).
            voxel_size (_type_): voxel size in microns.
            optical_axis (::obj:`np.ndarray`): The optical axis (this is the ray path
                at sample_rotation=np.eye(3)), shape=(3,)
            magnification (:obj:`float`): The crl magnification.
            sample_rotation (:obj:`np.ndarray`): The rotation matrix that brings sample vectors
                to lab frame. I.e the goniometer setting. Defaults to np.eye(3) in which case
                sample and lab frames are considered to be aligned. shape=(3,3).
            sample_translation (:obj:`np.ndarray`): The absolute translation of the sample in lab frame
                in units of microns. shape=(3,) (x,y,z). Defaults to np.zeros(3).

        Returns:
            :obj:`np.ndarray`: The detector image of shape=(det_row_count, det_col_count)
        """

        # The backrotation of the detector by the goniometer setting simulates the
        # fact that the sample and lab coordinate systems are not aligned. I.e the
        # sample voxel_volume is tilted. We bring the geometry to the sample frame.
        # This is a compatability requirement for the projector, which does not
        # support rotated volumes. Also, this is faster than rotating all the voxels
        # in the volume, here we require to only rotate the optical axis and the
        # detector corners.
        detector_corners = sample_rotation.T @ self.detector_corners
        ray_direction = sample_rotation.T @ optical_axis

        # The CRL will magnify the sample voxel_volume. We simulate this by passing
        # a virtual voxel size to the projector, scaled by the crl magnification.
        magnified_voxel_size = voxel_size * magnification

        # The voxel volume can now be tomographically ray-traced along the
        # diffracted ray_direction resultingin a detector image.
        detector_image = self._projector(
            voxel_volume,
            magnified_voxel_size,
            ray_direction,
            detector_corners,
            sample_translation,
        )

        # The resulting image should be inverted due to the CRL lens effect.
        detector_image = self._invert(detector_image)

        return detector_image

    def backpropagate(
        self,
        detector_image,
        voxel_volume_shape,
        voxel_size,
        optical_axis,
        magnification,
        sample_rotation=np.eye(3),
        sample_translation=np.zeros(3),
    ):
        """Backpropagate the pixel values of a detector image to the sample volume.

        This is similar to a tomographic back-projection operation.

        NOTE: this function internally handles the inversion properties of the crl such that
        the relationship beteen real-space and detector spaced can be mapped. Therefore, the
        input detector_image should be the image recorded by the detector, in the same way as
        is simulated by the render() method.

        NOTE: The values in the 3D backpropagated volumes are the values in the input detector_image
        pasted along the lines defined by the ray geometry. I.e, there is no accumulation of intensity
        or mulitplication by the voxel ray clip lenghts.

        Args:
            detector_image (:obj:`np.ndarray`): The 2d image to be backprojected. shape=(m,n)
            voxel_volume_shape (:obj:`tuple` of :obj:`int`): The sample voxel volume shape.
            voxel_size (:obj:`float`): voxel size in microns.
            optical_axis (::obj:`np.ndarray`): The optical axis (this is the ray path
                at sample_rotation=np.eye(3)), shape=(3,)
            magnification (:obj:`float`): The crl magnification.
            sample_rotation (:obj:`np.ndarray`): The rotation matrix that brings sample vectors
                to lab frame. I.e the goniometer setting. Defaults to np.eye(3) in which case
                sample and lab frames are considered to be aligned. shape=(3,3).
            sample_translation (:obj:`np.ndarray`): The absolute translation of the sample in lab frame
                in units of microns. shape=(3,) (x,y,z). Defaults to np.zeros(3).

        Returns:
            :obj:`np.ndarray`: The voxel volume populated by the backprojected values of detector_image.
        """

        detector_corners = sample_rotation.T @ self.detector_corners
        ray_direction = sample_rotation.T @ optical_axis
        magnified_voxel_size = voxel_size * magnification

        projection_image = self._invert(detector_image)

        acc_backprojection = self._projector.backproject(
            projection_image,
            voxel_volume_shape,
            magnified_voxel_size,
            ray_direction,
            detector_corners,
            sample_translation,
        )

        # This part is needed due to the fact that the number of detector pixels
        # that contribute to a voxel is not equal to 1 nor constant. Therefore,
        # we need to normalize the backprojection by the number of pixels that
        # contribute to each voxel. In general the detector will have more pixels
        # than the sample has voxels, since we are simulating a magnified view.
        normbp = self._projector.backproject(
            np.ones_like(projection_image),
            voxel_volume_shape,
            magnified_voxel_size,
            ray_direction,
            detector_corners,
            sample_translation,
        )

        backprojection = np.divide(
            acc_backprojection,
            normbp,
            where=normbp != 0,
        )
        #

        return backprojection

    def _add_noise(self, image, mu=99.453, std=2.317):
        """Thermal + Shot noise model for detector counting errors.

        Args:
            size (:obj:`np.ndarray`): Stack of images.
            mu (:obj:`float`): Mean thermal noise.
            std (:obj:`float`): Standard devation of thermal noise.

        Returns:
            :obj:`numpy array`: Noise array of shape=size.
        """
        noisy_image = np.random.poisson(lam=image).astype(float)  # shot noise
        noisy_image += np.random.normal(mu, std, size=image.shape)  # thermal noise
        return noisy_image

    def _get_psf_kernel(self, sigma, size=9):
        """Set the point spread function (PSF) kernel for convolution.

        Args:
            sigma (:obj:`float`): Standard deviation of point spread in units
                of pixels. Defaults to 1.
            size (:obj:`int`): Size of point spread kernel used in convolution.
                Must be odd. Defaults to 9.
        """
        if sigma == 0:
            return None
        assert size % 2 == 1, "psf kernel size must be odd"
        xg = np.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
        _pfs_kernel = np.exp(-0.5 * (xg**2) / (sigma**2))
        return _pfs_kernel / np.sum(_pfs_kernel)

    def _point_spread_function(self, frames):
        """Apply a point spread function (PSF) to the frames in place.

        Convolves a collection of frames with a symmetric 2d gaussian kernel,
        exploiting seperability.

        Args:
            frames (:obj:`np.ndarray`): frames of shape=(m,n,...)

        """
        convolve1d(frames, self._pfs_kernel, mode="nearest", axis=0, output=frames)
        convolve1d(frames, self._pfs_kernel, mode="nearest", axis=1, output=frames)

    def _invert(self, image):
        """Apply inversion simulating the effect of the crl on ray-paths."""
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
    # place detector in xy plane, we shall align such that the
    # optical axis is orthogonal to the detector plane, and, at
    # the same time, the projeciton of the lab-z-axis will be
    # along the v = d2 - d0 directon of the detector, such that
    # the rows on the detector alwys correspond to moving along
    # the vertical (z) direction.
    x, y, z = np.eye(3)
    dr = pixel_size * det_row_count / 2.0
    dc = pixel_size * det_col_count / 2.0
    d0 = -y * dc - z * dr
    d1 = d0 + y * det_col_count * pixel_size
    d2 = d0 + z * det_row_count * pixel_size
    detector_corners = np.array([d0, d1, d2]).T

    # rotation 1 around the lab z axis
    oxy = crl.optical_axis[0:2] / np.linalg.norm(crl.optical_axis[0:2])
    alpha = np.arccos(y[0:2] @ oxy)
    beta = np.pi / 2.0 - alpha
    R_beta = Rotation.from_rotvec(z * beta).as_matrix()
    detector_corners = R_beta @ detector_corners

    # rotation 2 around the lab z x optical_axis direction
    gamma = np.arccos(crl.optical_axis @ z)
    beta = np.pi / 2.0 - gamma
    axis = np.cross(crl.optical_axis, z)
    axis /= np.linalg.norm(axis)
    R_beta = Rotation.from_rotvec(axis * beta).as_matrix()
    detector_corners = R_beta @ detector_corners

    detector_corners += crl.optical_axis.reshape(3, 1) * crl.L

    # i.e, we have now fulfilled the following:
    # d0, d1, d2 = detector_corners.T
    # v = ((d2 - d0) / np.linalg.norm(d2 - d0))
    # u = ((d1 - d0) / np.linalg.norm(d1 - d0))
    # pz = z - crl.optical_axis * ( crl.optical_axis @ z )
    # pz /= np.linalg.norm(pz)
    # assert np.allclose(u @ crl.optical_axis, 0)
    # assert np.allclose(v @ crl.optical_axis, 0)
    # assert np.allclose( v @ pz, 1)

    return detector_corners


if __name__ == "__main__":
    data = np.zeros((128, 128, 128), dtype=np.float32)
    pn, wn = 15, 4

    data[
        data.shape[0] // 2, data.shape[1] // 2 - wn : data.shape[1] // 2 + wn, wn:-wn
    ] = np.linspace(1, 4, data.shape[2] - 2 * wn)  # z-axis
    for i in range(pn):
        data[
            data.shape[0] // 2,
            data.shape[1] // 2 - pn + i : data.shape[1] // 2 + pn - i,
            data.shape[2] - pn + i,
        ] = 4  # z

    data[
        data.shape[0] // 2, wn:-wn, data.shape[2] // 2 - wn : data.shape[2] // 2 + wn
    ] = np.linspace(4, 7, data.shape[1] - 2 * wn)[:, np.newaxis]  # y-axis
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

    from scipy.spatial.transform import Rotation

    np.random.seed(0)
    rotvec = np.random.normal(scale=np.radians(25), size=(3,))
    sample_rotation = Rotation.from_rotvec(rotvec).as_matrix()

    crl.goto(theta=np.radians(8), eta=0)

    detector = Detector.orthogonal_mount(
        crl,
        pixel_size,
        det_row_count,
        det_col_count,
        super_sampling=2,
        exposure=1,
        dynamic_range=np.inf,
        psf_width=0,
        noise=False,
    )

    image = detector.render(
        data,
        voxel_size,
        crl.optical_axis,
        crl.magnification,
        sample_rotation=sample_rotation,
    )

    backprojection = detector.backpropagate(
        image,
        data.shape,
        voxel_size,
        crl.optical_axis,
        crl.magnification,
        sample_rotation=sample_rotation,
    )
    print(np.max(backprojection), np.max(data), image.dtype)

    m, n, o = backprojection.shape

    fontsize = 32  # General font size for all text
    ticksize = 32  # tick size
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["xtick.labelsize"] = ticksize
    plt.rcParams["ytick.labelsize"] = ticksize
    plt.style.use("dark_background")

    fig, ax = plt.subplots(1, 3, figsize=(28, 14))
    ax[2].set_title("cut in Backprojection")
    im = ax[2].imshow(backprojection[m // 2, :, :])
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    ax[0].set_title("cut in Volume")
    im = ax[0].imshow(data[m // 2, :, :])
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    ax[1].set_title("Projection")
    im = ax[1].imshow(image)
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    plt.tight_layout()

    plt.show()
