import numpy as np
import matplotlib.pyplot as plt
import astra


class GpuProjector(object):
    """Project sample voxels along the CRL optical axis.

    This implementation uses astra-toolbox https://github.com/astra-toolbox/astra-toolbox, which
    implements ray tracing primitives in cuda that run on NVIDIA GPUs.

    Args:
        detector_corners (:obj:`numpy array`): The corners of the detector in microns. shape=(3,3).
            Each column is a corner as: [d0,d1,d2] where d0 is the bottom right corner (as viewed
            from the crl opening). d1 is to the left of d0 and d2 above d0. d0,d1,d2 is arranged counter
            clockwise. The vector from d0 to d1 defined the negative columns of the detector and the
            vector from d0 to d2 defines the negative rows.
        pixel_size (:obj:`float`): Side length of detector pixel in units of microns.
        det_row_count (:obj:`int`): Number of pixels along the detector rows (vertical).
        det_col_count (:obj:`int`): Number of pixels along the detector columns (horizontal).
        super_sampling (:obj:`int`): Number of points to trace per detector pixel. Defautls to 1.

    Attributes:
        detector_corners (:obj:`numpy array`): The corners of the detector in microns. shape=(3,3).
            Each column is a corner as: [d0,d1,d2] where d0 is the bottom right corner (as viewed
            from the crl opening). d1 is to the left of d0 and d2 above d0. d0,d1,d2 is arranged counter
            clockwise. The vector from d0 to d1 defined the negative columns of the detector and the
            vector from d0 to d2 defines the negative rows.
        pixel_size (:obj:`float`): Side length of detector pixel in units of microns.
        det_row_count (:obj:`int`): Number of pixels along the detector rows (vertical).
        det_col_count (:obj:`int`): Number of pixels along the detector columns (horizontal).
        super_sampling (:obj:`int`): Number of points to trace per detector pixel. Defautls to 1.

    """

    def __init__(
        self,
        detector_corners,
        pixel_size,
        det_row_count,
        det_col_count,
        super_sampling,
    ):

        assert super_sampling > 0, "The super sampling must be positive"
        assert isinstance(super_sampling, int), "The super sampling must be integer"
        assert (
            det_row_count % super_sampling == 0
        ), "The super sampling must divise the detector size in an integer number of blocks."
        assert (
            det_col_count % super_sampling == 0
        ), "The super sampling must divise the detector size in an integer number of blocks."

        self.detector_corners = detector_corners
        self.pixel_size = pixel_size
        self.det_row_count = det_row_count
        self.det_col_count = det_col_count
        self.super_sampling = super_sampling

    def __call__(self, voxel_volume, voxel_size, ray_direction):
        """Project the voxel volume along the optical axis.

        The rays are parallel to the optical axis. This uses astra-toolbox to compute
        path lengths of the rays though the voxel volume.

        NOTE: The coordinate system is defined in terms of the indexing='ij' convention. I.e any voxel
            array that is projected is assumed to have axis=0 along the sample x-axis the axis=1 along the
            sample y-axis and the axis=2 along the sample z-axis. The projection that is returned is viewed
            such that axis=0 is along negative imaging z-axis and axis=1 is along negative imaging y-axis.
            I.e the produced image corresponds to looking up at the camera from the crl exit point.

        Args:
            voxel_volume (:obj:`numpy array`): The voxel data array. shape=(m,n,o).
            voxel_size (:obj:`float`): Side length of sample voxel in units of microns.
            ray_direction (:obj:`numpy array`): The propagation direction of the rays. shape=(3,)

        Returns:
            :obj:`numpy array`: The projected image. shape=(det_row_count, det_col_count).

        """
        projection_image = self._project(voxel_volume, voxel_size, ray_direction)
        projection_image = self._bin_projection(projection_image)
        return projection_image

    def _bin_projection(self, projection_image):
        """bin the image if super sampling was selected."""
        if self.super_sampling > 1:
            # each super_sampling x super_sampling block in the image is averaged
            # such that the new image size is reducde by the factor super_sampling
            # along both axis=0 and axis=1.
            m, n = projection_image.shape
            k = self.super_sampling
            projection_image = projection_image.reshape(m // k, k, n // k, k)
            projection_image = projection_image.sum(axis=(1, 3)) / (k**2)
        return projection_image

    def _project(self, voxel_volume, voxel_size, ray_direction):
        """Project a voxel volume block on gpu."""
        proj_geom = self._get_astra_projector(voxel_size, ray_direction)
        astra_voxel_volume = np.swapaxes(voxel_volume, 0, 2)
        n_slices, n_rows, n_cols = astra_voxel_volume.shape
        vol_geom = astra.create_vol_geom((n_rows, n_cols, n_slices))
        sino_id, sino = astra.create_sino3d_gpu(astra_voxel_volume, proj_geom, vol_geom)
        astra.data3d.delete(sino_id)
        return sino[:, 0, :]

    def _get_astra_projector(self, voxel_size, ray_direction):
        """Get astra vector geometry in lab frame."""
        vectors = self._get_astra_vectors(voxel_size, ray_direction)
        nc = self.super_sampling * self.det_col_count
        nr = self.super_sampling * self.det_row_count
        proj_geom = astra.create_proj_geom("parallel3d_vec", nc, nr, vectors)
        return proj_geom

    def _get_astra_vectors(self, voxel_size, ray_direction):
        """Get astra vector geometry in lab frame.

        The vector geometry corresponds to projecting aling the optical axis.

        """

        # 3D corner coordinates of the detector
        d0, d1, d2 = self.detector_corners.T
        dy = self.pixel_size * (d1 - d0) / np.linalg.norm(d1 - d0)
        dz = self.pixel_size * (d2 - d0) / np.linalg.norm(d2 - d0)

        # 3D center coordinates of the detector
        detector_center = (
            d0 + (dy * self.det_col_count / 2.0 ) + (dz * self.det_row_count / 2.0)
        )

        # vector from pixel (0,0) to (0,1) i.e detector cols
        u = -dy / (voxel_size * self.super_sampling)

        # vector from pixel (0,0) to (1,0) i.e detector rows
        v = -dz / (voxel_size * self.super_sampling)

        return np.concatenate((ray_direction, detector_center, u, v)).reshape(1, 12)


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

    # Detector size
    det_row_count = 256
    det_col_count = 256
    voxel_size = 1
    pixel_size = 2.

    # we have the optical axis to project along in lab cooridnates
    theta = np.radians(30)
    s, c = np.sin(2*theta), np.cos(2*theta)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    optical_axis = Ry.T @ np.array([1, 0, 0])
    ray_direction = optical_axis
    detector_distance = 256

    y = np.array([0, 1, 0])
    z = np.array([0, 0, 1])

    zim = z
    # theta = np.arccos(optical_axis[0]) / 2.
    # s, c = np.sin(2*theta), np.cos(2*theta)
    # Ry = np.array([[c,0,s],[0,1,0],[-s,0,c]])
    # zim = Ry.T @ z

    dr = pixel_size * det_row_count / 2.0
    dc = pixel_size * det_col_count / 2.0
    d0 = optical_axis * detector_distance - y * dc - zim * dr
    d1 = d0 + y * det_col_count * pixel_size
    d2 = d0 + zim * det_row_count * pixel_size

    detector_corners = np.array([d0, d1, d2]).T

    a = d1-d0
    b = d2-d0
    c = np.cross(a, b)
    c /= np.linalg.norm(c)
    print(c, optical_axis)
    print(d0-optical_axis*det_col_count)

    projector = GpuProjector(
        detector_corners, pixel_size, det_row_count, det_col_count, super_sampling=2
    )

    import cProfile
    import pstats
    import time

    pr = cProfile.Profile()
    pr.enable()
    t1 = time.perf_counter()

    image = projector(data, voxel_size, ray_direction)

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
