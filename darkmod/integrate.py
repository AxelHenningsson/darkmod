import numpy as np
from scipy.spatial import ConvexHull, HalfspaceIntersection


class DegenerateVoxelIntegrator:
    """
    Integrates embedded 1D functions over finite cubic isotropic domains.
    """

    def __init__(self, cube_side_length):
        self.cube_side_length = cube_side_length

        self._unit_cube_vertices_0 = np.array(
            [
                [-0.5, -0.5, -0.5],
                [0.5, -0.5, -0.5],
                [0.5, 0.5, -0.5],
                [-0.5, 0.5, -0.5],
                [-0.5, -0.5, 0.5],
                [0.5, -0.5, 0.5],
                [0.5, 0.5, 0.5],
                [-0.5, 0.5, 0.5],
            ]
        )

    def get_cube_hull(self, cube_orientation):
        vertices = (cube_orientation @ self._unit_cube_vertices_0.T).T
        return ConvexHull(vertices * self.cube_side_length)

    def get_halfspaces(self, hull):
        return np.vstack([np.array(hull.equations), np.array([0, 0, -1, 0])])

    def get_hull_interior_point(self, hull):
        return np.mean(hull.points, axis=0)

    def get_translation_range(self, hull, nsteps):
        zmin = np.min(hull.points[:, 2])
        zmax = np.max(hull.points[:, 2])
        dz = zmax - zmin
        return np.linspace(0, dz, nsteps)

    def get_cross_section_area(self, halfspaces, hull_interior_point):
        hsi_points = HalfspaceIntersection(
            halfspaces,
            hull_interior_point,
        ).intersections
        mask = np.abs(hsi_points[:, 2] - halfspaces[-1, -1]) < 1e-8
        return ConvexHull(hsi_points[mask, 0:2]).volume

    def reflection_fill_csa(self, cross_section_areas):
        N = len(cross_section_areas)
        ii = N // 2
        for k, i in enumerate(range(ii, N)):
            cross_section_areas[i] = cross_section_areas[ii - k]
        return cross_section_areas

    def csa(self, cube_orientation, nsteps):
        if nsteps % 2 == 0:
            raise ValueError("nsteps must be odd")
        hull = self.get_cube_hull(cube_orientation)
        halfspaces = self.get_halfspaces(hull)
        zmin = np.min(hull.points[:, 2])
        interior_point = self.get_hull_interior_point(hull) + 1e-8
        cube_translations = self.get_translation_range(hull, nsteps)
        cross_section_areas = np.zeros((nsteps,))
        for i in range(1, (nsteps) // 2 + 1):
            s = cube_translations[i]
            halfspaces[-1, -1] = zmin + s  # moves the intersecting plane
            if interior_point[2] > halfspaces[-1, -1]:
                cross_section_areas[i] = self.get_cross_section_area(
                    halfspaces,
                    interior_point,
                )
        cross_section_areas = self.reflection_fill_csa(cross_section_areas)
        return cross_section_areas, cube_translations + zmin

    def interpolate_to_grid(self, ds, coord1D, func1d):
        _half_upper_grid = np.arange(coord1D[len(coord1D) // 2], coord1D[-1] + ds, ds)
        _half_lower_grid = np.flip(
            np.arange(coord1D[len(coord1D) // 2], coord1D[0] - ds, -ds)
        )
        interp_coord1D = np.zeros((len(_half_upper_grid) + len(_half_lower_grid) - 1,))
        interp_coord1D[0 : len(_half_lower_grid)] = _half_lower_grid[:]
        interp_coord1D[len(_half_lower_grid) :] = _half_upper_grid[1:]
        return interp_coord1D, np.interp(interp_coord1D, coord1D, func1d)

    def get_moving_window_integral(self, func1d, coord1D, csa, cube_translations):
        if len(func1d) % 2 == 0:
            raise ValueError("len(func1d) must be odd")
        if len(coord1D) != len(func1d):
            raise ValueError("len(coord1D) must be equal to len(func1d)")
        ds = cube_translations[1] - cube_translations[0]
        interp_coord1D, interp_func1d = self.interpolate_to_grid(ds, coord1D, func1d)
        moving_window_integral = np.convolve(interp_func1d, csa, mode="full") * ds
        coord = np.arange(0, len(moving_window_integral)) * ds
        coord = coord - coord[len(coord) // 2]
        coord -= interp_coord1D[len(interp_coord1D) // 2]
        return coord, moving_window_integral

    def __call__(self, func1d, coord1D, cube_offsets, cube_orientation, nsteps):
        """Integrate an embedded degenerate function (1D) over a set of isotropic 3D cubes.

        The cubic integration domain is rotated by cube_orientation and sliced into nsteps planes
        used to Riemann integrate func1d over the cubes. To this end func1d is linearly re-interpolated
        to a grid with spacing matching the cube slice spacing.

        Args:
            func1d (:obj:`numpy array`): 1d function to integrate over rendered on a shape=(n,) grid.
                it must hold that func1d[i] = f(coord1D[i]) for all i. The function is assumed to be
                symmetric about the center of the func1d grid and n is odd.
            coord1D (:obj:`numpy array`): 1D coordinates of the func1d grid shape=(n,) where n is odd and
                the center of the func1d grid is at coord1D[n//2].
            cube_offsets (:obj:`numpy array`): 1D cube_offsets of m isotropic cubes to be integrate func1d over shape=(m,)
            cube_orientation (:obj:`numpy array`): The constant rotation matrix of the cubes shape=(3,3).
            nsteps (:obj:`int`): Number of planar cube slices used when Riemann integrating over the cubes.
                must be odd such that the cross-section area is symmetric about the center of the cube.
                it is assumed that the coord1D array represents a coarser grid than the cube_translations array,
                if not an error is raised.

        Returns:
            :obj:`numpy array`: The scalar integral of the func1d over the cubes. shape=(len(cube_offsets),)
        """
        csa, cube_translations = self.csa(cube_orientation, nsteps)
        ds1 = cube_translations[1] - cube_translations[0]
        ds2 = coord1D[1] - coord1D[0]
        if ds2 < ds1:
            raise ValueError(
                "The coord1D stepsize is finer than the cube_translations stepsize, please increase nsteps"
            )

        coord, moving_window_integral = self.get_moving_window_integral(
            func1d, coord1D, csa, cube_translations
        )
        return np.interp(cube_offsets, coord, moving_window_integral)


if __name__ == "__main__":
    import time

    import matplotlib.pyplot as plt

    dvi = DegenerateVoxelIntegrator(cube_side_length=2)
    CB_color_cycle = [
        "#377eb8",
        "#ff7f00",
        "#4daf4a",
        "#f781bf",
        "#a65628",
        "#984ea3",
        "#999999",
        "#e41a1c",
        "#dede00",
    ]

    fontsize = 22
    ticksize = 22
    plt.style.use("dark_background")
    plt.rcParams["font.size"] = fontsize
    plt.rcParams["xtick.labelsize"] = ticksize
    plt.rcParams["ytick.labelsize"] = ticksize
    plt.rcParams["font.family"] = "Times New Roman"

    nsteps = 255
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))

    rot = np.eye(3)  # Rotation.random().as_matrix()

    t1 = time.perf_counter()
    csa, s = dvi.csa(rot, nsteps)
    ds = s[1] - s[0]
    sigma = 1 * ds
    coord1D = np.linspace(-3, 3, 511)
    f = np.exp(-0.5 * (coord1D - 0.5) ** 2 / sigma * 2)
    coord, moving_window_integral = dvi.get_moving_window_integral(f, coord1D, csa, s)
    t2 = time.perf_counter()

    print(f"Time taken: {t2 - t1} seconds")
    print(
        f"coord.shape: {coord.shape}, moving_window_integral.shape: {moving_window_integral.shape}"
    )

    ax.plot(s, csa, "-", alpha=0.5, c=CB_color_cycle[0], label="csa", linewidth=3)

    ax.plot(
        coord,
        moving_window_integral,
        "--",
        alpha=0.5,
        c=CB_color_cycle[1],
        linewidth=3,
        label="moving window integral",
    )
    ax.plot(
        coord1D,
        f,
        "-.",
        alpha=0.5,
        c=CB_color_cycle[2],
        linewidth=3,
        label="Degenerate function",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(True, alpha=0.25)
    ax.legend()
    ax.set_xlabel("Offset from bottom of cube")
    ax.set_ylabel("Area of plane-cube intersection")
    plt.show()
