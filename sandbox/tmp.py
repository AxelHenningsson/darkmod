import cProfile
import pstats
import time

import numpy as np
from scipy.spatial.transform import Rotation

if __name__ == "__main__":
    R1 = Rotation.random()
    R2 = Rotation.random()
    R3 = Rotation.random()
    R4 = Rotation.random()
    v = np.random.rand(3, 67 * 67 * 9)
    N = 21 * 21 * 11

    def get_R():
        R = R1 * R2 * R3 * R4
        return R.as_matrix()

    def dp(R, v):
        return R @ v

    pr = cProfile.Profile()
    pr.enable()

    t1 = time.perf_counter()

    for _ in range(N):
        R = get_R()
        q = dp(R, v)

    t2 = time.perf_counter()

    pr.disable()
    pr.dump_stats("tmp_profile_dump")
    ps = pstats.Stats("tmp_profile_dump").strip_dirs().sort_stats("cumtime")
    ps.print_stats(15)
    print("\n\nCPU time is : ", t2 - t1, "s")
