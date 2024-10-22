import laue
import matplotlib.pyplot as plt
import numpy as np

from darkmod import deformation
from darkmod.goniometer import Goniometer
from darkmod.crystal import Crystal

from darkmod.resolution import DualKentGauss


resolution_function = DualKentGauss(params...)
resolution_function.compile(qgrid)

crl = CompundRefractiveLens(params...)
detector = Detector(params)


crystal = Crystal(X, Y, Z, unit_cell, orientation, defgrad)
crystal.bring_to_bragg(hkl = np.array([0, 0, 2]), energy=energy)

for phi in np.linspace(crystal.goniometer.phi-0.1, crystal.goniometer.phi+0.1, 5):
    for chi in np.linspace(crystal.goniometer.chi-0.1, crystal.goniometer.chi+0.1, 5):
        crystal.goniometer.goto(phi, chi, crystal.goniometer.omega, crystal.goniometer.mu)
        image = crystal.diffract( resolution_function, crl, detector )


if __name__ == "__main__":
    orientation = np.eye(3,3)
    a = b = c = 4.0493
    unit_cell = [a, b, c, 90., 90., 90.]
    xg   = np.linspace(-5, 5, 128)
    yg   = np.linspace(-5, 5, 128)
    zg   = np.linspace(-0.1, 0.1, 16)
    X, Y, Z = np.meshgrid(xg, yg, zg, indexing='ij')
    F = deformation.edge_dislocation(X[:,:,0], Y[:,:,0])
    defgrad = np.zeros((*X.shape, 3, 3))
    for i in range(len(zg)): defgrad[:,:,i,:,:] = F[:,:,:,:]
    energy = 17.1

    import cProfile
    import pstats
    import time
    pr = cProfile.Profile()
    pr.enable()
    t1 = time.perf_counter()

    crystal = Crystal(X, Y, Z, unit_cell, orientation, defgrad)
    Q = crystal.get_Q_lab( np.array([1, 0, 2]) )

    #crystal.align(hkl = np.array([1, 4, 2]), axis=np.array([0, 0, 1]))
    crystal.bring_to_bragg(hkl = np.array([0, 0, 2]), energy=energy)

    crystal.goniometer.info

    print(crystal.U)

    t2 = time.perf_counter()
    pr.disable()
    pr.dump_stats('tmp_profile_dump')
    ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
    ps.print_stats(15)
    print('\n\nCPU time is : ', t2-t1, 's')
    fig, axs = plt.subplots(1, 3, figsize=(12, 9), constrained_layout=True)
    for i in range(3):
        im = axs[i].imshow( Q[:, :, 0, i]-np.mean(Q[:, :, 0, i]))
        cbar = fig.colorbar(im, ax=axs[i])
    plt.show()
    raise

    # b = U @ B np.array([1,-1,0])/2.
    # n = U @ B np.array([1,1,-1])
    # t = U @ B np.array([1,1,2])
    # U_d = np.array([b,n,t]).T
    xg   = np.linspace(-5, 5, 512)
    yg   = np.linspace(-5, 5, 512)
    X, Y = np.meshgrid(xg, yg, indexing='ij')
    x0 = [(x+(5./512),y+(5./512)) for x,y in zip(xg[0::128], yg[0::128])]


    import cProfile
    import pstats
    import time
    pr = cProfile.Profile()
    pr.enable()
    t1 = time.perf_counter()
    F = edge_dislocation(X, Y, x0)
    H = F_to_H(F)
    t2 = time.perf_counter()
    pr.disable()
    pr.dump_stats('tmp_profile_dump')
    ps = pstats.Stats('tmp_profile_dump').strip_dirs().sort_stats('cumtime')
    ps.print_stats(15)
    print('\n\nCPU time is : ', t2-t1, 's')


    fig, axs = plt.subplots(3, 3, figsize=(12, 9), constrained_layout=True)
    for i in range(3):
        for j in range(3):
            im = axs[i, j].pcolormesh(X, Y, H[:, :, i, j], vmin=-1.3*1e-4, vmax=1.3*1e-4)
            cbar = fig.colorbar(im, ax=axs[i, j])
            cbar.ax.set_ylabel('H_'+str(i+1)+str(j+1))
    plt.show()

