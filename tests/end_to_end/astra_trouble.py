import numpy as np
import matplotlib.pyplot as plt
import astra


if __name__ == "__main__":

    vectors = np.array([[  9.84135559e-01, -2.21772633e-04,  1.77418014e-01,  
                           5.51041864e+06, -1.24175987e+03,  9.93407382e+05, 
                          -0.00000000e+00, -1.69999867e-01, -2.12499945e-04,
                           3.01610859e-02,  2.09128752e-04, -1.67302914e-01]])
    proj_geom = astra.create_proj_geom("parallel3d_vec", 256, 256, vectors)
    vol_geom = astra.create_vol_geom((32,32,32))
    data = np.ones((32,32,32), dtype=np.float32)
    sino_id, sino = astra.create_sino3d_gpu(data, proj_geom, vol_geom)
    astra.data3d.delete(sino_id)

    fontsize = 16 # General font size for all text
    ticksize= 16 # tick size
    plt.rcParams['font.size'] = fontsize
    plt.rcParams['xtick.labelsize'] = ticksize
    plt.rcParams['ytick.labelsize'] = ticksize
    plt.style.use('dark_background')
    fig, ax = plt.subplots(1, 3, figsize=(13,7))
    im = ax[0].imshow(sino[:, 0, :][:, :])
    ax[0].set_title('Full projection')
    fig.colorbar(im, ax=ax[0], fraction=0.046, pad=0.04)
    im = ax[1].imshow(sino[:, 0, :][:, 135:165])
    ax[1].set_title('Zoom in on projection')
    fig.colorbar(im, ax=ax[1], fraction=0.046, pad=0.04)
    im = ax[2].imshow(np.diff(sino[:, 0, :],axis=1)[:, 135:165])
    ax[2].set_title('Diff along axis=1')
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()