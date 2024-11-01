from darkmod.beam import GaussianLineBeam
from darkmod.detector import Detector
from darkmod.resolution import DualKentGauss, PentaGauss
from darkmod.crystal import Crystal
from darkmod.crl import CompundRefractiveLens
from darkmod.laue import keV_to_angstrom
from darkmod import laue
from darkmod.deformation import linear_gradient, unity_field
import numpy as np
import matplotlib.pyplot as plt
import cProfile
import pstats
import time


if __name__ == "__main__":

    number_of_lenses = 69
    lens_space = 1600  # microns
    lens_radius = 50  # microns
    refractive_decrement = (2.359/2.) * 1e-6
    magnification = 15.1
    crl = CompundRefractiveLens(
        number_of_lenses, lens_space, lens_radius, refractive_decrement, magnification
    )
    
    hkl = np.array([1, -1, 1])
    energy = 17 # keV
    lambda_0 = laue.keV_to_angstrom(energy)

    # Instantiate a cubic diamond crystal (Fd3m space group (space group 227)
    unit_cell = [3.56, 3.56, 3.56, 90.0, 90.0, 90.0]

    orientation = np.eye(3, 3)
    crystal = Crystal(unit_cell, orientation)

    # remount the crystal to align Q with z-axis
    crystal.align(hkl, axis=np.array([0, 0, 1]))
    crystal.remount()  # this updates U.

    # Find the reflection with goniometer motors.
    theta, eta = crystal.bring_to_bragg(hkl, energy)
    theta0 = theta

    # Bring the CRL to diffracted beam.
    crl.goto(theta, eta)

    # Discretize the crystal
    xg = np.linspace(-1, 1, 32)  # microns
    yg = np.linspace(-1, 1, 32)  # microns
    zg = np.linspace(-1, 1, 32)  # microns
    dx = xg[1] - xg[0]
    X, Y, Z = np.meshgrid(xg, yg, zg, indexing="ij")
    # defgrad = linear_gradient(
    # X.shape,
    # component=(2, 2),
    # axis=1,
    # magnitude=0.003,
    # )
    defgrad = unity_field(X.shape)

    crystal.discretize(X, Y, Z, defgrad)
    # crystal.write("test")

    Q_lab = crystal.goniometer.R @ crystal.UB_0 @ hkl

    # Beam divergence params
    desired_FWHM_N = 0.027 * 1e-3

    # Beam wavelength params
    sigma_e = (6 * 1e-5) / (2 * np.sqrt(2 * np.log(2)))
    epsilon = np.random.normal(0, sigma_e, size=(20000,))
    random_energy = energy + epsilon * energy
    sigma_lambda = laue.keV_to_angstrom(random_energy).std()
    mu_lambda = lambda_0

    FWHM_CRL_vertical = 0.556 * 1e-3
    angular_tilt = 0.83 * 1e-3 # perhaps this is what happened in Poulsen 2017?
    dh = (crl.length * np.sin( angular_tilt ))*1e-6
    FWHM_CRL_horizontal = ( FWHM_CRL_vertical - 2 * dh ) 

    # Detector size
    det_row_count = 512
    det_col_count = 512
    pixel_size = 0.09677419354838701 * 2
    print("pixel_size", pixel_size)


    resolution_function = PentaGauss(
        crl.optical_axis,
        1e-9 / (2 * np.sqrt(2 * np.log(2))),
        #desired_FWHM_N / (2 * np.sqrt(2 * np.log(2))),
        desired_FWHM_N / (2 * np.sqrt(2 * np.log(2))),
        FWHM_CRL_horizontal / (2 * np.sqrt(2 * np.log(2))),
        FWHM_CRL_vertical / (2 * np.sqrt(2 * np.log(2))),
        mu_lambda,
        sigma_lambda,
    )
    resolution_function.compile()

    M = resolution_function._get_M()

    beam = GaussianLineBeam(z_std=0.2, energy=energy)

    detector = Detector.orthogonal_mount(
        crl, pixel_size, det_row_count, det_col_count, super_sampling=2
    )
    
    # 2 THETA CURVE
    npoints = 60
    th_values = crl.theta + np.linspace(-1, 1, npoints)*1e-3


    Q0 = resolution_function._p_Q.mu

    #th_values = np.linspace(0, np.radians(90), npoints)
    #th_values = crl.theta + np.linspace(-1, 1, npoints)*1e-3

    resolution_function = PentaGauss(
        crl.optical_axis,
        1e-9 / (2 * np.sqrt(2 * np.log(2))),
        #desired_FWHM_N / (2 * np.sqrt(2 * np.log(2))),
        desired_FWHM_N / (2 * np.sqrt(2 * np.log(2))),
        FWHM_CRL_horizontal / (2 * np.sqrt(2 * np.log(2))),
        FWHM_CRL_vertical / (2 * np.sqrt(2 * np.log(2))),
        mu_lambda,
        sigma_lambda,
    )
    resolution_function.compile()
    cov0 = resolution_function._p_Q.cov
    mu0 = resolution_function._p_Q.mu

    rc = np.zeros((npoints,))

    x_lab = crystal.goniometer.R @ crystal._x
    data = beam(x_lab).reshape(X.shape)
    for i in range(npoints):

        crl.goto(theta=th_values[i], eta=crl.eta)
        detector.remount_to_crl(crl)


        # This is not exactly equivalent to recompiling interestingly...
        # but at small angles it is a very good approximation...
        # one thing that is different is the horizontal defintiion of rays
        # which are now not rotated around the imaging z axis, but around
        # the nominal imaging z axis

        # TODO: standardize this hack...
        # delta_theta = th_values[i] - theta0
        # shift =  2*M[:,4]*delta_theta
        # resolution_function._p_Q.mu = Q0 + 2*M[:,4]*delta_theta
        resolution_function.theta_shift(th_values[i])

        # This is the alernative way, in which the covariance will actually
        # change as a result of moving the CRL.
        # resolution_function = PentaGauss(
        #     crl.optical_axis,
        #     1e-9 / (2 * np.sqrt(2 * np.log(2))),
        #     #desired_FWHM_N / (2 * np.sqrt(2 * np.log(2))),
        #     desired_FWHM_N / (2 * np.sqrt(2 * np.log(2))),
        #     FWHM_CRL_horizontal / (2 * np.sqrt(2 * np.log(2))),
        #     FWHM_CRL_vertical / (2 * np.sqrt(2 * np.log(2))),
        #     mu_lambda,
        #     sigma_lambda,
        # )
        # resolution_function.compile()


        image = crystal.diffract(
            hkl,
            resolution_function,
            crl,
            detector,
            beam,
        )

        #image = detector.render(data, crystal.voxel_size, crl, crystal.goniometer.R)

        # plt.style.use('dark_background')
        # fig, ax = plt.subplots(1, 1, figsize=(7,7))
        # im = ax.imshow(image)
        # fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        # plt.tight_layout()
        # plt.show()

        rc[i] = image.sum()
        print(i, rc[i], np.degrees(crl.theta))

    w = rc/np.sum(rc)
    tth_values = 2*th_values
    sigma_tth = np.sqrt(np.sum(w*((tth_values-2*theta0)**2)))
    FWHM_tth = sigma_tth * (2 * np.sqrt(2 * np.log(2)))
    
    plt.style.use('dark_background')# place a text box in upper left in axes coords
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax.set_title('$theta$ curve for a Diamond 1$\\bar{1}$1 at 17keV')
    props = dict(boxstyle='round', facecolor='gray', alpha=0.25)
    textstr = '\n'.join((
        r'$FWHM=%.3f$  mrad' % (FWHM_tth*1e3, ),
        r'$\sigma=%.3f$  mrad' % (sigma_tth*1e3, )
        ))
    ax.text(0.65, 0.9, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax.plot(tth_values*1e3 - theta*1e3, rc, 'ro--')
    ax.grid(True, alpha=0.25)
    ax.set_xlabel('$theta$ - miliradians')
    plt.show()
