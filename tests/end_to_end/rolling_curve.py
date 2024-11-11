from darkmod.beam import GaussianLineBeam
from darkmod.detector import Detector
from darkmod.resolution import DualKentGauss, PentaGauss, TruncatedPentaGauss
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
    angular_tilt = 0.73 * 1e-3 # perhaps this is what happened in Poulsen 2017?
    # the idea is that a slight horizontal titlt of the CRL will cause the
    # NA in the horixontal plane to decrease which would explain the rolling curve
    # discrepancies.
    dh = (crl.length * np.sin( angular_tilt ))*1e-6
    FWHM_CRL_horizontal = FWHM_CRL_vertical- 2 * dh

    # # TODO: truncation wont help
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


    print('')
    print('-------------------------------------------------')
    print('Rolling prediction from Poulsen 2017')
    print('-------------------------------------------------')
    sigma_a = FWHM_CRL_horizontal / (2 * np.sqrt(2 * np.log(2)))
    dsh = 1e-9 / (2 * np.sqrt(2 * np.log(2)))
    sigma_Q_roll = (np.linalg.norm(Q_lab)/(2.*np.sin(theta)))*np.sqrt(dsh**2 + sigma_a**2)
    FWHM__Q_roll = sigma_Q_roll * (2 * np.sqrt(2 * np.log(2)))
    from darkmod.transforms import   _lab_to_Q_rot_mat
    qhatroll = _lab_to_Q_rot_mat(Q_lab)[:,1]
    ql = Q_lab / np.linalg.norm(Q_lab)
    Qr = Q_lab + qhatroll*sigma_Q_roll
    qr = Qr / np.linalg.norm(Qr)
    print('std:', np.arccos( qr @ ql )*1e3)
    Qr = Q_lab + qhatroll*FWHM__Q_roll
    qr = Qr / np.linalg.norm(Qr)
    print('FWHM:', np.arccos( qr @ ql )*1e3)
    print('-------------------------------------------------')
    print('')

    # Detector size
    det_row_count = 512
    det_col_count = 512
    pixel_size = 0.09677419354838701 * 2
    print("pixel_size", pixel_size)

    detector = Detector.wall_mount(
        crl, pixel_size, det_row_count, det_col_count, super_sampling=1
    )

    beam = GaussianLineBeam(z_std=0.2, energy=energy)

    # ROLLING CURVE
    dphis = []
    dths = []
    npoints = 61
    chi_values = np.linspace(-2, 2, npoints)*1e-3 
    rc = np.zeros((npoints,))
    for i in range(npoints):

        crystal.goniometer.chi = chi_values[i]


        image = crystal.diffract(
            hkl,
            resolution_function,
            crl,
            detector,
            beam,
        )

        Q = crystal.get_Q_lab(hkl)[0,0,0]
        q = Q / np.linalg.norm(Q)
        th = -(np.arccos( -q[0] )-np.pi/2.)
        dth = theta - th # distance to perfect bragg condition

        dphi = (1-np.cos(theta))*np.abs(crystal.goniometer.chi)
        rc[i] = image.sum()
        print(i, rc[i], dth, dphi, crystal.goniometer.chi)

        dphis.append(dphi)
        dths.append(dth)

    #f = lambda _chi: theta - (np.arccos( np.cos(theta)**2 - np.cos(_chi)*np.sin(theta)**2)/2.)
    # f = lambda _chi: np.arccos( (np.cos(theta)**2 - np.cos( 2*theta )) / (np.sin(theta)**2) ) - _chi
    
    # f1 = lambda _chi: (2*np.sqrt(-np.sin(theta)**4 + np.sin(theta)**2)) / (np.sin(theta)*np.sin(theta))

    plt.figure(figsize=(8,6))
    plt.plot(chi_values*1e3, np.array(dphis)*1e3,'yo--')
    plt.plot(chi_values*1e3, np.array(dths)*1e3,'ko--')
    # plt.plot(chi_values*1e3, f(chi_values),'ro--')
    # plt.plot(chi_values*1e3, f1(chi_values),'bo--')

    # plt.grid(True)
    # plt.show()

    def find_fwhm(x, y):
        half_max = np.max(y) / 2
        indices = np.where(y >= half_max)[0]
        fwhm = x[indices[-1]] - x[indices[0]]
        return fwhm


    w = rc/np.sum(rc)
    sigma_roll = np.sqrt(np.sum(w*(chi_values**2)))
    FWHM_roll = find_fwhm(chi_values, w)
    
    plt.style.use('dark_background')# place a text box in upper left in axes coords
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax.set_title('Rolling curve for a Diamond 1$\\bar{1}$1 at 17keV')
    props = dict(boxstyle='round', facecolor='gray', alpha=0.25)
    textstr = '\n'.join((
        r'$FWHM=%.2f$  mrad' % (FWHM_roll*1e3, ),
        r'$\sigma=%.2f$  mrad' % (sigma_roll*1e3, )
        ))
    ax.text(0.65, 0.9, textstr, transform=ax.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)
    ax.plot(chi_values*1e3, rc, 'ro--')
    ax.grid(True, alpha=0.25)
    ax.set_xlabel('$\chi$ - miliradians')
    plt.show()
