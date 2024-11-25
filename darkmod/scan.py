"""Module for performing standard DFXM scans.

Darkmod is designed to support customized scanning sequences, but it can be convenient to quickly set up
classical or standard scans. This module provides functions for common scans, such as the mosaicity scan
(scanning in phi and chi) and the strain-mosaic scan (which includes the CRL theta dimension).

These functions not only facilitate standard scans but also serve as examples for setting up custom scans.
"""

import matplotlib.pyplot as plt
import numpy as np


def phi_chi(
    hkl,
    phi_values,
    chi_values,
    crystal,
    crl,
    detector,
    beam,
    resolution_function,
    spatial_artefact=True,
    detector_noise=True,
    scan_mask=None,
):
    """
    Simulate a mosaicity scan over phi and chi angles.

    This function performs a scan of a crystal over a range of phi (rotation around the
    y-axis) and chi (rotation around the x-axis) values, simulating the diffraction
    images for a given set of Miller indices (hkl). The scan is executed for each
    combination of phi and chi angles, filling a 4D array that represents the diffraction
    signal at each angle combination.

    Args:
        hkl (:obj:`numpy array`): Miller indices to specify the crystal plane being scanned.
        phi_values (:obj:`numpy array`): Array of phi angles (in radians) to scan.
        chi_values (:obj:`numpy array`): Array of chi angles (in radians) to scan.
        crystal (:obj:`Crystal`): The crystal object to be scanned.
        crl (:obj:`darkmod.crl.CompundRefractiveLens`): CRL object representing the X-ray optical system.
        detector (:obj:`darkmod.detector.Detector`): Detector object to capture the diffracted signal.
        beam (:obj:`darkmod.beam.Beam`): Beam object representing the incident X-ray beam.
        resolution_function (:obj:`darkmod.resolution`): Function defining the resolution function in Q-space.
        spatial_artefact (:obj:`bool`): Simulate spatial artifacts due to optical axis ofset. Defaults to True.
        detector_noise (:obj:`bool`): If True, adds noise to the simulated detector data. Defaults to True.
        scan_mask (:obj:`numpy array`): A 2D boolean shape = (len(phi_values), len(chi_values)) mask. Only where
            the scan_mask is true will an image be taken. Speeds up simulation for sparse scans. Defaults to None,
            in which case all values on a grid defined by phi_values and chi_values are scanned.

    Returns:
        mosa (:obj:`numpy array`): A 4D array representing the diffraction signal over the scanned phi and chi values.
                                    The shape is (detector.det_row_count, detector.det_col_count, len(phi_values), len(chi_values)).
                                    The data is scaled (np.uint16 range) and rounded to simulate actual detector response.
    """

    # Store the initial goniometer angles to reset them later
    th0, phi0, chi0 = _get_dfxm_setup(crystal.goniometer, crl)

    # Initialize a 4D array to store the simulated diffraction patterns
    image_stack = np.zeros(
        (
            detector.det_row_count,  # Number of detector rows
            detector.det_col_count,  # Number of detector columns
            len(phi_values),  # Number of phi angle steps
            len(chi_values),  # Number of chi angle steps
        )
    )

    # Iterate over all combinations of phi and chi angles
    for i in range(len(phi_values)):
        for j in range(len(chi_values)):

            if scan_mask is None or scan_mask[i, j]:

                # Update goniometer angles to the current phi and chi values
                _set_dfxm_setup(
                    crystal.goniometer,
                    crl,
                    detector,
                    resolution_function,
                    th0,
                    phi_values[i],
                    chi_values[j],
                )

                # Simulate the diffraction pattern for the current angles
                image_stack[..., i, j] = crystal.diffract(
                    hkl,
                    resolution_function,
                    crl,
                    detector,
                    beam,
                    spatial_artefact=spatial_artefact,
                )

    # Reset the CRL, resolution function, and goniometer to the original angles
    _set_dfxm_setup(
        crystal.goniometer,
        crl,
        detector,
        resolution_function,
        th0,
        phi0,
        chi0,
    )

    # Normalize the diffraction pattern to (approximately) the camera's range
    image_stack = _normalize_image_stack(image_stack, detector_noise, detector)

    return image_stack


def theta_phi_chi(
    hkl,
    delta_theta_values,
    phi_values,
    chi_values,
    crystal,
    crl,
    detector,
    beam,
    resolution_function,
    spatial_artefact=True,
    detector_noise=True,
):
    """
    Simulate a strain-mosaicity scan over theta, phi, and chi angles.

    This function performs a scan of a crystal over a range of theta (adjusting the CRL to
    change strain sensitivity), phi (rotation around the y-axis), and chi (rotation around
    the x-axis) values. It simulates the diffraction images for a given set of Miller indices (hkl)
    for each combination of theta, phi, and chi angles. The result is a 5D array that captures the
    diffraction signal across these three scanned dimensions.

    NOTE: In this template function the detector has been selected to move with the theta scan, such
        that the relative distance and tilt between the detector and the crl remains constant while
        moving the crl in theta.

    Args:
        hkl (:obj:`numpy array`): Miller indices to specify the crystal plane being scanned.
        delta_theta_values (:obj:`numpy array`): Array of theta shift values (in radians) to scan using the CRL.
        phi_values (:obj:`numpy array`): Array of phi angles (in radians) to scan.
        chi_values (:obj:`numpy array`): Array of chi angles (in radians) to scan.
        crystal (:obj:`Crystal`): The crystal object to be scanned.
        crl (:obj:`darkmod.crl.CompoundRefractiveLens`): CRL object representing the X-ray optical system.
        detector (:obj:`darkmod.detector.Detector`): Detector object to capture the diffracted signal.
        beam (:obj:`darkmod.beam.Beam`): Beam object representing the incident X-ray beam.
        resolution_function (:obj:`darkmod.resolution`): Function defining the resolution function in Q-space.
        spatial_artefact (:obj:`bool`): Simulate spatial artifacts due to optical axis offset. Defaults to True.
        detector_noise (:obj:`bool`): If True, adds noise to the simulated detector data. Defaults to True.

    Returns:
        image_stack (:obj:`numpy array`): A 5D array representing the diffraction signal over the scanned
                                          theta, phi, and chi values. The shape is
                                          (detector.det_row_count, detector.det_col_count, len(delta_theta_values),
                                          len(phi_values), len(chi_values)). The data is scaled (np.uint16 range)
                                          and rounded to simulate actual detector response.
    """

    # Store the initial goniometer and CRL angles to reset them later
    th0, phi0, chi0 = _get_dfxm_setup(crystal.goniometer, crl)

    # Initialize a 5D array to store the simulated diffraction patterns
    image_stack = np.zeros(
        (
            detector.det_row_count,  # Number of detector rows
            detector.det_col_count,  # Number of detector columns
            len(delta_theta_values),  # Number of theta angle steps
            len(phi_values),  # Number of phi angle steps
            len(chi_values),  # Number of chi angle steps
        )
    )

    # Iterate over all combinations of theta, phi, and chi angles
    for i in range(len(delta_theta_values)):
        for j in range(len(phi_values)):
            for k in range(len(chi_values)):

                # Update goniometer and crl
                _set_dfxm_setup(
                    crystal.goniometer,
                    crl,
                    detector,
                    resolution_function,
                    th0 + delta_theta_values[i],
                    phi_values[j],
                    chi_values[k],
                )

                # Simulate the diffraction pattern for the current angles
                image_stack[..., i, j, k] = crystal.diffract(
                    hkl,
                    resolution_function,
                    crl,
                    detector,
                    beam,
                    spatial_artefact=spatial_artefact,
                )

    # Reset the CRL, resolution function, and goniometer to the original angles
    _set_dfxm_setup(
        crystal.goniometer,
        crl,
        detector,
        resolution_function,
        th0,
        phi0,
        chi0,
    )

    # Normalize the diffraction pattern to (approximately) the camera's range
    image_stack = _normalize_image_stack(image_stack, detector_noise, detector)

    return image_stack


def _get_dfxm_setup(goniometer, crl):
    """Extract the theta, phi, and chi of the current dfxm setup."""
    theta = crl.theta
    phi = goniometer.phi
    chi = goniometer.chi
    return theta, phi, chi


def _set_dfxm_setup(
    goniometer,
    crl,
    detector,
    resolution_function,
    theta,
    phi,
    chi,
):
    """Propagate and set the theta, phi, and chi of the current dfxm setup"""
    if crl.theta != theta:
        crl.goto(theta=theta, eta=crl.eta)
        detector.remount_to_crl(crl)
        resolution_function.theta_shift(theta)
    goniometer.phi = phi
    goniometer.chi = chi


def _normalize_image_stack(image_stack, detector_noise, detector):
    """
    Normalize the diffraction pattern and apply noise, simulating the camera response.

    This function scales the diffraction pattern to the dynamic range of the camera, adds noise
    if specified, and converts the image stack to a 16-bit unsigned integer format.

    Args:
        image_stack (:obj:`numpy array`): The 4D or 5D diffraction pattern stack to be processed.
        detector_noise (:obj:`bool`): If True, adds noise to the simulated detector data.
        detector (:obj:`darkmod.detector.Detector`): The detector object used to simulate noise.

    Returns:
        (:obj:`numpy array`): The normalized and noise-added diffraction pattern, rounded to 16-bit unsigned integers.
    """

    # Normalize the diffraction pattern to use (approximately) the camera's full range (uint16)
    image_stack /= np.max(image_stack)
    if detector_noise:
        # Add simulated detector noise if specified
        noise = detector.noise(image_stack.shape)
        max_noise = int(np.ceil(np.max(noise)))

        # Rescale such that np.max(image_stack)==64000
        image_stack *= (64000 - max_noise)
        image_stack += noise
    else:
        # Rescale such that np.max(image_stack)==64000
        image_stack *= 64000

    # Round values and convert to 16-bit unsigned integers for the camera format
    image_stack = image_stack.round(out=image_stack).astype(np.uint16, copy=False)

    return image_stack


if __name__ == "__main__":
    pass
