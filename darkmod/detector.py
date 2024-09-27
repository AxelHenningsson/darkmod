import numpy as np
import matplotlib.pyplot as plt

class Detector(object):

    def __init__(self, pixel_y_size, pixel_z_size, npix_y, npix_z):
        self.pixel_y_size = pixel_y_size
        self.pixel_z_size = pixel_z_size
        self.npix_y = npix_y
        self.npix_z = npix_z

        self._det_yhat = None
        self._det_zhat = None

    def render(self, dety, detz, intensity):
        image = np.zeros((self.npix_y, self.npix_z))
        # interpolate the scattered data, dety, detz, intensity, to a grid..
        return image

if __name__ == "__main__":
    pass