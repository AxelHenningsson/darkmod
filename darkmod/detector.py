import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

class Detector(object):

    def __init__(self, pixel_y_size, pixel_z_size, npix_y, npix_z):
        self.pixel_y_size = pixel_y_size
        self.pixel_z_size = pixel_z_size
        self.npix_y = npix_y
        self.npix_z = npix_z

    def render(self, y, z, intensity):
        image = np.zeros((self.npix_y, self.npix_z))
        yi = ( (-y/self.pixel_y_size) + (self.npix_y//2) ).astype(int)
        zi = ( (-z/self.pixel_z_size) + (self.npix_z//2) ).astype(int)
        np.add.at( image, (zi, yi),  intensity)
        #image = gaussian_filter(image, sigma=1.0)
        return image

if __name__ == "__main__":
    pass