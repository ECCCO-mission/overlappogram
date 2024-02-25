from dataclasses import dataclass
from math import sqrt

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
from astropy.constants import c, k_B
from astropy.io import fits
from astropy.table import Table
from ndcube import NDCube
from photutils.datasets import make_gaussian_sources_image

from overlappogram.element import Element


@dataclass(order=True)
class Image:
    cube: NDCube
    element: Element
    sigma_psf: np.float64
    pixel_delta_wavelength: np.float64
    camera_angle: np.float64 = 0.0
    def __post_init__(self):
        # Verify wcs and data shape match
        assert self.cube.wcs.pixel_shape == np.shape(self.cube.data)[::-1]
        assert self.cube.wcs.naxis == 2
        self.num_y_pixels, self.num_x_pixels = np.shape(self.cube.data)
        self.x0, self.y0 = self.cube.wcs.wcs.crpix
        self.angle = self.camera_angle * np.pi/180.
        self.calculate_detector_dispersion(self.element.temperature,
                                           self.element.mass,
                                           self.element.rest_wavelength,
                                           self.sigma_psf,
                                           self.pixel_delta_wavelength)
        self.sources = Table()
        self.sources['amplitude'] = [self.amplitude]
        self.sources['x_mean'] = [0]
        self.sources['y_mean'] = [0]
        self.sources['x_stddev'] = [self.sigma_along_disp]
        self.sources['y_stddev'] = [self.sigma_psf]
        self.sources['theta'] = [self.angle]
        self.crop_roi_coords = []

    def calculate_detector_dispersion(self,element_temperature, element_mass,
                                      element_rest_wavelength, sigma_psf,
                                      pixel_delta_wavelength):
        thermal_velocity = \
            np.sqrt(k_B.value * element_temperature / element_mass)/1.e3
        self.amplitude = 1.0 / (thermal_velocity * sqrt(2 * np.pi))
        self.width_of_pix_in_km_s = \
            pixel_delta_wavelength / element_rest_wavelength * (c.value / 1.e3)
        sigma_thermal = thermal_velocity/self.width_of_pix_in_km_s
        self.sigma_along_disp = np.sqrt(sigma_psf**2 + sigma_thermal**2)

    def data(self):
        return self.cube.data[:][:]

    def create_kernel(self, x, y, vel):
        #start_time = time()
#        pixel_y, pixel_x = self.cube.world_to_pixel(y, x)
        pixel_y, pixel_x = y, x
        #end_time = time()
        #print("gaussian create time =", end_time - start_time)
        if pixel_x != np.nan and pixel_y != np.nan:
#            newx0 = pixel_x.value + vel.to(u.km / u.s).value/self.width_of_pix_in_km_s * np.cos(self.angle)
#            newy0 = pixel_y.value + vel.to(u.km / u.s).value/self.width_of_pix_in_km_s * np.sin(self.angle)
            newx0 = pixel_x + vel/self.width_of_pix_in_km_s * np.cos(self.angle)
            newy0 = pixel_y + vel/self.width_of_pix_in_km_s * np.sin(self.angle)
            self.sources['x_mean'] = [newx0]
            self.sources['y_mean'] = [newy0]
            tshape = (self.num_y_pixels, self.num_x_pixels)
            #start_time = time()
            kernel = (make_gaussian_sources_image(tshape, self.sources))
            #end_time = time()
            #print("gaussian create time =", end_time - start_time)
            kernel[kernel < 1.e-3] = 0.0
            kernel = np.reshape(kernel, self.num_y_pixels * self.num_x_pixels)
            # Normalize kernel.
            kernel_sum = np.sum(kernel)
            if (kernel_sum != 0.0):
                kernel = kernel / kernel_sum
        else:
            kernel = np.zeros(self.num_y_pixels * self.num_x_pixels)
        return kernel

    def crop_roi(self, lower, upper):
        pixel_y, pixel_x = self.cube.world_to_pixel(lower[0], lower[1])
        print("crop_roi pixels ll =", pixel_y, pixel_x)
        pixel_y, pixel_x = self.cube.world_to_pixel(upper[0], upper[1])
        print("crop_roi pixels ur =", pixel_y, pixel_x)
        pixel_y, pixel_x = self.cube.world_to_pixel(lower[0], upper[1])
        print("crop_roi pixels lr =", pixel_y, pixel_x)
        pixel_y, pixel_x = self.cube.world_to_pixel(upper[0], lower[1])
        print("crop_roi pixels ul =", pixel_y, pixel_x)
        image_roi = self.cube.crop_by_coords(lower_corner=lower, upper_corner=upper)
        print(image_roi)
        plt.figure()
        plt.imshow(image_roi.data, origin='lower')
        # image_roi.plot()

        new_image = Image(image_roi, self.element, self.sigma_psf, self.pixel_delta_wavelength, self.camera_angle)

        crop_roi_coords = []
        world_y, world_x = new_image.cube.pixel_to_world(0 * u.pix, 0 * u.pix)
        pixel_y, pixel_x = self.cube.world_to_pixel(world_y, world_x)
        crop_roi_coords.append([int(np.rint(pixel_y.value)), int(np.rint(pixel_x.value))])
        print("crop_roi new image 0, 0 =", pixel_y, pixel_x)
        num_y, num_x = np.shape(new_image.data())
        world_y, world_x = new_image.cube.pixel_to_world(num_y * u.pix, num_x * u.pix)
        pixel_y, pixel_x = self.cube.world_to_pixel(world_y, world_x)
        print("crop_roi new image num_y, num_x =", pixel_y, pixel_x)
        crop_roi_coords.append([int(np.rint(pixel_y.value)), int(np.rint(pixel_x.value))])
        print("inside image crop_roi_coords =", crop_roi_coords)
        new_image.set_crop_roi_coords(crop_roi_coords)

        return new_image

    def set_crop_roi_coords(self, roi_coords):
        self.crop_roi_coords = roi_coords

    def get_crop_roi_coords(self):
        print("inside image get_crop_roi_coords =", self.crop_roi_coords)
        return self.crop_roi_coords

    def add_simulated_data(self, x, y, vel, em):
        pixel_y, pixel_x = self.cube.world_to_pixel(y, x)
        kernel = self.create_kernel(x, y, vel) * em
        kernel = np.reshape(kernel, (self.num_y_pixels, self.num_x_pixels))
        self.cube.data[:, :] += kernel

    def write(self, filename : str):
        fits_header = self.cube.wcs.to_header()
        fits_hdu = fits.PrimaryHDU(data = self.cube.data, header = fits_header)
        fits_hdu.writeto(filename, overwrite=True)
