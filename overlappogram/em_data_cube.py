from __future__ import annotations

import math
import typing as tp
from dataclasses import dataclass
from time import time

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from astropy.io import fits
from ndcube import NDCube
from photutils.datasets import apply_poisson_noise
from sklearn.linear_model import ElasticNet as enet


def create_background(num_x: int, num_y: int, background_value: float, add_noise: bool = False) -> np.ndarray:
    '''
    Create a background image for simulated data.

    Parameters
    ----------
    num_x : int
        Width of image (i.e. number of x pixels).
    num_y : int
        Height of image (i.e. number of y pixels).
    background_value : float
        Background value.
    add_noise : bool, optional
        Add Poisson noise to background. The default is False.

    Returns
    -------
    background : ndarray
        Created background.

    '''
    background = np.zeros((num_y, num_x))
    background[:, :] = background_value
    if add_noise == True:
        background = apply_poisson_noise(background, random_state=1)
    return background

@dataclass(order=True)
class EMDataCube:
    cube: NDCube
    background: tp.Optional[tp.Union[np.ndarray]] = None

    def __post_init__(self):
        # Verify wcs and data shape match
        assert self.cube.wcs.pixel_shape == np.shape(self.cube.data)[::-1]
        assert self.cube.wcs.naxis == 3
        self.inversion_image_list = []
        zero_velocities = np.where(self.cube.axis_world_coords(2) == 0.0 * u.km/u.s)
        # Verify only one zero velocity
        assert len(zero_velocities) == 1
        self.zero_velocity = zero_velocities[0][0]
        if self.background is not None:
            # Add background at zero velocity
            y, x, vel = np.shape(self.cube.data)
            assert np.ndim(self.background) == 2
            # Verify background shape matches spatial shape
            assert np.shape(self.background) == (y ,x)
            self.cube.data[:, :, self.zero_velocity] = self.background

    def add_explosive_event(self, filename: str, locations: list):
        '''
        Add explossive event to emission cube.

        Parameters
        ----------
        filename : str
            Filename of explosive event which contains velocity versus emission.
        locations : list
            One or more locations to place the explosive event in the data cube.

        Returns
        -------
        None.

        '''
        # Read explosive event file
        ee_data = np.loadtxt(filename)
        # Velocity units are km/s
        vel = ee_data[:, 0]
        em = ee_data[:, 1]
        num_y, num_x, num_vel = self.cube.wcs.array_shape
        # Create explosive event velocity vector
        vel_vector = np.zeros(num_vel)
        velocities = self.cube.axis_world_coords(2)
        #print(velocities)
        for vel_num in range(len(vel)):
            em_vel_index = np.where(velocities == vel[vel_num] * u.km/u.s)
            if len(em_vel_index) == 1:
                #print("vel index =", em_vel_index, vel[vel_num] * u.km/u.s)
                vel_vector[em_vel_index[0][0]] = em[vel_num]
        for x, y in locations:
            pixel_coords = self.cube.world_to_pixel(y, x, 0 * u.km/u.s)
            pixel_x = int(np.rint(pixel_coords[1].value))
            pixel_y = int(np.rint(pixel_coords[0].value))
            # Verify coordinates within cube
            if (pixel_x >= 0 and pixel_x < num_x) and (pixel_y >= 0 and pixel_y < num_y):
                self.cube.data[pixel_y, pixel_x, :] = 0.0
                self.cube.data[pixel_y, pixel_x, :] = vel_vector

    def create_simulated_data(self, image_list: list):
        '''
        Creates simulated camera/image data.

        Parameters
        ----------
        image_list : list
            List of Image classes.

        Returns
        -------
        None.

        '''
        num_y, num_x, num_vel = self.cube.wcs.array_shape
        y_pixels, x_pixels, vel_pixels = np.where(self.cube.data[:, :, :] != 0.0)
        for i in range(len(x_pixels)):
            x, y, z = x_pixels[i], y_pixels[i], vel_pixels[i]
            world_y, world_x, world_vel = self.cube.pixel_to_world(y * u.pix, x * u.pix, z * u.pix)
            for image in image_list:
                image.add_simulated_data(world_x, world_y, world_vel, self.cube.data[y, x, z])

    def prep_inversion(self, image_list):
        # Initialize cube data
        self.cube.data[:, :, :] = 0.0

        # Calculate crop ROI in world coordinates
        num_y, num_x, num_vel = self.cube.wcs.array_shape
        print("prep inversion", num_x, num_y, num_vel)
        self.num_em_values = num_x * num_y * num_vel
        self.x1  = np.zeros(self.num_em_values)
        self.y1 = np.zeros(self.num_em_values)
        self.vel1   = np.zeros(self.num_em_values)
        world_y, world_x, world_vel = self.cube.axis_world_coords(edges=True)
        #print("world x", world_x)
        #print("world y", world_y)
        min_world_x = world_x[0][0]
        max_world_x = world_x[num_y][num_x]
        min_world_y = world_y[0][0]
        max_world_y = world_y[num_y][num_x]
        print("min world =", min_world_x, min_world_y)
        print("max world =", max_world_x, max_world_y)

        # Create images to invert
        for image in image_list:
            self.inversion_image_list.append(image.crop_roi([min_world_y, min_world_x], [max_world_y, max_world_x]))

        # diff_image2 = self.inversion_image_list[1].cube.data[:, :] - self.inversion_image_list[2].cube.data[:, :]
        # plt.figure()
        # plt.imshow(diff_image2, origin='lower')

        # Calculate inversion data length
        self.inversion_data_len = 0
        self.inversion_data = np.array([])
        for inversion_image in self.inversion_image_list:
            image_data = inversion_image.data()
            y_pixels, x_pixels = np.shape(image_data)
            #print("x pixels =", x_pixels, "y pixels =", y_pixels)
            self.inversion_data_len += (y_pixels * x_pixels)
            self.inversion_data = np.append(self.inversion_data, np.reshape(image_data, (y_pixels * x_pixels)))

        # Create response function
        self.resp1  = np.zeros((self.num_em_values, self.inversion_data_len))

        c = 0
        for j in range(0, num_y):
            for i in range(0, num_x):
                for k in range(0, num_vel):
                    y_out, x_out, vel_out = self.cube.pixel_to_world(j * u.pix, i * u.pix, k * u.pix)
                    kernel = np.array([])
                    #print("i =", i, "j =", j, "k =", k)
                    for inversion_image in self.inversion_image_list:
                        image_kernel = inversion_image.create_kernel(x_out, y_out, vel_out)
                        kernel = np.append(kernel, image_kernel)
                    self.resp1[c,:] = kernel

                    self.x1[c] = x_out.to(u.arcsec).value  # arcsec
                    self.y1[c] = y_out.to(u.arcsec).value  # arcsec
                    self.vel1[c] = vel_out.to(u.km / u.s).value  # km / s
                    c = c + 1

        self.resp1 = self.resp1.transpose()
        self.resp0 = np.copy(self.resp1)

    def invert_data(self, alpha=0.0025, rho=0.975, slope=0.0, bias=1.0):
        # Adjust the resp1 to reflect the weight
        weight = abs(self.vel1)*slope + bias
        for i in range(0, self.num_em_values):
            self.resp1[:, i] = self.resp0[:, i] * weight[i]
#            self.resp1[:, i] = self.resp0[:, i] / weight[i]

        enet_model = enet(alpha=alpha, l1_ratio = rho, precompute=True, normalize=True, positive=True, fit_intercept=True, selection='random')
        enet_model.fit(self.resp1, self.inversion_data)
        data_out = enet_model.predict(self.resp1)
        em = enet_model.coef_

        # Take the weight out of the EM
        for i in range(0, self.num_em_values):
            em[i] = em[i] / weight[i]

        # Update cube data
        c = 0
        num_y, num_x, num_vel = self.cube.wcs.array_shape
        for j in range(0, num_y):
            for i in range(0, num_x):
                for k in range(0, num_vel):
                    self.cube.data[j, i, k] = em[c]
                    c = c + 1

        #wcs_anim = ArrayAnimatorWCS(self.cube.data,self.cube.wcs,slices = (0, 'x','y'))
        #plt.show()

        plt.figure()
        plt.scatter(self.vel1, em, c='r', marker='.')
        plot_title='alpha='+str('%f' % alpha)+' l1ratio='+str('%f' % rho)+' slope='+str('%f' % slope)
        plt.grid(b=True)
        plt.title(plot_title)

        # Display inverted data
        image_offset = 0
        for inversion_image in self.inversion_image_list:
            image_data = inversion_image.data()
            y_pixels, x_pixels = np.shape(image_data)
            inverted_data = data_out[image_offset : image_offset + (y_pixels * x_pixels)]
            print("pearson correlation =", scipy.stats.pearsonr(np.reshape(image_data, (y_pixels * x_pixels)), inverted_data))
            print("linregress =", scipy.stats.linregress(np.reshape(image_data, (y_pixels * x_pixels)), inverted_data))
            inverted_data = np.reshape(inverted_data, (y_pixels, x_pixels))
            plt.figure()
            plt.imshow(inverted_data)
            plt.gca().invert_yaxis()
            image_offset += (y_pixels * x_pixels)

    def prep_inversion1(self, image_list):
        # Initialize cube data
        self.cube.data[:, :, :] = 0.0

        # Calculate crop ROI in world coordinates
        num_y, num_x, num_vel = self.cube.wcs.array_shape
        world_y, world_x, world_vel = self.cube.axis_world_coords(edges=True)
        #print("world x", world_x)
        #print("world y", world_y)
        min_world_x = world_x[0][0]
        max_world_x = world_x[num_y][num_x]
        min_world_y = world_y[0][0]
        max_world_y = world_y[num_y][num_x]
        print("min world =", min_world_x, min_world_y)
        print("max world =", max_world_x, max_world_y)

        # Create images to invert
        for image in image_list:
            self.inversion_image_list.append(image.crop_roi([min_world_y, min_world_x], [max_world_y, max_world_x]))

        # Calculate inversion data length
        self.inversion_data_len = 0
        self.inversion_data = np.array([])
        for inversion_image in self.inversion_image_list:
            image_data = inversion_image.data()
            y_pixels, x_pixels = np.shape(image_data)
            #print("x pixels =", x_pixels, "y pixels =", y_pixels)
            self.inversion_data_len += (y_pixels * x_pixels)
            self.inversion_data = np.append(self.inversion_data, np.reshape(image_data, (y_pixels * x_pixels)))

    def invert_data1(self, alpha=0.0025, rho=0.975, slope=0.0, bias=1.0, inversion_image_data_list: tp.Optional[tp.Union[list[np.ndarray]]] = None):
        num_y, num_x, num_vel = self.cube.wcs.array_shape

        ### First pass - zero velocity only
        num_em_values = num_x * num_y * 1
        x1_i = np.zeros(num_em_values)
        y1_j = np.zeros(num_em_values)
        vel1_k = np.zeros(num_em_values)
        vel1 = np.zeros(num_em_values)

        start_time = time()

        # Create response function
        resp1  = np.zeros((num_em_values, self.inversion_data_len))

        csc_row_vec = np.array([])
        csc_col_vec = np.array([])
        csc_rf_vec = np.array([])

        c = 0
        for j in range(0, num_y):
            for i in range(0, num_x):
                #start_time1 = time()
                y_out, x_out, vel_out = self.cube.pixel_to_world(j * u.pix, i * u.pix, self.zero_velocity * u.pix)
#                y_out, x_out, vel_out = self.cube.pixel_to_world(j * u.pix, i * u.pix, (self.zero_velocity + 1) * u.pix)
                #end_time1 = time()
                #print("create kernel time =", end_time1 - start_time1)
                kernel = np.array([])
                for inversion_image in self.inversion_image_list:
                    #start_time1 = time()
                    image_kernel = inversion_image.create_kernel(x_out, y_out, vel_out)
                    #end_time1 = time()
                    #print("create kernel time =", end_time1 - start_time1)
                    kernel = np.append(kernel, image_kernel)
                resp1[c,:] = kernel
                row_vec = np.where(kernel != 0.0)
                col_vec = np.full(np.size(row_vec), c)
                rf_vec = kernel[row_vec]
                csc_row_vec = np.append(csc_row_vec, row_vec)
                csc_col_vec = np.append(csc_col_vec, col_vec)
                csc_rf_vec = np.append(csc_rf_vec, rf_vec)

                x1_i[c] = i
                y1_j[c] = j
                vel1_k[c] = self.zero_velocity
#                vel1_k[c] = (self.zero_velocity + 1)
                vel1[c] = vel_out.to(u.km / u.s).value  # km / s
                c = c + 1

        resp1 = resp1.transpose()
        resp0 = np.copy(resp1)

        #resp1 = csc_matrix((csc_rf_vec, (csc_row_vec, csc_col_vec)), shape=(self.inversion_data_len, c)).toarray()

        end_time = time()
        print("first pass, response function create time =", end_time - start_time)
        start_time = end_time

        enet_model = enet(alpha=alpha, l1_ratio = rho, precompute=True, normalize=True, positive=True, fit_intercept=True, selection='random')
        enet_model.fit(resp1, self.inversion_data)
        data_out = enet_model.predict(resp1)
        em = enet_model.coef_

        end_time = time()
        print("first pass, inversion time =", end_time - start_time)
        start_time = end_time

        # Update cube data
        c = 0
        num_y, num_x, num_vel = self.cube.wcs.array_shape
        for j in range(0, num_y):
            for i in range(0, num_x):
                self.cube.data[j, i, self.zero_velocity] = em[c]
#                self.cube.data[j, i, (self.zero_velocity + 1)] = em[c]
                c = c + 1

        plt.figure()
        plt.scatter(vel1, em, c='r', marker='.')
        plot_title='alpha='+str('%f' % alpha)+' l1ratio='+str('%f' % rho)+' slope='+str('%f' % slope)
        plt.grid(b=True)
        plt.title(plot_title)

        # Display inverted data
        image_offset = 0
        for inversion_image in self.inversion_image_list:
            image_data = inversion_image.data()
            y_pixels, x_pixels = np.shape(image_data)
            inverted_data = data_out[image_offset : image_offset + (y_pixels * x_pixels)]
            print("pearson correlation =", scipy.stats.pearsonr(np.reshape(image_data, (y_pixels * x_pixels)), inverted_data))
            print("linregress =", scipy.stats.linregress(np.reshape(image_data, (y_pixels * x_pixels)), inverted_data))
            inverted_data = np.reshape(inverted_data, (y_pixels, x_pixels))
            plt.figure()
            plt.imshow(inverted_data, origin='lower')
            image_offset += (y_pixels * x_pixels)
            diff_image = inverted_data - image_data
            # plt.figure()
            # plt.imshow(diff_image, origin='lower')
            plt_fig = plt.figure()
            plt_im = plt.imshow(diff_image, origin='lower')
            plt_fig.colorbar(plt_im)
            plt.show()
            diff_count = np.count_nonzero(diff_image < 0.0)
            print("diff cout = ", diff_count)
            #print(diff_image)

        # Calculate intensity
        intensity_values = np.zeros((num_y, num_x), dtype=np.float64)
        for x in range(num_x):
            for y in range(num_y):
                intensity = np.sum(self.cube.data[y, x, :])
                intensity_values[y, x] = intensity

        y, x = np.shape(intensity_values)
        X, Y = np.meshgrid(np.linspace(0, x, len(intensity_values[0,:])), np.linspace(0, y, len(intensity_values[:,0])))
        fig = plt.figure()
        ax=fig.add_subplot(111, projection='3d')
        cp = ax.scatter3D(X, Y, intensity_values)
        #fig.colorbar(cp) # Add a colorbar to a plot
        ax.set_title('Intensity')
        plt.show()

        ### Second pass
        zero_count = np.count_nonzero(intensity_values == 0.0)
#        zero_count = np.count_nonzero(intensity_values <= 50.0)
        nonzero_count = intensity_values.size - zero_count
        print("second pass ", zero_count, nonzero_count)
#        num_em_values = nonzero_count + (zero_count * num_vel)
        num_em_values = (nonzero_count * num_vel) + zero_count
        x1_i = np.zeros(num_em_values)
        y1_j = np.zeros(num_em_values)
        vel1_k = np.zeros(num_em_values)
        vel1 = np.zeros(num_em_values)

        start_time = time()

        # Create response function
        resp1  = np.zeros((num_em_values, self.inversion_data_len))

        csc_row_vec = np.array([])
        csc_col_vec = np.array([])
        csc_rf_vec = np.array([])

        c = 0
        for j in range(0, num_y):
            for i in range(0, num_x):
#                if intensity_values[j, i] == 0.0:
                if intensity_values[j, i] != 0.0:
#                if intensity_values[j, i] > 50.0:
                    for k in range(0, num_vel):
                        y_out, x_out, vel_out = self.cube.pixel_to_world(j * u.pix, i * u.pix, k * u.pix)
                        kernel = np.array([])
                        for inversion_image in self.inversion_image_list:
                            image_kernel = inversion_image.create_kernel(x_out, y_out, vel_out)
                            kernel = np.append(kernel, image_kernel)
                        resp1[c,:] = kernel
                        row_vec = np.where(kernel != 0.0)
                        col_vec = np.full(np.size(row_vec), c)
                        rf_vec = kernel[row_vec]
                        csc_row_vec = np.append(csc_row_vec, row_vec)
                        csc_col_vec = np.append(csc_col_vec, col_vec)
                        csc_rf_vec = np.append(csc_rf_vec, rf_vec)

                        x1_i[c] = i
                        y1_j[c] = j
                        vel1_k[c] = k
                        vel1[c] = vel_out.to(u.km / u.s).value  # km / s
                        c = c + 1
                else:
                    y_out, x_out, vel_out = self.cube.pixel_to_world(j * u.pix, i * u.pix, self.zero_velocity * u.pix)
                    kernel = np.array([])
                    for inversion_image in self.inversion_image_list:
                        image_kernel = inversion_image.create_kernel(x_out, y_out, vel_out)
                        kernel = np.append(kernel, image_kernel)
                    resp1[c,:] = kernel
                    row_vec = np.where(kernel != 0.0)
                    col_vec = np.full(np.size(row_vec), c)
                    rf_vec = kernel[row_vec]
                    csc_row_vec = np.append(csc_row_vec, row_vec)
                    csc_col_vec = np.append(csc_col_vec, col_vec)
                    csc_rf_vec = np.append(csc_rf_vec, rf_vec)

                    x1_i[c] = i
                    y1_j[c] = j
                    vel1_k[c] = self.zero_velocity
                    vel1[c] = vel_out.to(u.km / u.s).value  # km / s
                    c = c + 1

        end_time = time()
        print("second pass, response function create time =", end_time - start_time)
        start_time = end_time

        resp1 = resp1.transpose()
        resp0 = np.copy(resp1)

        #resp1 = csc_matrix((csc_rf_vec, (csc_row_vec, csc_col_vec)), shape=(self.inversion_data_len, c)).toarray()

#         # Adjust the resp1 to reflect the weight
#         print("vel out =", vel1)
#         print("slope =", slope)
#         weight = abs(vel1)*slope + bias
#         print("weight =", weight[np.where(weight != 1)])
#         for i in range(0, num_em_values):
#             resp1[:, i] = resp0[:, i] * weight[i]
# #            resp1[:, i] = resp0[:, i] / weight[i]

        enet_model = enet(alpha=alpha, l1_ratio = rho, precompute=True, normalize=True, positive=True, fit_intercept=True, selection='random')
        enet_model.fit(resp1, self.inversion_data)
        data_out = enet_model.predict(resp1)
        em = enet_model.coef_

        end_time = time()
        print("second pass, inversion time =", end_time - start_time)
        start_time = end_time

        # # Take the weight out of the EM
        # for i in range(0, num_em_values):
        #     em[i] = em[i] / weight[i]

        # Update cube data
        for c in range(0, num_em_values):
            self.cube.data[int(y1_j[c]), int(x1_i[c]), int(vel1_k[c])] = em[c]

        plt.figure()
        plt.scatter(vel1, em, c='r', marker='.')
        plot_title='alpha='+str('%f' % alpha)+' l1ratio='+str('%f' % rho)+' slope='+str('%f' % slope)
        plt.grid(b=True)
        plt.title(plot_title)

        # Display inverted data
        image_offset = 0
        image_count = 0
        for inversion_image in self.inversion_image_list:
            image_data = inversion_image.data()
            y_pixels, x_pixels = np.shape(image_data)
            inverted_data = data_out[image_offset : image_offset + (y_pixels * x_pixels)]
            print("pearson correlation =", scipy.stats.pearsonr(np.reshape(image_data, (y_pixels * x_pixels)), inverted_data))
            print("linregress =", scipy.stats.linregress(np.reshape(image_data, (y_pixels * x_pixels)), inverted_data))
            inverted_data = np.reshape(inverted_data, (y_pixels, x_pixels))
            if inversion_image_data_list is not None:
                crop_roi_coords = inversion_image.get_crop_roi_coords()
                print("crop_roi_coords =", crop_roi_coords)
                if crop_roi_coords == []:
                    continue
                y1, x1 = crop_roi_coords[0]
                y2, x2 = crop_roi_coords[1]
                inversion_image_data_list[image_count][y1:y2, x1:x2] = inverted_data
            # plt.figure()
            # plt.imshow(inverted_data)
            # plt.gca().invert_yaxis()
            image_offset += (y_pixels * x_pixels)
            image_count += 1

#             # diff_image = image_data - inverted_data
#             # plt_fig = plt.figure()
#             # plt_im = plt.imshow(diff_image, origin='lower')
#             # plt_fig.colorbar(plt_im)
#             # plt.show()

    def crop_tile(self, x1: int, y1: int, x2: int, y2: int) -> EMDataCube:
        '''
        Create a slice of the emission data cube as a tile for inversions.

        Parameters
        ----------
        x1 : int
            Lower x coordinate.
        y1 : int
            Lower y coordinate.
        x2 : int
            Upper x coordinate.
        y2 : int
            Upper y coordinate.

        Returns
        -------
        EMDataCube
            A slice/tile reference of the emission data cube.

        '''
        em_tile = self.cube[y1:y2, x1:x2, :]
        print(em_tile)

        new_em = EMDataCube(em_tile)

        return new_em

    def calculate_moments(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        '''
        Calculate moments 0 - 3.

        Returns
        -------
        intensity_values : TYPE
            DESCRIPTION.
        vbar_values : TYPE
            DESCRIPTION.
        line_width_values : TYPE
            DESCRIPTION.
        skew_values : TYPE
            DESCRIPTION.

        '''
        num_y, num_x, num_vel = self.cube.wcs.array_shape
        intensity_values = np.zeros((num_y, num_x), dtype=np.float64)
        vbar_values = np.zeros((num_y, num_x), dtype=np.float64)
        line_width_values = np.zeros((num_y, num_x), dtype=np.float64)
        skew_values = np.zeros((num_y, num_x), dtype=np.float64)
        # Velocities in km/s
        velocities = self.cube.axis_world_coords(2) / 1000.0
        velocities = velocities[:].value
        #print("velocities =", velocities, num_x, num_y)
        for x in range(num_x):
            for y in range(num_y):
                # Calculate mathematical moments 0 - 3
                intensity = vbar = line_width = skew = 0.0
                intensity = np.sum(self.cube.data[y, x, :])
                if intensity != 0.0:
                    vbar = np.sum(self.cube.data[y, x, :] * velocities / intensity)
                    # Calculate line width and skew
                    line_width = math.sqrt(np.sum(self.cube.data[y, x, :] * (velocities - vbar)**2. / intensity))
                    skew = np.sum(self.cube.data[y, x, :] * (velocities - vbar)**3. / intensity)
                    if skew < 0.0:
                        skew = (abs(skew)**(1./3.)) * -1.
                    else:
                        skew = skew**(1./3.)
                    intensity_values[y, x] = intensity
                    vbar_values[y, x] = vbar
                    line_width_values[y, x] = line_width
                    skew_values[y, x] = skew
        return intensity_values, vbar_values, line_width_values, skew_values

    def write(self, filename : str):
        fits_header = self.cube.wcs.to_header()
        fits_hdu = fits.PrimaryHDU(data = self.cube.data, header = fits_header)
        fits_hdu.writeto(filename, overwrite=True)
