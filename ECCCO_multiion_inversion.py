#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  9 08:52:51 2021

@author: dbeabout
"""
from magixs_data_products import MaGIXSDataProducts
from overlappogram.create_gnt_image import create_gnt_image
from overlappogram.create_color_color_plot import create_color_color_plot
import numpy as np
from astropy.io import fits
import os
import re
import matplotlib.pyplot as plt
from overlappogram.inversion_field_angles import Inversion
#from overlappogram.inversion_field_angles_logts_ions import Inversion
from sklearn.linear_model import ElasticNet as enet
from overlappogram.elasticnet_model import ElasticNetModel as model
from sklearn.linear_model import LassoLars as llars
# from overlappogram.lassolars_model import LassoLarsModel as llars_model
import time
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
import warnings
from sklearn.exceptions import ConvergenceWarning

'''def calculate_weights(data, weights, sig_read, exp_time):
    # Read image
    image_hdul = fits.open(data)
    image = image_hdul[0].data
    image_height, image_width = np.shape(image)
    photon_convert = np.loadtxt(weights)[:,1]
    error_mag = np.sqrt(((image * photon_convert[None, :] * exp_time) + (exp_time * sig_read**2)) / exp_time)
    sample_weights = 1.0 / error_mag**2

    basename = os.path.splitext(data)[0]
    sample_weight_file = basename + "_sample_weights.fits"
    hdu = fits.PrimaryHDU(data = sample_weights)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(sample_weight_file, overwrite=True)
    return sample_weight_file'''

if __name__ == '__main__':

    channel = 'all'
    abund='feldman'
    fwhm='0'
    #psfs=[2,3,4,5]
    #psfs=[1,2,3,4,5]
    #psfs=[3,5]
    psfs=[4]
    for psf in psfs:


        mdp=MaGIXSDataProducts()
    #PSA for magixs1

    # Response function file path.
        response_dir ='data/'

    # Response file.
        #cube_file = response_dir + 'eccco_is_response_feldman_m_el_with_tables_lw_pm1230_'+str(psf)+'pix.fits'
        #cube_file = response_dir +'D16Feb2024_eccco_response_feldman_m_el_with_tables_s_i_slw_coopersun.fits'
        cube_file = response_dir + 'D1Aug2023_eccco_response_feldman_m_el_with_tables_lw.fits'
        #cube_file = response_dir + 'D14Feb2024_eccco_response_feldman_m_el_with_tables_lw.fits'

        #weight_file = response_dir + 'oawave_eccco_is_lw.txt'

    #Data directory and data file
        data_dir ='data/'
    #    summed_image  = data_dir + 'eccco_lw_forwardmodel_thermal_response_psf'+str(psf)+'pix_el_decon.fits'
        summed_image  = data_dir+'eccco_is_lw_forwardmodel_thermal_response_psf'+str(psf)+'pix_el.fits'
        sample_weights_data  = data_dir +'eccco_is_lw_forwardmodel_sample_weights_psf'+str(psf)+'pix_el.fits'
        #summed_image  = data_dir+'eccco_lw_forwardmodel_thermal_response_psf'+str(psf)+'pix_el.fits'
        #sample_weights_data  = data_dir +'eccco_lw_forwardmodel_sample_weights_psf'+str(psf)+'pix_el.fits'

    #The inversion directory is where the output will be written
        inversion_dir = 'output/'

    #Read in response,

        rsp_func_hdul = fits.open(cube_file)

        solution_fov_width = 2
        detector_row_range = [450, 1450]
        #detector_row_range = None
        #field_angle_range = [-2160, 2160]
        #field_angle_range = [-1260,1260]
        field_angle_range = None

        rsp_dep_name = 'logt'
        rsp_dep_list = np.round((np.arange(57,78, 1) / 10.0), decimals=1)

    #smooth_over = 'spatial'
        smooth_over = 'dependence'

        inversion = Inversion(rsp_func_cube_file=cube_file,
                          rsp_dep_name=rsp_dep_name, rsp_dep_list=rsp_dep_list,
                          solution_fov_width=solution_fov_width,smooth_over=smooth_over,field_angle_range=field_angle_range)



        #inversion.initialize_input_data(summed_image)#,image_mask_file)
        #sample_weights_data = calculate_weights(summed_image, weight_file, 8., 1.)
        #print(sample_weights_data)
        #syntax (summed image, mask image, sample weights image)
        inversion.initialize_input_data(summed_image, None, sample_weights_data)

        #new forweights
        # (photon convert file name:str, sigma read:float,exptime:float)
        #error_parameters=(response_dir + 'oawave_eccco_is_lw.txt',8.,1.)
        alphas = [5]
        rhos = [.1]
        for rho in rhos:
            for alpha in alphas:
                enet_model = enet(alpha=alpha, l1_ratio=rho, max_iter=100000,precompute=True, positive=True, fit_intercept=False, selection='cyclic')
                inv_model = model(enet_model)
                basename = os.path.splitext(os.path.basename(summed_image))[0]

                start = time.time()
                inversion.multiprocessing_invert(inv_model, inversion_dir, output_file_prefix=basename,
                #inversion.invert(inv_model, inversion_dir, output_file_prefix=basename,
                            output_file_postfix='x'+str(solution_fov_width)+'_'+str(rho*10)+'_'+str(alpha)+'_wpsf' ,detector_row_range=detector_row_range, score=False)
                end = time.time()
                print("Inversion Time =", end - start)

            ##create spectrally pure
     ####           mdp = MaGIXSDataProducts()
    ####            dir_path=inversion_dir
     #           image_list = [dir_path + 'eccco_lw_forwardmodel_thermal_response_psf'+str(psf)+'pix_el_decon_em_data_cube_x'+str(solution_fov_width)+'_'+str(rho*10)+'_'+str(alpha)+'.fits]'
    ####            image_list = [dir_path + 'eccco_is_lw_psf'+str(psf)+'pix_el_em_data_cube_x'+str(solution_fov_width)+'_'+str(rho*10)+'_'+str(alpha)+'_wpsf.fits']
    ####            gnt_dir=response_dir
     ####           gnt_file = gnt_dir + 'master_gnt_eccco_inelectrons_cm3perspersr_with_tables.fits'

    ####            rsp_dep_list = np.round((np.arange(56, 68, 1) / 10.0), decimals=1)
    ####            mdp.create_level2_0_spectrally_pure_images(image_list, gnt_file, rsp_dep_list, dir_path)


