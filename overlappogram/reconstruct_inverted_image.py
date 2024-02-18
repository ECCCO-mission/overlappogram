#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:38:24 2021

@author: dbeabout
"""

import numpy as np
from astropy.io import fits 
import pandas as pd
import os

def reconstruct_inverted_image(em_data_cube_slot_data: str, rsp_dep_file_fmt: str,
                               output_dir_path: str, rsp_dep_list: np.ndarray = None):
    '''
    Creates an image from the EM data cube.  If the response dependence
    list is None, a data cube is created for all dependences in binary table.

    Parameters
    ----------
    em_data_cube_slot_data : str
        Emission data cube from an inversion.
    rsp_dep_file_fmt : str
        Path including format of filename (e.g. 'logt_{:.1f}.txt').
    output_dir_path : str
        Path for created dependence data cube FITS files.
    rsp_dep_list : str
        Subset of the dependence to create data cube(s) for.  The default is None.

    Returns
    -------
    None.

    '''
    image_hdul = fits.open(em_data_cube_slot_data)
    em_data_cube = image_hdul[0].data
    print(np.shape(em_data_cube))
    num_rows, num_slits, num_deps = np.shape(em_data_cube)
    
    try:
        pixel_fov_width = image_hdul[0].header['PIXELFOV']
        solution_fov_width = image_hdul[0].header['SLTNFOV']
        calc_shift_width = divmod(solution_fov_width, pixel_fov_width)
        slit_shift_width = int(round(calc_shift_width[0]))
    except:
        slit_shift_width = 1
    print("slit shift width =", slit_shift_width)
    
    binary_table_exists = True
    try:
        #dep_name = image_hdul[0].header['DEPNAME']
        #print("dep name =", dep_name)
        dep_indices = image_hdul[1].data['index']
        #dep_list = image_hdul[1].data[dep_name]
        dep_list = image_hdul[1].data['ion']
        print("1", dep_indices, dep_list)
        if rsp_dep_list is not None:
            dep_mask = np.isin(dep_list, rsp_dep_list)
            new_dep_indices = []
            new_dep_list = []
            for index in range(len(dep_mask)):
                if dep_mask[index] == True:
                    new_dep_indices.append(dep_indices[index])
                    new_dep_list.append(dep_list[index])
            if len(new_dep_list) > 0:
                dep_indices = new_dep_indices
                dep_list = new_dep_list
    #except Exception as e:
    except:
        binary_table_exists = False
        #print(repr(e))
        
    image_allocated = False
        
    if binary_table_exists:
        calc_half_slits = divmod(num_slits, 2)
        num_half_slits = int(calc_half_slits[0])
        for index in range(len(dep_list)):
            dep_rsp_file = rsp_dep_file_fmt.format(dep_list[index])
            dep_rsp_data = pd.read_csv(dep_rsp_file, delim_whitespace=True)
            dep_rsp = dep_rsp_data.iloc[:, 2].values
            if image_allocated == False:
                image_data = np.zeros((num_rows, len(dep_rsp)))
                image_allocated = True
            for slit_index in range(num_slits):
                slit_shift = (slit_index - num_half_slits) * slit_shift_width
                if slit_shift < 0:
                    slit_rsp = np.pad(dep_rsp, (0, -slit_shift), mode='constant')[-slit_shift:]
                elif slit_shift > 0:
                    slit_rsp = np.pad(dep_rsp, (slit_shift, 0), mode='constant')[:-slit_shift]
                else:
                    slit_rsp = dep_rsp
                for row in range(num_rows):
                    image_data[row, :] += slit_rsp * em_data_cube[row, slit_index, dep_indices[index]]

        # Create output directory.
        os.makedirs(output_dir_path, exist_ok=True)
                    
        image_file = output_dir_path + "reconstructed_inverted_image.fits"
        print(image_file)
        hdu = fits.PrimaryHDU(image_data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(image_file, overwrite=True)
