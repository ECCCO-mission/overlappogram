#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 10:38:24 2021

@author: dbeabout
"""

import numpy as np
from astropy.io import fits 
from PIL import Image

def create_color_color_plot(dep_list: list, dep_file_fmt: str,
                            saturatdep: float, lambda_scale: float,
                            output_plot_filename: str):
    '''
    

    Parameters
    ----------
    dep_list : list
        DESCRIPTION.
    dep_file_fmt : str
        DESCRIPTION.
    saturatdep : float
        DESCRIPTION.
    lambda_scale : float
        DESCRIPTION.
    output_plot_filename : str
        DESCRIPTION.

    Returns
    -------
    None.

    '''
    top = 255.0
    
    assert len(dep_list) == 3 or len(dep_list) == 4
    first_dep = True
    for dep_index in range(len(dep_list)):
        data_file = dep_file_fmt.format(dep_list[dep_index])
        image_hdul = fits.open(data_file)
        dep_data_cube = image_hdul[0].data
        height, num_slits= np.shape(dep_data_cube)
        if first_dep:
            dep_data = np.zeros((len(dep_list), height, num_slits), dtype=np.float32)
            try:
                pixel_fov_width = image_hdul[0].header['PIXELFOV']
                solution_fov_width = image_hdul[0].header['SLTNFOV']
                slit_fov_width = image_hdul[0].header['SLITFOV']
                solution_scale = float(int(round(solution_fov_width / pixel_fov_width)))
        
          #except Exception as e:
            except:
                solution_solution = 1.0
            first_dep = False
        dep_data[dep_index, :, :] = dep_data_cube
        print(np.max(dep_data[dep_index]), np.min(dep_data[dep_index]))
    average_slits = np.average(dep_data, axis = 0)
    
    for dep_index in range(len(dep_list)):
        dep_data[dep_index] = dep_data[dep_index] - average_slits
        print(np.max(dep_data[dep_index]), np.min(dep_data[dep_index]))
        dep_data[dep_index] = np.maximum(np.minimum(dep_data[dep_index], saturatdep), 0.0)
    
    if (len(dep_list) == 4):
        slit_data = dep_data[3]
        max_value = np.max(slit_data)
        min_value = np.min(slit_data)
        print("yellow", max_value, min_value)
        y_channel = np.maximum(np.minimum(  
                ((top+0.9999)*(slit_data-min_value)/(max_value-min_value)).astype(np.int16)
                , top),0)
    
    slit_data = dep_data[0]
    max_value = np.max(slit_data)
    min_value = np.min(slit_data)
    print("red", max_value, min_value)
    r_channel = np.maximum(np.minimum(  
            ((top+0.9999)*(slit_data-min_value)/(max_value-min_value)).astype(np.int16)
            , top),0)
    if (len(dep_list) == 4):
        r_channel += y_channel
    r = Image.fromarray(r_channel.astype(np.uint8), mode=None)
    slit_data = dep_data[1]
    max_value = np.max(slit_data)
    min_value = np.min(slit_data)
    print("green", max_value, min_value)
    g_channel = np.maximum(np.minimum(  
            ((top+0.9999)*(slit_data-min_value)/(max_value-min_value)).astype(np.int16)
            , top),0)
    if (len(dep_list) == 4):
        g_channel += y_channel
    g = Image.fromarray(g_channel.astype(np.uint8), mode=None)
    slit_data = dep_data[2]
    max_value = np.max(slit_data)
    min_value = np.min(slit_data)
    print("blue", max_value, min_value)
    b_channel = np.maximum(np.minimum(  
            ((top+0.9999)*(slit_data-min_value)/(max_value-min_value)).astype(np.int16)
            , top),0)
    b = Image.fromarray(b_channel.astype(np.uint8), mode=None)
    
    rgb_image = Image.merge("RGB", (r, g, b))
    scaled_image = rgb_image.resize((int(rgb_image.width * solution_scale * lambda_scale), int(rgb_image.height)))
    scaled_image.show()
    scaled_image.save(output_plot_filename)
        
