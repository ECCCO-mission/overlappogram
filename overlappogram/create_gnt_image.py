import typing as tp

import numpy as np
import pandas as pd
from astropy.io import fits


def create_gnt_image(em_data_cube_data: tp.Union[str, list], gnt_ions: np.ndarray,
                     gnt_file_fmt: str, image_file_fmt: str):
    """
    Creates a gnt image for each dependence.

    Parameters
    ----------
    em_data_cube_data : str
        Emission data cube(s) from an inversion.  If a list of EM data cubes are
        specified, the mean of the cubes are used.
    gnt_ions : np.ndarray
        List of gnt ions to create images.
    gnt_file_fmt : str
        Path including format of filename (e.g. 't_and_g_{:s}.txt').
    image_file_fmt : str
        Path including format of filename for created image FITS files (i.e. gnt_image_logt_{:.1f}).

    Returns
    -------
    None.

    """
    if type(em_data_cube_data) == str:
        image_hdul = fits.open(em_data_cube_data)
        em_data_cube = image_hdul[0].data
        em_data_cube = np.transpose(em_data_cube, axes=(1, 2, 0))
        em_data_cube_header = image_hdul[0].header
        print(np.shape(em_data_cube))
        #num_rows, num_slits, num_deps = np.shape(em_data_cube)
        height, num_slits, width = np.shape(em_data_cube)
    else:
        num_runs = len(em_data_cube_data)
        first_run = True
        for index in range(num_runs):
            # EM Data Cube
            image_hdul = fits.open(em_data_cube_data[index])
            em_data_cube = image_hdul[0].data
            em_data_cube_header = image_hdul[0].header
            em_data_cube = np.transpose(em_data_cube, axes=(1, 2, 0))
            #print(np.shape(image_hdul[0].data))
            #height, num_slits, width = np.shape(image_hdul[0].data)
            height, num_slits, width = np.shape(em_data_cube)
            if first_run:
                ref_height = height
                ref_num_slits = num_slits
                ref_width = width
                run_em_data_cube = np.zeros((num_runs, height, num_slits, width), dtype=np.float32)
                first_run = False
            else:
                assert height == ref_height and num_slits == ref_num_slits and width == ref_width
            run_em_data_cube[index, :, :, :] = em_data_cube
        em_data_cube = np.mean(run_em_data_cube, axis=0)

    binary_table_exists = True
    try:
        dep_name = image_hdul[0].header['DEPNAME']
        #print("dep name =", dep_name)
        dep_indices = image_hdul[1].data['index']
        dep_list = image_hdul[1].data[dep_name]
    #except Exception as e:
    except:
        binary_table_exists = False
        #print(repr(e))

    if binary_table_exists:
        for index in range(len(gnt_ions)):
            gnt_file = gnt_file_fmt.format(gnt_ions[index])
            gnt_data = pd.read_csv(gnt_file, delim_whitespace=True, header=None)
            gnt_logts = gnt_data.iloc[:, 0].values.astype(str)
            gnt_values = gnt_data.iloc[:, 1].values
            gnt_values_list = list(gnt_values)
            # Check list of logt used in inverversion versus logt in gnt file.
            gnt_dep_values = np.zeros(len(dep_list), dtype=np.float32)
            # Check for matching values.
            if len(gnt_logts) == len(dep_list):
                gnt_dep_values = gnt_values
            else:
                gnt_logts_list = list(gnt_logts)
                for dep_index, dep in enumerate(dep_list):
                    gnt_index = gnt_logts_list.index(f'{dep:.2}')
                    gnt_dep_values[dep_index] = gnt_values[gnt_index]

            gnt_image = (em_data_cube[:,:,0:width] * gnt_dep_values).sum(axis=2)

            gnt_image_file = image_file_fmt.format(gnt_ions[index]) + ".fits"
            fits_hdu = fits.PrimaryHDU(data = gnt_image, header = em_data_cube_header)
            fits_hdu.writeto(gnt_image_file, overwrite=True)
