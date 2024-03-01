import typing as tp

import numpy as np
import pandas as pd
from astropy.io import fits


def create_dependence_image(
    em_data_cube_data: tp.Union[str, list],
    rsp_dep_file_fmt: str,
    image_file_fmt: str,
    rsp_dep_list: np.ndarray = None,
):
    """
    Creates a dependence image for each dependence.  If the response dependence
    list is None, a data cube is created for all dependences in binary table.

    Parameters
    ----------
    em_data_cube_data : str
        Emission data cube(s) from an inversion.  If a list of EM data cubes are
        specified, the mean of the cubes are used.
    rsp_dep_file_fmt : str
        Path including format of filename (e.g. 'logt_{:.1f}.txt').
    image_file_fmt : str
        Path incudeing format of filename for created image FITS files (i.e. dep_image_logt_{:.1f}).
    rsp_dep_list : np.ndarray
        Subset of the dependence to create data cube(s) for.  The default is None.

    Returns
    -------
    None.

    """
    if isinstance(em_data_cube_data, str):
        image_hdul = fits.open(em_data_cube_data)
        em_data_cube = image_hdul[0].data
        em_data_cube_header = image_hdul[0].header
        print(np.shape(em_data_cube))
        num_rows, num_slits, num_deps = np.shape(em_data_cube)
    else:
        num_runs = len(em_data_cube)
        first_run = True
        for index in range(len(em_data_cube_data)):
            # EM Data Cube
            image_hdul = fits.open(em_data_cube_data[index])
            em_data_cube = image_hdul[0].data
            # print(np.shape(image_hdul[0].data))
            height, num_slits, width = np.shape(image_hdul[0].data)
            if first_run:
                em_data_cube_header = image_hdul[0].header
                run_em_data_cube = np.zeros(
                    (num_runs, height, num_slits, width), dtype=np.float32
                )
                first_run = False
            else:
                pass
            run_em_data_cube[index, :, :, :] = em_data_cube
        em_data_cube = np.mean(run_em_data_cube, axis=0)

    keywords_and_table_exists = True
    try:
        pixel_fov_width = image_hdul[0].header["PIXELFOV"]
        solution_fov_width = image_hdul[0].header["SLTNFOV"]
        slit_shift_width = int(round(solution_fov_width / pixel_fov_width))
        # print("solution fov = ", solution_fov_width, pixel_fov_width)

        dep_name = image_hdul[0].header["DEPNAME"]
        print("dep name =", dep_name)
        dep_indices = image_hdul[1].data["index"]
        dep_list = image_hdul[1].data[dep_name]
        if rsp_dep_list is not None:
            dep_mask = np.isin(dep_list, rsp_dep_list)
            new_dep_indices = []
            new_dep_list = []
            for index in range(len(dep_mask)):
                if dep_mask[index]:
                    new_dep_indices.append(dep_indices[index])
                    new_dep_list.append(dep_list[index])
            if len(new_dep_list) > 0:
                dep_indices = new_dep_indices
                dep_list = new_dep_list
    # except Exception as e:
    except:  # noqa: E722 # TODO figure out what exception was expected
        keywords_and_table_exists = False
        # print(repr(e))

    if keywords_and_table_exists:
        calc_half_slits = divmod(num_slits, 2)
        num_half_slits = int(calc_half_slits[0])
        for index in range(len(dep_list)):
            dep_rsp_file = rsp_dep_file_fmt.format(dep_list[index])
            dep_rsp_data = pd.read_csv(dep_rsp_file, delim_whitespace=True)
            dep_rsp = dep_rsp_data.iloc[:, 2].values
            if index == 0:
                dep_data_cube = np.zeros(
                    (num_rows, num_slits, len(dep_rsp)), dtype=np.float32
                )
            for slit_index in range(num_slits):
                slit_shift = (slit_index - num_half_slits) * slit_shift_width
                if slit_shift < 0:
                    slit_rsp = np.pad(dep_rsp, (0, -slit_shift), mode="constant")[
                        -slit_shift:
                    ]
                elif slit_shift > 0:
                    slit_rsp = np.pad(dep_rsp, (slit_shift, 0), mode="constant")[
                        :-slit_shift
                    ]
                else:
                    slit_rsp = dep_rsp
                for row in range(num_rows):
                    dep_data_cube[row, slit_index, :] = (
                        slit_rsp * em_data_cube[row, slit_index, dep_indices[index]]
                    )

            dep_data_cube_file = image_file_fmt.format(dep_list[index]) + ".fits"
            # print(dep_data_cube_file)
            fits_hdu = fits.PrimaryHDU(data=dep_data_cube, header=em_data_cube_header)
            fits_hdu.writeto(dep_data_cube_file, overwrite=True)
