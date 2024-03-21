import os

import numpy as np
from astropy.io import fits

__all__ = [
    "create_spectrally_pure_images",
]


def create_spectrally_pure_images(image_list: list, gnt_file: str, rsp_dep_list: list, output_dir: str):
    """
    Creates Level 2.x spectrally pure image from EM data cubes.

    Parameters
    ----------
    image_list : list
        List of Level 2.x filenames.
    gnt_file : str
       GNT data file.
    rsp_dep_list: list
        List of dependence items.  If None, use all dependence values.
    output_dir : str
       Directory to write Level 2.x spectrally pure images.

    Returns
    -------
    None.

    """
    # Create output directory.
    os.makedirs(output_dir, exist_ok=True)
    num_images = len(image_list)
    if num_images > 0:
        with fits.open(gnt_file) as gnt_hdul:
            gnt_filename = os.path.basename(gnt_file)
            gnt_data_values = gnt_hdul[0].data.astype(np.float64)
            num_gnts, num_gnt_deps = np.shape(gnt_data_values)
            gnt_dep_list = gnt_hdul[1].data["logt"]
            try:
                ion_wavelength_table_format = gnt_hdul[0].header["IWTBLFMT"]
                if ion_wavelength_table_format == "ion@wavelength":
                    ion_wavelength_name = "ion_wavelength"
                else:
                    ion_wavelength_name = ion_wavelength_table_format

                ion_wavelength_values = gnt_hdul[2].data[ion_wavelength_name]
            except KeyError:
                ion_wavelength_values = []
            assert len(ion_wavelength_values) == num_gnts
            if rsp_dep_list is None:
                rsp_dep_list = gnt_dep_list
                num_rsp_deps = len(rsp_dep_list)
                gnt_values = gnt_data_values
            else:
                num_rsp_deps = len(rsp_dep_list)
                gnt_values = np.zeros((num_gnts, num_rsp_deps), dtype=np.float64)
                dep_cnt = 0
                for dep in rsp_dep_list:
                    try:
                        index = np.where(gnt_dep_list == np.around(dep, decimals=2))
                    except:  # noqa: E722 # TODO figure out what exception was expected
                        pass
                    if len(index[0] == 1):
                        gnt_values[:, dep_cnt] = gnt_data_values[:, index].ravel()
                        dep_cnt += 1
                assert dep_cnt == num_rsp_deps

            for index in range(len(image_list)):
                # Create spectrally pure data cube.
                with fits.open(image_list[index]) as em_hdul:
                    em_data_cube = em_hdul[0].data.astype(np.float64)
                    em_data_cube = np.transpose(em_data_cube, axes=(1, 2, 0))
                    # em_dep_list = em_hdul[1].data['logt']
                    # print(em_dep_list)
                    if index == 0:
                        image_height, num_slits, num_logts = np.shape(em_data_cube)
                        gnt_data_cube = np.zeros((image_height, num_slits, num_gnts), dtype=np.float64)
                    else:
                        gnt_data_cube = np.transpose(gnt_data_cube.astype(np.float32), axes=(1, 2, 0))
                        gnt_data_cube[:, :, :] = 0.0
                    for gnt_num in range(num_gnts):
                        gnt_image = (
                            em_data_cube[:, :, 0:num_rsp_deps] * 10**26 * gnt_values[gnt_num, 0:num_rsp_deps]
                        ).sum(axis=2)
                        gnt_data_cube[:, :, gnt_num] = gnt_image
                    basename = os.path.splitext(os.path.basename(image_list[index]))[0]
                    # print(type(basename))
                    slice_index = basename.find("_em_data_cube")
                    # print(type(basename))
                    postfix_val = basename.split("_x")
                    postfix_val = postfix_val[1]
                    # print(postfix_val)
                    basename = basename[:slice_index]
                    basename += "_spectrally_pure_data_cube_x" + postfix_val + ".fits"
                    gnt_data_cube_file = output_dir + basename
                    # Transpose data (wavelength, y, x).  Readable by ImageJ.
                    gnt_data_cube = np.transpose(gnt_data_cube.astype(np.float32), axes=(2, 0, 1))
                    em_hdul[0].data = gnt_data_cube
                    em_hdul[0].header["UNITS"] = "Ph s-1 sr-1 cm-2"
                    em_hdul[0].header["GNT"] = (gnt_filename, "GNT Filename")
                    em_hdul[0].header["DEPNAME"] = ("wavelength", "Dependence Name")
                    em_hdul[0].header["IWTBLFMT"] = (
                        ion_wavelength_table_format,
                        "Ion/Wavelength Table Format",
                    )
                    # Add binary table.
                    gnt_index_list = list(range(len(ion_wavelength_values)))
                    col1 = fits.Column(name="index", format="1I", array=gnt_index_list)
                    col2 = fits.Column(
                        name=ion_wavelength_name,
                        format="15A",
                        array=ion_wavelength_values,
                    )
                    table_hdu = fits.BinTableHDU.from_columns([col1, col2])
                    em_hdul[1].data = table_hdu.data
                    em_hdul[1].header = table_hdu.header
                    em_hdul.writeto(gnt_data_cube_file, overwrite=True)
