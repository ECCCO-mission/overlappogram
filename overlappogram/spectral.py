from __future__ import annotations

import astropy.wcs as wcs
import numpy as np
from astropy.io import fits
from ndcube import NDCube

__all__ = [
    "create_spectrally_pure_images",
]


def create_spectrally_pure_images(image_list: list[NDCube],
                                  gnt_path: str,
                                  rsp_dep_list: list | None) -> NDCube:
    # from Dyana Beabout
    num_images = len(image_list)
    if num_images > 0:
        with fits.open(gnt_path) as gnt_hdul:
            ions = gnt_hdul[2].data
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
                for em_data in image_list:
                    em_data_cube = em_data.data.astype(np.float64)
                    em_data_cube = np.transpose(em_data_cube, axes=(1, 2, 0))
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

    out_wcs = wcs.WCS(naxis=2)
    out = NDCube(data=np.transpose(gnt_data_cube, (2, 0, 1)),
                 wcs=out_wcs,
                 meta={
                     "temperatures": image_list[0].meta["temperatures"],
                     "ions": ions})
    out.meta.update(image_list[0].meta)

    return out
