from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube

__all__ = ["load_overlappogram", "load_response_cube"]


def load_overlappogram(image_path, weights_path) -> NDCube:
    with fits.open(image_path) as image_hdul:
        image = image_hdul[0].data
        header = image_hdul[0].header
        wcs = WCS(image_hdul[0].header)
    with fits.open(weights_path) as weights_hdul:
        weights = weights_hdul[0].data
    return NDCube(image, wcs=wcs, uncertainty=StdDevUncertainty(1/weights), meta=dict(header))


def load_response_cube(path) -> NDCube:
    with fits.open(path) as hdul:
        response = hdul[0].data
        header = hdul[0].header
        wcs = WCS(hdul[0].header)
        temperatures = hdul[1].data
        field_angles = hdul[2].data
    meta = dict(header)
    meta.update({'temperatures': temperatures, 'field_angles': field_angles})
    return NDCube(response, wcs=wcs, meta=meta)

# # Create output directory.
# os.makedirs(output_dir, exist_ok=True)
#
# # Save EM data cube.
# base_filename = output_file_prefix
# if len(output_file_prefix) > 0 and output_file_prefix[-1] != "_":
#     base_filename += "_"
# base_filename += "em_data_cube"
# if len(output_file_postfix) > 0 and output_file_postfix[0] != "_":
#     base_filename += "_"
# base_filename += output_file_postfix
# em_data_cube_file = output_dir + base_filename + ".fits"
# # Transpose data (wavelength, y, x).  Readable by ImageJ.
# em_data_cube = np.transpose(self.mp_em_data_cube, axes=(2, 0, 1))
# em_data_cube_header = self.image_hdul[0].header.copy()
# em_data_cube_header["LEVEL"] = (level, "Level")
# em_data_cube_header["UNITS"] = ("1e26 cm-5", "Units")
# self.__add_fits_keywords(em_data_cube_header)
# em_data_cube_header['INVMDL'] = ('Elastic Net', 'Inversion Model')
# em_data_cube_header['ALPHA'] = (alpha, 'Inversion Model Alpha')
# em_data_cube_header['RHO'] = (rho, 'Inversion Model Rho')
# hdu = fits.PrimaryHDU(data=em_data_cube, header=em_data_cube_header)
# # Add binary table.
# col1 = fits.Column(name="index", format="1I", array=self.dep_index_list)
# col2 = fits.Column(
#     name=self.rsp_dep_name, format=self.rsp_dep_desc_fmt, array=self.dep_list
# )
# table_hdu = fits.BinTableHDU.from_columns([col1, col2])
# hdulist = fits.HDUList([hdu, table_hdu])
# hdulist.writeto(em_data_cube_file, overwrite=True)
#
# # Save model predicted data.
# base_filename = output_file_prefix
# if len(output_file_prefix) > 0 and output_file_prefix[-1] != "_":
#     base_filename += "_"
# base_filename += "model_predicted_data"
# if len(output_file_postfix) > 0 and output_file_postfix[0] != "_":
#     base_filename += "_"
# base_filename += output_file_postfix
# data_file = output_dir + base_filename + ".fits"
# model_predicted_data_hdul = self.image_hdul.copy()
# model_predicted_data_hdul[0].data = self.mp_inverted_data
# model_predicted_data_hdul[0].header["LEVEL"] = (level, "Level")
# model_predicted_data_hdul[0].header["UNITS"] = "Electron s-1"
# model_predicted_data_hdul[0].header['INVMDL'] = ('Elastic Net', 'Inversion Model')
# model_predicted_data_hdul[0].header['ALPHA'] = (alpha, 'Inversion Model Alpha')
# model_predicted_data_hdul[0].header['RHO'] = (rho, 'Inversion Model Rho')
# self.__add_fits_keywords(model_predicted_data_hdul[0].header)
# model_predicted_data_hdul.writeto(data_file, overwrite=True)
#
# if score:
#     # Save score.
#     base_filename = output_file_prefix
#     if len(output_file_prefix) > 0 and output_file_prefix[-1] != "_":
#         base_filename += "_"
#     base_filename += "model_score_data"
#     if len(output_file_postfix) > 0 and output_file_postfix[0] != "_":
#         base_filename += "_"
#     base_filename += output_file_postfix
#     score_data_file = output_dir + base_filename + ".fits"
#     # print("score", data_file)
#     hdu = fits.PrimaryHDU(data=self.mp_score_data)
#     hdulist = fits.HDUList([hdu])
#     hdulist.writeto(score_data_file, overwrite=True)
#
# return em_data_cube_file
#
# def __add_fits_keywords(self, header):
#     """
#     Add FITS keywords to FITS header.
#
#     Parameters
#     ----------
#     header : class 'astropy.io.fits.hdu.image.PrimaryHDU'.
#         FITS header.
#
#     Returns
#     -------
#     None.
#
#     """
#     header["INV_DATE"] = (self.inv_date, "Inversion Date")
#     header["RSPFUNC"] = (self.rsp_func_date, "Response Functions Filename")
#     header["RSP_DATE"] = (
#         self.rsp_func_cube_filename,
#         "Response Functions Creation Date",
#     )
#     header["ABUNDANC"] = (self.abundance, "Abundance")
#     header["ELECDIST"] = (self.electron_distribution, "Electron Distribution")
#     header["CHIANT_V"] = (self.chianti_version, "Chianti Version")
#     header["INVIMG"] = (self.input_image, "Inversion Image Filename")
#     header["INVMASK"] = (self.image_mask_filename, "Inversion Mask Filename")
#     header["SLTNFOV"] = (self.solution_fov_width, "Solution FOV Width")
#     header["DEPNAME"] = (self.rsp_dep_name, "Dependence Name")
#     header["SMTHOVER"] = (self.smooth_over, "Smooth Over")
#     header["LOGT_MIN"] = (f"{self.dep_list[0]:.2f}", "Minimum Logt")
#     header["LOGT_DLT"] = (f"{self.max_dep_list_delta:.2f}", "Delta Logt")
#     header["LOGT_NUM"] = (len(self.dep_list), "Number Logts")
#     header["FA_MIN"] = (
#         f"{self.field_angle_range_list[0]:.3f}",
#         "Minimum Field Angle",
#     )
#     header["FA_DLT"] = (
#         f"{self.max_field_angle_list_delta:.3f}",
#         "Delta Field Angle",
#     )
#     header["FA_NUM"] = (self.num_field_angles, "Number Field Angles")
#     header["FA_CDELT"] = (
#         f"{self.solution_fov_width * self.max_field_angle_list_delta:.3f}",
#         "Field Angle CDELT",
#     )
#     header["DROW_MIN"] = (self.detector_row_min, "Minimum Detector Row")
#     header["DROW_MAX"] = (self.detector_row_max, "Maximum Detector Row")
