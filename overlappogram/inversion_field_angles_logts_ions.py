import datetime
import os
import typing as tp
import warnings
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
from astropy.io import fits
from sklearn.exceptions import ConvergenceWarning


@dataclass(order=True)
class Inversion:
    """
    Inversion for overlap-a-gram data.

    Attributes
    ----------
    rsp_func_cube_file: str
        Filename of response function cube.
    rsp_func_selection: tuble
        Response function selection. ((ions), (logt_deltas), (logt_mins), (logt_maxs))
    solution_fov_width: np.int32
        Solution field-of-view width.  1 (all field angles), 2 (every other one), etc.  The default is 1.
    smooth_over: str, optional
        Inversion smoothing (i.e. 'spatial' or 'dependence').  The default is 'spatial'.
    field_angle_range: list, optional
        Beginning and ending field angles to invert over.

    Returns
    -------
    None.

    """

    rsp_func_cube_file: str
    rsp_func_selection: tuple
    solution_fov_width: np.int32 = 1
    smooth_over: str = "spatial"
    field_angle_range: list = None

    def __post_init__(self):
        self.image_height = 0
        self.image_width = 0

        # Open response function cube file.
        rsp_func_hdul = fits.open(self.rsp_func_cube_file)
        rsp_func_cube = rsp_func_hdul[0].data
        self.num_ions, self.num_logts, num_field_angles, rsp_func_width = np.shape(
            rsp_func_cube
        )
        # print(self.num_ions, self.num_logts, num_field_angles, rsp_func_width)
        self.rsp_func_cube_filename = os.path.basename(self.rsp_func_cube_file)

        self.inv_date = (
            datetime.datetime.now()
            .isoformat(timespec="milliseconds")
            .replace("+00:00", "Z")
        )

        try:
            self.rsp_func_date = rsp_func_hdul[0].header["DATE"]
        except KeyError:
            self.rsp_func_date = ""
        try:
            self.abundance = rsp_func_hdul[0].header["ABUNDANC"]
        except KeyError:
            self.abundance = ""
        try:
            self.electron_distribution = rsp_func_hdul[0].header["ELECDIST"]
        except KeyError:
            self.electron_distribution = ""
        try:
            self.chianti_version = rsp_func_hdul[0].header["CHIANT_V"]
        except KeyError:
            self.chianti_version = ""
        # Field Angles (field_angle)
        self.field_angle_list = rsp_func_hdul[1].data["field_angle"]
        # self.field_angle_list = rsp_func_hdul[2].data['field_angle']
        self.field_angle_list = np.round(self.field_angle_list, decimals=2)
        # Logt (logt)
        logt_list = rsp_func_hdul[2].data["logt"]
        # logt_list = rsp_func_hdul[1].data['logt']
        logt_list = np.round(logt_list, decimals=2)
        print(logt_list)
        # Ion (ion)
        print(rsp_func_hdul[3].header)
        print(rsp_func_hdul[3].data)
        ion_list = rsp_func_hdul[3].data["ion"]
        # ion_list = rsp_func_hdul[4].data['ION_PRESSURE'] #density
        # ion_list = rsp_func_hdul[3].data['ionlist']
        # ion_list = np.round(ion_list, decimals=2)
        print(ion_list)

        logt_ion_table = np.zeros((len(logt_list), len(ion_list)), dtype=np.uint8)
        for index in range(len(self.rsp_func_selection[0])):
            (ion_index,) = np.where(ion_list == self.rsp_func_selection[0][index])
            print(self.rsp_func_selection[0][index])
            assert len(ion_index == 1)
            logts = np.arange(
                self.rsp_func_selection[2][index],
                self.rsp_func_selection[3][index] + self.rsp_func_selection[1][index],
                self.rsp_func_selection[1][index],
            )
            logts = np.round(logts, decimals=2)
            for logt in logts:
                (logt_index,) = np.where(np.isclose(logt_list, logt))
                assert len(logt_index == 1)
                logt_ion_table[logt_index, ion_index[0]] = 1

        print(logt_ion_table)
        inv_selection = np.where(logt_ion_table == 1)
        print(inv_selection, type(inv_selection))
        self.num_selections = len(inv_selection[0])
        self.inverted_selection = deepcopy(list(inv_selection))
        # print(self.inverted_selection, type(self.inverted_selection))
        inv_logt_index_list = [*set(inv_selection[0])]
        inv_logt_index_list = np.array(inv_logt_index_list, dtype=np.int32)
        inv_logt_index_list.sort()
        self.inv_logt_list = logt_list[inv_logt_index_list]
        print(self.inv_logt_list)
        for count, value in enumerate(inv_logt_index_list):
            # print(count, value)
            self.inverted_selection[0][np.where(inv_selection[0] == value)] = count
        inv_ion_index_list = [*set(inv_selection[1])]
        inv_ion_index_list = np.array(inv_ion_index_list, dtype=np.int32)
        inv_ion_index_list.sort()
        self.inv_ion_list = ion_list[inv_ion_index_list]
        print(self.inv_ion_list)
        for count, value in enumerate(inv_ion_index_list):
            # print(count, value)
            self.inverted_selection[1][np.where(inv_selection[1] == value)] = count
        # print(inv_selection, type(inv_selection))
        print(self.inverted_selection, type(self.inverted_selection))

        self.rsp_func_width = rsp_func_width

        field_angle_list_deltas = abs(np.diff(self.field_angle_list))
        self.max_field_angle_list_delta = max(field_angle_list_deltas)
        # print(self.max_field_angle_list_delta)
        if self.field_angle_range is None:
            begin_slit_index = np.int64(0)
            end_slit_index = np.int64(len(self.field_angle_list) - 1)
            print("begin index", begin_slit_index, ", end index", end_slit_index)
            self.field_angle_range_index_list = [begin_slit_index, end_slit_index]
            self.field_angle_range_list = self.field_angle_list[
                self.field_angle_range_index_list
            ]
        else:
            assert len(self.field_angle_range) == 2
            angle_index_list = []
            for angle in self.field_angle_range:
                delta_angle_list = abs(self.field_angle_list - angle)
                angle_index = np.argmin(delta_angle_list)
                if (
                    abs(self.field_angle_list[angle_index] - angle)
                    < self.max_field_angle_list_delta
                ):
                    # print(angle, angle_index, self.field_angle_list[angle_index])
                    angle_index_list = np.append(angle_index_list, angle_index)
            print(angle_index_list)
            new_index_list = [*set(angle_index_list)]
            new_index_list = np.array(new_index_list, dtype=np.int32)
            new_index_list.sort()
            self.field_angle_range_index_list = new_index_list
            assert len(self.field_angle_range_index_list) == 2
            self.field_angle_range_list = self.field_angle_list[new_index_list]
            begin_slit_index = self.field_angle_range_index_list[0]
            end_slit_index = self.field_angle_range_index_list[1]
            num_field_angles = (end_slit_index - begin_slit_index) + 1

        # Check if number of field angles is even.
        calc_half_fields_angles = divmod(num_field_angles, 2)
        if calc_half_fields_angles[1] == 0.0:
            end_slit_index = end_slit_index - 1
            self.field_angle_range_index_list[1] = end_slit_index
            self.field_angle_range_list[1] = self.field_angle_list[end_slit_index]
            num_field_angles = (end_slit_index - begin_slit_index) + 1

        calc_num_slits = divmod(num_field_angles, self.solution_fov_width)
        self.num_slits = int(calc_num_slits[0])
        # Check if number of slits is even.
        calc_half_num_slits = divmod(self.num_slits, 2)
        if calc_half_num_slits[1] == 0.0:
            self.num_slits -= 1
        # self.num_slits = num_field_angles * self.solution_fov_width
        assert self.num_slits >= 3
        # print("number slits =", self.num_slits)
        # self.center_slit = divmod(num_field_angles, 2)
        self.half_slits = divmod(self.num_slits, 2)
        # if self.half_slits[0] * self.solution_fov_width > self.center_slit[0]:
        #     self.num_slits = self.num_slits - 2
        #     self.half_slits = divmod(self.num_slits, 2)

        self.half_fov = divmod(self.solution_fov_width, 2)
        # assert self.half_fov[1] == 1

        # print("old center slit", self.center_slit)
        # self.center_slit = self.center_slit + begin_slit_index
        self.center_slit = (
            divmod(end_slit_index - begin_slit_index, 2) + begin_slit_index
        )
        print("center slit", self.center_slit, self.num_slits, self.half_slits)

        # Check if even FOV.
        # if self.half_fov[1] == 0:
        #     begin_slit_index = self.center_slit[0] - (self.half_fov[0] - 1)
        #     - (self.half_slits[0] * self.solution_fov_width)
        # else:
        #     begin_slit_index = self.center_slit[0] - self.half_fov[0] - (self.half_slits[0] * self.solution_fov_width)
        begin_slit_index = (
            self.center_slit[0]
            - self.half_fov[0]
            - (self.half_slits[0] * self.solution_fov_width)
        )
        end_slit_index = (
            self.center_slit[0]
            + self.half_fov[0]
            + (self.half_slits[0] * self.solution_fov_width)
        )
        # assert begin_slit_index >= 0 and end_slit_index <= (max_num_field_angles - 1)
        print(
            "begin_slit_index =", begin_slit_index, "end_slit_index =", end_slit_index
        )
        # print(self.center_slit, (self.half_slits[0], self.solution_fov_width))
        # begin_slit_index = self.center_slit - (self.half_slits[0] * self.solution_fov_width)
        # end_slit_
        index = self.center_slit + (self.half_slits[0] * self.solution_fov_width)
        # print(begin_slit_index, end_slit_index)
        num_field_angles = (end_slit_index - begin_slit_index) + 1
        self.field_angle_range_index_list = [begin_slit_index, end_slit_index]
        self.field_angle_range_list = self.field_angle_list[
            self.field_angle_range_index_list
        ]
        self.num_field_angles = num_field_angles

        response_count = 0
        self.response_function = np.zeros(
            (self.num_selections * self.num_slits, self.rsp_func_width),
            dtype=np.float32,
        )
        if self.smooth_over == "dependence":
            # Smooth over dependence.
            # for slit_num in range(self.num_slits):
            for slit_num in range(
                self.center_slit[0] - (self.half_slits[0] * self.solution_fov_width),
                self.center_slit[0]
                + ((self.half_slits[0] * self.solution_fov_width) + 1),
                self.solution_fov_width,
            ):
                for index in range(self.num_selections):
                    if self.solution_fov_width == 1:
                        self.response_function[response_count, :] = rsp_func_cube[
                            inv_selection[1][index],
                            inv_selection[0][index],
                            slit_num,
                            :,
                        ]
                    else:
                        # Check if even FOV.
                        if self.half_fov[1] == 0:
                            self.response_function[response_count, :] = (
                                rsp_func_cube[
                                    inv_selection[1][index],
                                    inv_selection[0][index],
                                    slit_num
                                    - (self.half_fov[0] - 1) : slit_num
                                    + (self.half_fov[0] - 1)
                                    + 1,
                                    :,
                                ].sum(axis=0)
                                + (
                                    rsp_func_cube[
                                        inv_selection[1][index],
                                        inv_selection[0][index],
                                        slit_num - self.half_fov[0],
                                        :,
                                    ]
                                    * 0.5
                                )
                                + (
                                    rsp_func_cube[
                                        inv_selection[1][index],
                                        inv_selection[0][index],
                                        slit_num + self.half_fov[0],
                                        :,
                                    ]
                                    * 0.5
                                )
                            )
                        else:
                            self.response_function[response_count, :] = rsp_func_cube[
                                inv_selection[1][index],
                                inv_selection[0][index],
                                slit_num
                                - self.half_fov[0] : slit_num
                                + self.half_fov[0]
                                + 1,
                                :,
                            ].sum(axis=0)
                    response_count += 1
        else:
            self.smooth_over = "spatial"
            # Smooth over spatial.
            for index in range(self.num_selections):
                # for slit_num in range(self.num_slits):
                for slit_num in range(
                    self.center_slit[0]
                    - (self.half_slits[0] * self.solution_fov_width),
                    self.center_slit[0]
                    + ((self.half_slits[0] * self.solution_fov_width) + 1),
                    self.solution_fov_width,
                ):
                    if self.solution_fov_width == 1:
                        self.response_function[response_count, :] = rsp_func_cube[
                            inv_selection[1][index],
                            inv_selection[0][index],
                            slit_num,
                            :,
                        ]
                    else:
                        # Check if even FOV.
                        if self.half_fov[1] == 0:
                            self.response_function[response_count, :] = (
                                rsp_func_cube[
                                    inv_selection[1][index],
                                    inv_selection[0][index],
                                    slit_num
                                    - (self.half_fov[0] - 1) : slit_num
                                    + (self.half_fov[1] - 1)
                                    + 1,
                                    :,
                                ].sum(axis=0)
                                + (
                                    rsp_func_cube[
                                        inv_selection[1][index],
                                        inv_selection[0][index],
                                        slit_num - self.half_fov[0],
                                        :,
                                    ]
                                    * 0.5
                                )
                                + (
                                    rsp_func_cube[
                                        inv_selection[1][index],
                                        inv_selection[0][index],
                                        slit_num + self.half_fov[0],
                                        :,
                                    ]
                                    * 0.5
                                )
                            )
                        else:
                            self.response_function[response_count, :] = rsp_func_cube[
                                inv_selection[1][index],
                                inv_selection[0][index],
                                slit_num
                                - self.half_fov[0] : slit_num
                                + self.half_fov[0]
                                + 1,
                                :,
                            ].sum(axis=0)
                    response_count += 1

        # print("response count =", response_count)
        self.response_function = self.response_function.transpose()
        print(np.shape(self.response_function))

    def get_response_function(self):
        return self.response_function

    def initialize_input_data(self, input_image: str, image_mask: str = None):
        """
        Initialize input image and optional mask.

        Parameters
        ----------
        input_image : str
            Input image to invert.
        image_mask : str, optional
            Image mask where pixel values of 0 are ignored.  Mask is the same size as image. The default is None.

        Returns
        -------
        None.

        """
        # Read image
        image_hdul = fits.open(input_image)
        image = image_hdul[0].data
        image_height, image_width = np.shape(image)
        print(image_height, image_width)
        # Verify image width equals the response function width in cube.
        assert image_width == self.rsp_func_width
        self.image = image

        try:
            image_exposure_time = image_hdul[0].header["IMG_EXP"]
        except KeyError:
            image_exposure_time = 1.0
        self.image /= image_exposure_time
        self.image[np.where(self.image < 0.0)] = 0.0

        self.image_hdul = image_hdul
        # print("image (h, w) =", image_height, image_width)
        self.image_width = image_width
        self.image_height = image_height
        self.input_image = os.path.basename(input_image)

        if image_mask is not None:
            # Read mask
            mask_hdul = fits.open(image_mask)
            mask_height, mask_width = np.shape(mask_hdul[0].data)
            self.image_mask = mask_hdul[0].data
            if len(np.where(self.image_mask == 0)) == 0:
                self.image_mask = None
        else:
            # self.image_mask = np.ones((image_height, image_width), dtype=np.float32)
            self.image_mask = None
        if self.image_mask is not None:
            self.image_mask_filename = os.path.basename(image_mask)
        else:
            self.image_mask_filename = ""

    def invert(
        self,
        model,
        output_dir: str,
        output_file_prefix: str = "",
        output_file_postfix: str = "",
        level: str = "2.0",
        detector_row_range: tp.Union[list, None] = None,
    ):
        """
        Invert image.

        Parameters
        ----------
        model : Class derived from AbstractModel.
            Inversion model.
        output_dir : str
            Directory to write out EM data cube and inverted data image.
        output_file_prefix : str, optional
            A string prefixed to the output base filenames. The default is ''.
        output_file_postfix : str, optional
            A string postfixed to the output base filenames. The default is ''.
        level: str, optional
            Level value for FITS keyword LEVEL.
        detector_row_range: list, optional
            Beginning and ending row numbers to invert.  If None, invert all rows.  The default is None.

        Returns
        -------
        None.

        """
        # Verify input data has been initialized.
        # assert self.image_width != 0 and self.image_height != 0
        if detector_row_range is not None:
            # assert len(detector_row_range) == 2
            # assert detector_row_range[1] >= detector_row_range[0]
            # assert detector_row_range[0] < self.image_height and detector_row_range[1] < self.image_height
            self.detector_row_min = detector_row_range[0]
            self.detector_row_max = detector_row_range[1]
        else:
            self.detector_row_min = 0
            self.detector_row_max = self.image_height - 1
        em_data_cube = np.zeros(
            (
                self.image_height,
                self.num_slits,
                len(self.inv_logt_list),
                len(self.inv_ion_list),
            ),
            dtype=np.float32,
        )
        em_data_cube[:, :, :, :] = -1.0
        inverted_data = np.zeros(
            (self.image_height, self.image_width), dtype=np.float32
        )
        num_nonconvergences = 0
        if detector_row_range is None:
            image_row_number_range = range(self.image_height)
        else:
            image_row_number_range = range(
                detector_row_range[0], detector_row_range[1] + 1
            )

        scorelist = []
        for image_row_number in image_row_number_range:
            # if (image_row_number % 10 == 0):
            if image_row_number % 100 == 0:
                print("image row number =", image_row_number)
            # print(image_row_number)
            image_row = self.image[image_row_number, :]
            masked_rsp_func = self.response_function
            if self.image_mask is not None:
                mask_row = self.image_mask[image_row_number, :]
                mask_pixels = np.where(mask_row == 0)
                if len(mask_pixels) > 0:
                    image_row[mask_pixels] = 0.0
                    # image_row[mask_pixels] = 1e-26
                    masked_rsp_func = self.response_function.copy()
                    masked_rsp_func[mask_pixels, :] = 0.0
                    # masked_rsp_func[mask_pixels, :] = 1e-26
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "error", category=ConvergenceWarning, module="sklearn"
                )
                try:
                    em, data_out, score = model.invert(masked_rsp_func, image_row)
                    scorelist.append(score)
                    response_count = 0
                    if self.smooth_over == "dependence":
                        for slit_num in range(self.num_slits):
                            for index in range(self.num_selections):
                                em_data_cube[
                                    image_row_number,
                                    slit_num,
                                    self.inverted_selection[0][index],
                                    self.inverted_selection[1][index],
                                ] = em[response_count]
                                if image_row_number == 512:
                                    print(
                                        "*",
                                        image_row_number,
                                        slit_num,
                                        self.inverted_selection[0][index],
                                        self.inverted_selection[1][index],
                                        response_count,
                                    )
                                response_count += 1
                    else:
                        for index in range(self.num_selections):
                            for slit_num in range(self.num_slits):
                                em_data_cube[
                                    image_row_number,
                                    slit_num,
                                    self.inverted_selection[0][index],
                                    self.inverted_selection[1][index],
                                ] = em[response_count]
                                response_count += 1
                    inverted_data[image_row_number, :] = data_out
                    # print("Row", image_row_number, "converged.")
                except Exception as e:
                    num_nonconvergences += 1
                    print(e)
                    print("Row", image_row_number, "did not converge!")
                    response_count = 0
                    if self.smooth_over == "dependence":
                        for slit_num in range(self.num_slits):
                            for index in range(self.num_selections):
                                em_data_cube[
                                    image_row_number,
                                    slit_num,
                                    self.inverted_selection[0][index],
                                    self.inverted_selection[1][index],
                                ] = 0.0
                                response_count += 1
                    else:
                        for index in range(self.num_selections):
                            for slit_num in range(self.num_slits):
                                em_data_cube[
                                    image_row_number,
                                    slit_num,
                                    self.inverted_selection[0][index],
                                    self.inverted_selection[1][index],
                                ] = 0.0
                                response_count += 1

        print("Number Nonconvergences", num_nonconvergences)

        # Create output directory.
        os.makedirs(output_dir, exist_ok=True)

        # Save EM data cube.
        base_filename = output_file_prefix
        if len(output_file_prefix) > 0 and output_file_prefix[-1] != "_":
            base_filename += "_"
        base_filename += "em_data_cube"
        if len(output_file_postfix) > 0 and output_file_postfix[0] != "_":
            base_filename += "_"
        base_filename += output_file_postfix
        em_data_cube_file = output_dir + base_filename + ".fits"
        # Transpose data (wavelength, y, x).  Readable by ImageJ.
        em_data_cube = np.transpose(em_data_cube, axes=(3, 2, 0, 1))
        em_data_cube_header = self.image_hdul[0].header.copy()
        em_data_cube_header["LEVEL"] = (level, "Level")
        em_data_cube_header["UNITS"] = ("1e26 cm-5", "Units")
        self.__add_fits_keywords(em_data_cube_header)
        model.add_fits_keywords(em_data_cube_header)
        hdu = fits.PrimaryHDU(data=em_data_cube, header=em_data_cube_header)
        # Add binary table (logt).
        index_list = range(len(self.inv_logt_list))
        col1 = fits.Column(name="index", format="1I", array=index_list)
        col2 = fits.Column(name="logt", format="1E", array=self.inv_logt_list)
        logt_hdu = fits.BinTableHDU.from_columns([col1, col2])
        # Add binary table (ion).
        index_list = range(len(self.inv_ion_list))
        col1 = fits.Column(name="index", format="1I", array=index_list)
        col2 = fits.Column(name="ion", format="10A", array=self.inv_ion_list)
        ion_hdu = fits.BinTableHDU.from_columns([col1, col2])
        hdulist = fits.HDUList([hdu, logt_hdu, ion_hdu])
        hdulist.writeto(em_data_cube_file, overwrite=True)

        # Save model predicted data.
        base_filename = output_file_prefix
        if len(output_file_prefix) > 0 and output_file_prefix[-1] != "_":
            base_filename += "_"
        base_filename += "model_predicted_data"
        if len(output_file_postfix) > 0 and output_file_postfix[0] != "_":
            base_filename += "_"
        base_filename += output_file_postfix
        data_file = output_dir + base_filename + ".fits"
        model_predicted_data_hdul = self.image_hdul.copy()
        model_predicted_data_hdul[0].data = inverted_data
        model_predicted_data_hdul[0].header["LEVEL"] = (level, "Level")
        model_predicted_data_hdul[0].header["UNITS"] = "Electron s-1"
        self.__add_fits_keywords(model_predicted_data_hdul[0].header)
        model.add_fits_keywords(model_predicted_data_hdul[0].header)
        model_predicted_data_hdul.writeto(data_file, overwrite=True)

        # save scores
        f = open(output_dir + "/inversion_scores.txt", "w")
        for row in range(len(scorelist)):
            f.write(str(row) + "  " + str(scorelist[row]) + "\n ")
        f.close()

    # def multiprocessing_callback(self, result):
    #     # image_row_number = result[0]
    #     # em = result[1]
    #     # data_out = result[2]
    #     # for slit_num in range(self.num_slits):
    #     #     if self.smooth_over == 'dependence':
    #     #         slit_em = em[slit_num * self.num_deps:(slit_num + 1) * self.num_deps]
    #     #     else:
    #     #         slit_em = em[slit_num::self.num_slits]
    #     #     self.mp_em_data_cube[image_row_number, slit_num, :] = slit_em

    #     # self.mp_inverted_data[image_row_number, :] = data_out

    #     for slit_num in range(self.num_slits):
    #         if self.smooth_over == 'dependence':
    #             slit_em = result[1][slit_num * self.num_deps:(slit_num + 1) * self.num_deps]
    #         else:
    #             slit_em = result[1][slit_num::self.num_slits]
    #         self.mp_em_data_cube[result[0], slit_num, :] = slit_em

    #     self.mp_inverted_data[result[0], :] = result[2]

    # def multiprocessing_invert_image_row(self, image_row_number: np.int32, model):
    #     #print("Inverting image row", image_row_number)
    #     image_row = self.image[image_row_number,:]
    #     masked_rsp_func = self.response_function
    #     if self.image_mask is not None:
    #         mask_row = self.image_mask[image_row_number,:]
    #         mask_pixels = np.where(mask_row == 0)
    #         if len(mask_pixels) > 0:
    #             image_row[mask_pixels] = 0
    #             masked_rsp_func = self.response_function.copy()
    #             masked_rsp_func[mask_pixels, :] = 0.0
    #     # # If image has zero pixel values, zero out corresponding response function pixels.
    #     # zero_image_pixels = np.where(image_row == 0.0)
    #     # if len(zero_image_pixels) > 0:
    #     #     masked_rsp_func = masked_rsp_func.copy()
    #     #     masked_rsp_func[zero_image_pixels, :] = 0.0

    #     # masked_rsp_func2 = preprocessing.MinMaxScaler().fit_transform(masked_rsp_func)
    #     # em, data_out = model.invert(masked_rsp_func2, image_row)
    #     #model = deepcopy(self.mp_model)
    #     #em, data_out = model.invert(masked_rsp_func, image_row)
    #     em, data_out = model.invert(masked_rsp_func, image_row)

    #     return [image_row_number, em, data_out]

    # async def produce(self, queue):
    #     for image_row_number in range(1024):
    #         image_row = self.image[image_row_number,:]
    #         masked_rsp_func = self.response_function
    #         if self.image_mask is not None:
    #             mask_row = self.image_mask[image_row_number,:]
    #             mask_pixels = np.where(mask_row == 0)
    #             if len(mask_pixels) > 0:
    #                 image_row[mask_pixels] = 0
    #                 masked_rsp_func = self.response_function.copy()
    #                 masked_rsp_func[mask_pixels, :] = 0.0
    #         model = deepcopy(self.mp_model)

    #         # put the item in the queue
    #         await queue.put((image_row_number, masked_rsp_func, image_row, model))

    # async def consume(self, queue, answer, i):
    #     print(i)
    #     while True:
    #         await asyncio.sleep(0.001)
    #         # wait for an item from the producer
    #         item = await queue.get()

    #         # process the item
    #         em, data_out = item[3].invert(item[1], item[2])
    #         #print(i)

    #         # Write inversion to queue.
    #         await answer.put((item[0], em, data_out))

    #         # Notify the queue that the item has been processed
    #         queue.task_done()
    #         #await asyncio.sleep(0.001)

    # async def run_multiprocessing_inversion(self):
    #     queue = asyncio.Queue()
    #     await self.produce(queue)

    #     # schedule consumers
    #     consumers = []
    #     #for _ in range(os.cpu_count()):
    #     for i in range(os.cpu_count()):
    #         #print(i)
    #         consumer = asyncio.create_task(self.consume(queue, self.output_queue, i))
    #         consumers.append(consumer)

    #     # run the producer and wait for completion
    #     #await self.produce(queue)
    #     # wait until the consumer has processed all items
    #     await queue.join()

    #     # the consumers are still awaiting for an item, cancel them
    #     for consumer in consumers:
    #         consumer.cancel()

    #     # wait until all worker tasks are cancelled
    #     await asyncio.gather(*consumers, return_exceptions=True)

    # def multiprocessing_invert(self, model, output_dir: str,
    #                            output_file_prefix: str = '',
    #             output_file_postfix: str = '',
    #             level: str = '2.0',
    #             detector_row_range: tp.Union[list, None] = None):
    #     '''
    #     Invert image.

    #     Parameters
    #     ----------
    #     model : Class derived from AbstractModel.
    #         Inversion model.
    #     output_dir : str
    #         Directory to write out EM data cube and inverted data image.
    #     output_file_prefix : str, optional
    #         A string prefixed to the output base filenames. The default is ''.
    #     output_file_postfix : str, optional
    #         A string postfixed to the output base filenames. The default is ''.
    #     level: str, optional
    #         Level value for FITS keyword LEVEL.
    #     detector_row_range: list, optional
    #         Beginning and ending row numbers to invert.  If None, invert all rows.  The default is None.

    #     Returns
    #     -------
    #     None.

    #     '''
    #     # Verify input data has been initialized.
    #     assert self.image_width != 0 and self.image_height != 0
    #     self.mp_em_data_cube = np.zeros((self.image_height, self.num_slits, self.num_deps), dtype=np.float32)
    #     self.mp_inverted_data = np.zeros((self.image_height, self.image_width), dtype=np.float32)
    #     self.mp_model = model

    #     # #with mp.Pool(processes=4) as pool:
    #     # with mp.Pool(processes=os.cpu_count()) as pool:
    #     #     for i in range(self.image_height):
    #     #         pool.apply_async(self.multiprocessing_invert_image_row,
    #     args = (i, model), callback = self.multiprocessing_callback)
    #     #         #pool.apply_async(self.multiprocessing_invert_image_row,
    #     args = (i, ), callback = self.multiprocessing_callback)
    #     #     pool.close()
    #     #     pool.join()

    #     self.output_queue = asyncio.Queue()
    #     asyncio.run(self.run_multiprocessing_inversion())
    #     #await self.run_multiprocessing_inversion()

    #     while not self.output_queue.empty():
    #         result = self.output_queue.get_nowait()
    #         for slit_num in range(self.num_slits):
    #             if self.smooth_over == 'dependence':
    #                 slit_em = result[1][slit_num * self.num_deps:(slit_num + 1) * self.num_deps]
    #             else:
    #                 slit_em = result[1][slit_num::self.num_slits]
    #             self.mp_em_data_cube[result[0], slit_num, :] = slit_em

    #         self.mp_inverted_data[result[0], :] = result[2]

    #     # Create output directory.
    #     os.makedirs(output_dir, exist_ok=True)

    #     # Save EM data cube.
    #     base_filename = output_file_prefix
    #     if len(output_file_prefix) > 0 and output_file_prefix[-1] != '_':
    #         base_filename += '_'
    #     base_filename += 'em_data_cube'
    #     if len(output_file_postfix) > 0 and output_file_postfix[0] != '_':
    #         base_filename += '_'
    #     base_filename += output_file_postfix
    #     em_data_cube_file = output_dir + base_filename + '.fits'
    #     # Transpose data (wavelength, y, x).  Readable by ImageJ.
    #     em_data_cube = np.transpose(self.mp_em_data_cube, axes=(2, 0, 1))
    #     em_data_cube_header = self.image_hdul[0].header.copy()
    #     em_data_cube_header['LEVEL'] = (level, 'Level')
    #     em_data_cube_header['UNITS'] = ('1e26 cm-5', 'Units')
    #     self.__add_fits_keywords(em_data_cube_header)
    #     model.add_fits_keywords(em_data_cube_header)
    #     hdu = fits.PrimaryHDU(data = em_data_cube, header = em_data_cube_header)
    #     # Add binary table.
    #     col1 = fits.Column(name='index', format='1I', array=self.dep_index_list)
    #     col2 = fits.Column(name=self.rsp_dep_name, format=self.rsp_dep_desc_fmt, array=self.dep_list)
    #     table_hdu = fits.BinTableHDU.from_columns([col1, col2])
    #     hdulist = fits.HDUList([hdu, table_hdu])
    #     hdulist.writeto(em_data_cube_file, overwrite=True)

    #     # Save model predicted data.
    #     base_filename = output_file_prefix
    #     if len(output_file_prefix) > 0 and output_file_prefix[-1] != '_':
    #         base_filename += '_'
    #     base_filename += 'model_predicted_data'
    #     if len(output_file_postfix) > 0 and output_file_postfix[0] != '_':
    #         base_filename += '_'
    #     base_filename += output_file_postfix
    #     data_file = output_dir + base_filename + ".fits"
    #     model_predicted_data_hdul = self.image_hdul.copy()
    #     model_predicted_data_hdul[0].data = self.mp_inverted_data
    #     model_predicted_data_hdul[0].header['LEVEL'] = (level, 'Level')
    #     model_predicted_data_hdul[0].header['UNITS'] = 'Electron s-1'
    #     self.__add_fits_keywords(model_predicted_data_hdul[0].header)
    #     model.add_fits_keywords(model_predicted_data_hdul[0].header)
    #     model_predicted_data_hdul.writeto(data_file, overwrite=True)

    def __add_fits_keywords(self, header):
        """
        Add FITS keywords to FITS header.

        Parameters
        ----------
        header : class 'astropy.io.fits.hdu.image.PrimaryHDU'.
            FITS header.

        Returns
        -------
        None.

        """
        header["INV_DATE"] = (self.inv_date, "Inversion Date")
        header["RSPFUNC"] = (self.rsp_func_date, "Response Functions Filename")
        header["RSP_DATE"] = (
            self.rsp_func_cube_filename,
            "Response Functions Creation Date",
        )
        header["ABUNDANC"] = (self.abundance, "Abundance")
        header["ELECDIST"] = (self.electron_distribution, "Electron Distribution")
        header["CHIANT_V"] = (self.chianti_version, "Chianti Version")
        header["INVIMG"] = (self.input_image, "Inversion Image Filename")
        header["INVMASK"] = (self.image_mask_filename, "Inversion Mask Filename")
        header["SLTNFOV"] = (self.solution_fov_width, "Solution FOV Width")
        # header['DEPNAME'] = (self.rsp_dep_name, 'Dependence Name')
        header["SMTHOVER"] = (self.smooth_over, "Smooth Over")
        header["FA_MIN"] = (
            f"{self.field_angle_range_list[0]:.3f}",
            "Minimum Field Angle",
        )
        header["FA_DLT"] = (
            f"{self.max_field_angle_list_delta:.3f}",
            "Delta Field Angle",
        )
        header["FA_NUM"] = (self.num_field_angles, "Number Field Angles")
        header["FA_CDELT"] = (
            f"{self.solution_fov_width * self.max_field_angle_list_delta:.3f}",
            "Field Angle CDELT",
        )
        header["DROW_MIN"] = (self.detector_row_min, "Minimum Detector Row")
        header["DROW_MAX"] = (self.detector_row_max, "Maximum Detector Row")

    def create_forward_model(
        self, em_data_cube_file: str, output_dir: str, image_mask_file: str = None
    ):
        print(em_data_cube_file)
        assert self.num_selections >= 1
        # Read EM data cube
        em_data_cube_hdul = fits.open(em_data_cube_file)
        em_data_cube = em_data_cube_hdul[0].data
        em_data_cube[np.where(em_data_cube == -1.0)] = 0.0
        em_ions, em_logts, em_rows, em_slits = np.shape(em_data_cube)

        if image_mask_file is not None:
            # Read mask
            mask_hdul = fits.open(image_mask_file)
            mask_height, mask_width = np.shape(mask_hdul[0].data)
            image_mask = mask_hdul[0].data
            if len(np.where(image_mask == 0)) == 0:
                image_mask = None
        else:
            image_mask = None

        # forward_model_image = np.zeros((self.image_height, self.image_width), dtype=np.float32)
        forward_model_image = np.zeros((em_rows, self.rsp_func_width), dtype=np.float32)
        if image_mask is None:
            response_count = 0
            if self.smooth_over == "dependence":
                # Smooth over dependence.
                for slit_num in range(self.num_slits):
                    for index in range(self.num_selections):
                        forward_model_image += np.dot(
                            em_data_cube[
                                self.inverted_selection[1][index],
                                self.inverted_selection[0][index],
                                :,
                                slit_num,
                            ][:, None],
                            self.response_function[:, response_count][None, :],
                        )
                        response_count += 1
            else:
                self.smooth_over = "spatial"
                # Smooth over spatial.
                for index in range(self.num_selections):
                    for slit_num in range(self.num_slits):
                        forward_model_image += np.dot(
                            em_data_cube[
                                self.inverted_selection[1][index],
                                self.inverted_selection[0][index],
                                :,
                                slit_num,
                            ][:, None],
                            self.response_function[:, response_count][None, :],
                        )
                        response_count += 1
        else:
            # for image_row_number in range(self.image_height):
            for image_row_number in range(499, 551):  # em_rows):
                # print("1", image_row_number)
                mask_row = self.image_mask[image_row_number, :]
                mask_pixels = np.where(mask_row == 0)
                masked_rsp_func = self.response_function.copy()
                masked_rsp_func[mask_pixels, :] = 0.0
                response_count = 0
                if self.smooth_over == "dependence":
                    # Smooth over dependence.
                    for slit_num in range(self.num_slits):
                        for index in range(self.num_selections):
                            forward_model_image[image_row_number, :] += np.dot(
                                em_data_cube[
                                    self.inverted_selection[1][index],
                                    self.inverted_selection[0][index],
                                    image_row_number,
                                    slit_num,
                                ],
                                masked_rsp_func[:, response_count],
                            )
                            response_count += 1
                else:
                    self.smooth_over = "spatial"
                    # Smooth over spatial.
                    for index in range(self.num_selections):
                        for slit_num in range(self.num_slits):
                            forward_model_image[image_row_number, :] += np.dot(
                                em_data_cube[
                                    self.inverted_selection[1][index],
                                    self.inverted_selection[0][index],
                                    image_row_number,
                                    slit_num,
                                ],
                                masked_rsp_func[:, response_count],
                            )
                            response_count += 1

        result = em_data_cube_file.find("em_data_cube")
        if result == -1:
            forward_model_file = output_dir + "forward_model.fits"
        else:
            forward_model_file = em_data_cube_file[0:result] + "forward_model.fits"
        em_data_cube_hdul[0].data = forward_model_image
        em_data_cube_hdul.writeto(forward_model_file, overwrite=True)
