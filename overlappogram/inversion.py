import concurrent.futures
import datetime
import os
import typing as tp
import warnings
from dataclasses import dataclass
from threading import Lock

import numpy as np
from astropy.io import fits
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet


@dataclass(order=True)
class Inversion:
    """
    Inversion for overlap-a-gram data.

    Attributes
    ----------
    rsp_func_cube_file: str
        Filename of response function cube.
    rsp_dep_name: str
        Response dependence name (e.g. 'ion' or 'logt').
    rsp_dep_list: list
        List of dependence items.  If None, use all dependence values.
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
    rsp_dep_name: str
    rsp_dep_list: list = None
    solution_fov_width: np.int32 = 1
    smooth_over: str = "spatial"
    field_angle_range: list = None

    def __post_init__(self):
        self.thread_count_lock = Lock()
        self.image_height = 0
        self.image_width = 0

        # Open response function cube file.
        rsp_func_hdul = fits.open(self.rsp_func_cube_file)
        rsp_func_cube = rsp_func_hdul[0].data
        num_dep, num_field_angles, rsp_func_width = np.shape(rsp_func_cube)
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

        dep_name = rsp_func_hdul[0].header["DEPNAME"]
        dep_list = rsp_func_hdul[1].data[dep_name]
        dep_list = np.round(dep_list, decimals=2)
        self.pixels = rsp_func_hdul[2].data["index"]
        self.field_angle_list = rsp_func_hdul[2].data["field_angle"]
        self.field_angle_list = np.round(self.field_angle_list, decimals=2)
        self.field_angle_index_list = rsp_func_hdul[2].data["index"]
        if self.rsp_dep_list is None:
            self.dep_index_list = rsp_func_hdul[1].data["index"]
            self.dep_list = dep_list
            dep_list_deltas = abs(np.diff(dep_list))
            self.max_dep_list_delta = max(dep_list_deltas)
        else:
            dep_list_deltas = abs(np.diff(dep_list))
            self.max_dep_list_delta = max(dep_list_deltas)
            dep_index_list = []
            for dep in self.rsp_dep_list:
                delta_dep_list = abs(dep_list - dep)
                dep_index = np.argmin(delta_dep_list)
                if abs(dep_list[dep_index] - dep) < self.max_dep_list_delta:
                    dep_index_list = np.append(dep_index_list, dep_index)
            new_index_list = [*set(dep_index_list)]
            new_index_list = np.array(new_index_list, dtype=np.int32)
            new_index_list.sort()
            self.dep_index_list = new_index_list
            self.dep_list = dep_list[new_index_list]

        self.num_deps = len(self.dep_list)
        self.rsp_func_width = rsp_func_width

        field_angle_list_deltas = abs(np.diff(self.field_angle_list))
        self.max_field_angle_list_delta = max(field_angle_list_deltas)
        if self.field_angle_range is None:
            begin_slit_index = np.int64(0)
            end_slit_index = np.int64(len(self.field_angle_list) - 1)
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
                    angle_index_list = np.append(angle_index_list, angle_index)
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
        self.half_slits = divmod(self.num_slits, 2)

        self.half_fov = divmod(self.solution_fov_width, 2)

        self.center_slit = (
            divmod(end_slit_index - begin_slit_index, 2) + begin_slit_index
        )

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

        num_field_angles = (end_slit_index - begin_slit_index) + 1
        self.field_angle_range_index_list = [begin_slit_index, end_slit_index]
        self.field_angle_range_list = self.field_angle_list[
            self.field_angle_range_index_list
        ]
        self.num_field_angles = num_field_angles

        response_count = 0
        self.response_function = np.zeros(
            (self.num_deps * self.num_slits, self.rsp_func_width), dtype=np.float32
        )
        for index in self.dep_index_list:
            if self.smooth_over == "dependence":
                # Smooth over dependence.
                slit_count = 0
                for slit_num in range(
                    self.center_slit[0]
                    - (self.half_slits[0] * self.solution_fov_width),
                    self.center_slit[0]
                    + ((self.half_slits[0] * self.solution_fov_width) + 1),
                    self.solution_fov_width,
                ):
                    # for slit_num in range(begin_slit_index, (end_slit_index + 1), self.solution_fov_width):
                    if self.solution_fov_width == 1:
                        self.response_function[
                            (self.num_deps * slit_count) + response_count, :
                        ] = rsp_func_cube[index, slit_num, :]
                    else:
                        # Check if even FOV.
                        if self.half_fov[1] == 0:
                            self.response_function[
                                (self.num_deps * slit_count) + response_count, :
                            ] = (
                                rsp_func_cube[
                                    index,
                                    slit_num
                                    - (self.half_fov[0] - 1) : slit_num
                                    + (self.half_fov[0] - 1)
                                    + 1,
                                    :,
                                ].sum(axis=0)
                                + (
                                    rsp_func_cube[index, slit_num - self.half_fov[0], :]
                                    * 0.5
                                )
                                + (
                                    rsp_func_cube[index, slit_num + self.half_fov[0], :]
                                    * 0.5
                                )
                            )
                        else:
                            self.response_function[
                                (self.num_deps * slit_count) + response_count, :
                            ] = rsp_func_cube[
                                index,
                                slit_num
                                - self.half_fov[0] : slit_num
                                + self.half_fov[0]
                                + 1,
                                :,
                            ].sum(
                                axis=0
                            )
                    slit_count += 1
                response_count += 1
            else:
                self.smooth_over = "spatial"
                # Smooth over spatial.
                for slit_num in range(
                    int(
                        self.center_slit[0]
                        - (self.half_slits[0] * self.solution_fov_width)
                    ),
                    int(
                        self.center_slit[0]
                        + ((self.half_slits[0] * self.solution_fov_width) + 1)
                    ),
                    int(self.solution_fov_width),
                ):
                    # for slit_num in range(begin_slit_index, (end_slit_index + 1), self.solution_fov_width):
                    # print(slit_num)
                    if self.solution_fov_width == 1:
                        self.response_function[response_count, :] = rsp_func_cube[
                            index, slit_num, :
                        ]
                    else:
                        # Check if even FOV.
                        if self.half_fov[1] == 0:
                            self.response_function[response_count, :] = (
                                rsp_func_cube[
                                    index,
                                    slit_num
                                    - (self.half_fov[0] - 1) : slit_num
                                    + (self.half_fov[1] - 1)
                                    + 1,
                                    :,
                                ].sum(axis=0)
                                + (
                                    rsp_func_cube[index, slit_num - self.half_fov[0], :]
                                    * 0.5
                                )
                                + (
                                    rsp_func_cube[index, slit_num + self.half_fov[0], :]
                                    * 0.5
                                )
                            )
                        else:
                            self.response_function[response_count, :] = rsp_func_cube[
                                index,
                                slit_num
                                - self.half_fov[0] : slit_num
                                + self.half_fov[0]
                                + 1,
                                :,
                            ].sum(axis=0)
                    response_count += 1

        self.response_function = self.response_function.transpose()

        if self.rsp_dep_name == "logt":
            self.rsp_dep_desc_fmt = "1E"
        else:
            max_dep_len = len(max(self.rsp_dep_list, key=len))
            self.rsp_dep_desc_fmt = str(max_dep_len) + "A"

    def get_response_function(self):
        return self.response_function

    def initialize_input_data(
        self, input_image: str, image_mask: str = None, sample_weights_data: str = None
    ):
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
        self.image = image

        try:
            image_exposure_time = image_hdul[0].header["IMG_EXP"]
        except KeyError:
            image_exposure_time = 1.0
        self.image /= image_exposure_time
        self.image[np.where(self.image < 0.0)] = 0.0

        self.image_hdul = image_hdul
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

        if sample_weights_data is not None:
            sample_weights_hdul = fits.open(sample_weights_data)
            sample_weights_height, sample_weights_width = np.shape(
                sample_weights_hdul[0].data
            )
            self.sample_weights = sample_weights_hdul[0].data
        else:
            self.sample_weights = None

    def _invert_image_row(
        self, image_row_number: np.int32, chunk_index: int, score=False
    ):
        model = self.models[chunk_index]
        image_row = self.image[image_row_number, :]
        masked_rsp_func = self.response_function
        if self.image_mask is not None:
            mask_row = self.image_mask[image_row_number, :]
            mask_pixels = np.where(mask_row == 0)
            if len(mask_pixels) > 0:
                image_row[mask_pixels] = 0
                masked_rsp_func = self.response_function.copy()
                masked_rsp_func[mask_pixels, :] = 0.0
        if self.sample_weights is not None:
            sample_weights_row = self.sample_weights[image_row_number, :]
        else:
            sample_weights_row = None
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "error", category=ConvergenceWarning, module="sklearn"
            )
            try:
                model.fit(masked_rsp_func, image_row, sample_weight=sample_weights_row)
                data_out = model.predict(masked_rsp_func)
                em = model.coef_
            except Exception:
                print("Row", image_row_number, "did not converge!")
                em = np.zeros((self.num_slits * self.num_deps), dtype=np.float32)
                data_out = np.zeros((self.image_width), dtype=np.float32)

        if score:
            score_data = model.score(masked_rsp_func, image_row)
            return [image_row_number, em, data_out, score_data]
        else:  # noqa: RET505
            return [image_row_number, em, data_out]

    def _progress_indicator(self, future):
        """used in multithreading to track progress of inversion"""
        with self.thread_count_lock:
            self.completed_row_count += 1
            print(f"{self.completed_row_count/self.total_row_count*100:3.0f}% complete", end="\r")


    def invert(
        self,
        model_config,
        alpha,
        rho,
        output_dir: str,
        output_file_prefix: str = "",
        output_file_postfix: str = "",
        level: str = "2.0",
        num_threads: int = 1,
        mode_switch_thread_count: int = 5,
        detector_row_range: tp.Union[list, None] = None,
        score=False,
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
        score: bool, optional
            Obtain scoring from model and write to file.  If None, ignore scoring.  The default is None.

        Returns
        -------
        None.

        """

        # Verify input data has been initialized.
        # assert self.image_width != 0 and self.image_height != 0
        self.mp_em_data_cube = np.zeros(
            (self.image_height, self.num_slits, self.num_deps), dtype=np.float32
        )
        self.mp_inverted_data = np.zeros(
            (self.image_height, self.image_width), dtype=np.float32
        )
        if score:
            self.mp_score_data = np.zeros((self.image_height, 1), dtype=np.float32)

        if detector_row_range is not None:
            self.detector_row_min = detector_row_range[0]
            self.detector_row_max = detector_row_range[1]
        else:
            self.detector_row_min = 0
            self.detector_row_max = self.image_height - 1

        self.completed_row_count = 0
        self.total_row_count = self.detector_row_max - self.detector_row_min

        starts = np.arange(
            self.detector_row_min,
            self.detector_row_max,
            (self.detector_row_max - self.detector_row_min) / num_threads,
        ).astype(int)
        ends = np.append(starts[1:], self.detector_row_max)

        futures = []
        executors = []
        self.models = []
        for chunk_index, (start, end) in enumerate(zip(starts, ends)):
            executors.append(concurrent.futures.ThreadPoolExecutor(max_workers=1))
            enet_model = ElasticNet(
                alpha=alpha,
                l1_ratio=rho,
                tol=model_config["tol"],
                max_iter=model_config["max_iter"],
                precompute=False,  # setting this to true slows down performance dramatically
                positive=True,
                copy_X=False,
                fit_intercept=False,
                selection=model_config["selection"],
                warm_start=model_config["warm_start"],
            )
            self.models.append(enet_model)

            new_futures = [executors[-1].submit(self._invert_image_row, row, chunk_index, score)
                           for row in range(start, end)]
            for future in new_futures:
                future.add_done_callback(self._progress_indicator)

            futures.extend(new_futures)

        # Wait for all tasks to complete and retrieve the results
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            for slit_num in range(self.num_slits):
                if self.smooth_over == "dependence":
                    slit_em = result[1][
                        slit_num * self.num_deps : (slit_num + 1) * self.num_deps
                    ]
                else:
                    slit_em = result[1][slit_num :: self.num_slits]
                self.mp_em_data_cube[result[0], slit_num, :] = slit_em

            self.mp_inverted_data[result[0], :] = result[2]
            if score:
                self.mp_score_data[result[0]] = result[3]

            with self.thread_count_lock:
                rows_remaining = self.total_row_count - self.completed_row_count
                if rows_remaining < mode_switch_thread_count:
                    print("switch mode")

        for executor in executors:
            executor.shutdown()

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
        em_data_cube = np.transpose(self.mp_em_data_cube, axes=(2, 0, 1))
        em_data_cube_header = self.image_hdul[0].header.copy()
        em_data_cube_header["LEVEL"] = (level, "Level")
        em_data_cube_header["UNITS"] = ("1e26 cm-5", "Units")
        self.__add_fits_keywords(em_data_cube_header)
        em_data_cube_header['INVMDL'] = ('Elastic Net', 'Inversion Model')
        em_data_cube_header['ALPHA'] = (alpha, 'Inversion Model Alpha')
        em_data_cube_header['RHO'] = (rho, 'Inversion Model Rho')
        hdu = fits.PrimaryHDU(data=em_data_cube, header=em_data_cube_header)
        # Add binary table.
        col1 = fits.Column(name="index", format="1I", array=self.dep_index_list)
        col2 = fits.Column(
            name=self.rsp_dep_name, format=self.rsp_dep_desc_fmt, array=self.dep_list
        )
        table_hdu = fits.BinTableHDU.from_columns([col1, col2])
        hdulist = fits.HDUList([hdu, table_hdu])
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
        model_predicted_data_hdul[0].data = self.mp_inverted_data
        model_predicted_data_hdul[0].header["LEVEL"] = (level, "Level")
        model_predicted_data_hdul[0].header["UNITS"] = "Electron s-1"
        model_predicted_data_hdul[0].header['INVMDL'] = ('Elastic Net', 'Inversion Model')
        model_predicted_data_hdul[0].header['ALPHA'] = (alpha, 'Inversion Model Alpha')
        model_predicted_data_hdul[0].header['RHO'] = (rho, 'Inversion Model Rho')
        self.__add_fits_keywords(model_predicted_data_hdul[0].header)
        model_predicted_data_hdul.writeto(data_file, overwrite=True)

        if score:
            # Save score.
            base_filename = output_file_prefix
            if len(output_file_prefix) > 0 and output_file_prefix[-1] != "_":
                base_filename += "_"
            base_filename += "model_score_data"
            if len(output_file_postfix) > 0 and output_file_postfix[0] != "_":
                base_filename += "_"
            base_filename += output_file_postfix
            score_data_file = output_dir + base_filename + ".fits"
            # print("score", data_file)
            hdu = fits.PrimaryHDU(data=self.mp_score_data)
            hdulist = fits.HDUList([hdu])
            hdulist.writeto(score_data_file, overwrite=True)

        return em_data_cube_file

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
        header["DEPNAME"] = (self.rsp_dep_name, "Dependence Name")
        header["SMTHOVER"] = (self.smooth_over, "Smooth Over")
        header["LOGT_MIN"] = (f"{self.dep_list[0]:.2f}", "Minimum Logt")
        header["LOGT_DLT"] = (f"{self.max_dep_list_delta:.2f}", "Delta Logt")
        header["LOGT_NUM"] = (len(self.dep_list), "Number Logts")
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
