import concurrent.futures
import datetime
import os
import warnings
from enum import Enum
from threading import Lock

import numpy as np
from astropy.io import fits
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet


class InversionMode(Enum):
    ROW = 0
    CHUNKED = 1


class Inverter:
    def __init__(self,
                 rsp_func_cube_file: str,
                 rsp_dep_name: str,
                 rsp_dep_list: list,
                 solution_fov_width: int = 1,
                 smooth_over: str = 'spatial',
                 field_angle_range: list = None,
                 detector_row_range: list = None):

        self.rsp_func_cube_file = rsp_func_cube_file
        self.rsp_dep_name = rsp_dep_name
        self.rsp_dep_list = rsp_dep_list
        self.solution_fov_width = solution_fov_width
        self.smooth_over = smooth_over
        self.field_angle_range = field_angle_range
        self.detector_row_range = detector_row_range

        self._mode = InversionMode.CHUNKED
        self._models = []
        self.completed_row_count = 0
        self.unconverged_rows = []

        self.__post_init__()

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

        if self.detector_row_range is None:
            self.detector_row_range = [0, self.image_height - 1]

        self.completed_row_count = 0
        self.total_row_count = self.detector_row_range[1] - self.detector_row_range[0]

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

    def _invert_image_row(self, row_index, chunk_index):
        model = self._models[chunk_index]
        image_row = self.image[row_index, :]
        masked_rsp_func = self.response_function
        if self.image_mask is not None:
            mask_row = self.image_mask[row_index, :]
            mask_pixels = np.where(mask_row == 0)
            if len(mask_pixels) > 0:
                image_row[mask_pixels] = 0
                masked_rsp_func = self.response_function.copy()
                masked_rsp_func[mask_pixels, :] = 0.0
        if self.sample_weights is not None:
            sample_weights_row = self.sample_weights[row_index, :]
        else:
            sample_weights_row = None

        warnings.filterwarnings("error", category=ConvergenceWarning, module="sklearn")
        try:
            model.fit(masked_rsp_func, image_row, sample_weight=sample_weights_row)
            data_out = model.predict(masked_rsp_func)
            em = model.coef_
            score_data = model.score(masked_rsp_func, image_row)
        except ConvergenceWarning:
            self.unconverged_rows.append(row_index)
            em = np.zeros((self.num_slits * self.num_deps), dtype=np.float32)
            data_out = np.zeros(self.image_width, dtype=np.float32)
            score_data = -999

        return row_index, em, data_out, score_data

    def _progress_indicator(self, future):
        """used in multithreading to track progress of inversion"""
        with self.thread_count_lock:
            if not future.cancelled():
                self.completed_row_count += 1
                print(f"{self.completed_row_count/self.total_row_count*100:3.0f}% complete", end="\r")

    def _switch_to_row_inversion(self, model_config, alpha, rho, num_row_threads=50):
        self._mode = InversionMode.ROW

        remaining_rows = []
        for future, (row_index, chunk_index) in self.futures.items():
            if not future.done() and not future.running():
                future.cancel()
                remaining_rows.append(row_index)
        self.futures = {}

        for executor in self.executors:
            executor.shutdown()

        self.executors = [concurrent.futures.ThreadPoolExecutor(max_workers=num_row_threads)]

        self._models = []
        for i, row_index in enumerate(remaining_rows):
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
                warm_start=False
            )
            self._models.append(enet_model)
            future = self.executors[-1].submit(self._invert_image_row, row_index, i)
            future.add_done_callback(self._progress_indicator)
            self.futures[future] = (row_index, i)

    def _collect_results(self, mode_switch_thread_count, model_config, alpha, rho):
        for future in concurrent.futures.as_completed(self.futures):
            row_index, em, data_out, score_data = future.result()
            for slit_num in range(self.num_slits):
                if self.smooth_over == "dependence":
                    slit_em = em[slit_num * self.num_deps: (slit_num + 1) * self.num_deps]
                else:
                    slit_em = em[slit_num:: self.num_slits]
                self._em_cube[row_index, slit_num, :] = slit_em
            self._inversion_prediction[row_index, :] = data_out
            self._row_scores[row_index] = score_data

            rows_remaining = self.total_row_count - self.completed_row_count

            if rows_remaining < mode_switch_thread_count and self._mode == InversionMode.CHUNKED:
                self._switch_to_row_inversion(model_config, alpha, rho)
                break

    def _start_chunk_inversion(self, model_config, alpha, rho, num_threads):
        starts = np.arange(self.detector_row_range[0],
                           self.detector_row_range[1],
                           (self.detector_row_range[1] - self.detector_row_range[0]) / num_threads).astype(int)
        ends = np.append(starts[1:], self.detector_row_range[1])

        self.futures = {}
        self.executors = []
        for chunk_index, (start, end) in enumerate(zip(starts, ends)):
            self.executors.append(concurrent.futures.ThreadPoolExecutor(max_workers=1))
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
            self._models.append(enet_model)

            new_futures = {self.executors[-1].submit(self._invert_image_row, row, chunk_index): (row, chunk_index)
                           for row in range(start, end)}
            for future in new_futures:
                future.add_done_callback(self._progress_indicator)

            self.futures.update(new_futures)

    def invert(
        self,
        model_config,
        alpha,
        rho,
        num_threads: int = 1,
        mode_switch_thread_count: int = 0,
    ):
        self._models = []
        self.completed_row_count = 0

        self._em_cube = np.zeros((self.image_height, self.num_slits, self.num_deps), dtype=np.float32)
        self._inversion_prediction = np.zeros((self.image_height, self.image_width), dtype=np.float32)
        self._row_scores = np.zeros((self.image_height, 1), dtype=np.float32)

        self._start_chunk_inversion(model_config, alpha, rho, num_threads)

        # Collect results during the chunk stage
        self._collect_results(mode_switch_thread_count, model_config, alpha, rho)

        # Collect results during the row stage, if there is a mode change
        self._collect_results(mode_switch_thread_count, model_config, alpha, rho)

        for executor in self.executors:
            executor.shutdown()

        return self._em_cube, self._inversion_prediction, self._row_scores, self.unconverged_rows
