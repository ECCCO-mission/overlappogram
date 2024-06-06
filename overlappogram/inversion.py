from __future__ import annotations

import concurrent.futures
import warnings
from enum import Enum
from threading import Lock

import astropy.wcs as wcs
import numpy as np
from ndcube import NDCube
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import ElasticNet
from tqdm import tqdm

from overlappogram.error import InvalidInversionModeError, NoWeightsWarnings
from overlappogram.response import prepare_response_function

__all__ = ["Inverter"]


class InversionMode(Enum):
    ROW = 0
    CHUNKED = 1
    HYBRID = 2


MODE_MAPPING = {"row": InversionMode.ROW, "chunked": InversionMode.CHUNKED, "hybrid": InversionMode.HYBRID}


class Inverter:
    def __init__(
        self,
        response_cube: NDCube,
        solution_fov_width: int = 1,
        smooth_over: str = "dependence",
        response_dependency_list: list = None,
        field_angle_range: list = None,
        detector_row_range: list = None,
    ):
        self._solution_fov_width = solution_fov_width
        self._smooth_over = smooth_over
        self._field_angle_range = field_angle_range
        self._detector_row_range = detector_row_range

        self._mode = InversionMode.CHUNKED
        self._models = []
        self._completed_row_count = 0
        self._unconverged_rows = []

        self._overlappogram: NDCube | None = None
        self._em_data: np.ndarray | None = None
        self._inversion_prediction: np.ndarray | None = None
        self._row_scores: np.ndarray | None = None
        self._overlappogram_width: int | None = None
        self._overlappogram_height: int | None = None

        self._thread_count_lock = Lock()

        self._response_function, self._num_slits, self._num_deps = prepare_response_function(
            response_cube,
            fov_width=solution_fov_width,
            field_angle_range=field_angle_range,
            response_dependency_list=response_dependency_list,
        )
        self._response_meta = response_cube.meta

        if response_dependency_list is not None:
            self._response_meta['temperatures'] = np.array([(i, t) for i, t in enumerate(response_dependency_list)],
                                                           dtype=[('index', '>i2'), ('logt', '>f4')])

        self._progress_bar = None  # initialized in invert call

    @property
    def is_inverted(self) -> bool:
        return not any(
            [
                self._overlappogram is None,
                self._em_data is None,
                self._inversion_prediction is None,
                self._row_scores is None,
            ]
        )

    def _invert_image_row(self, row_index, chunk_index):
        model = self._models[chunk_index]
        image_row = self._overlappogram.data[row_index, :]
        masked_response_function = self._response_function.copy()

        if self._overlappogram.mask is not None:
            mask_row = self._overlappogram.mask[row_index, :]
            mask_pixels = np.where(mask_row == 0)
            if len(mask_pixels) > 0:
                image_row[mask_pixels] = 0
                masked_response_function[mask_pixels, :] = 0.0

        if self._overlappogram.uncertainty is not None:
            sample_weights_row = 1.0 / self._overlappogram.uncertainty[row_index, :].array
        else:
            sample_weights_row = None

        warnings.filterwarnings("error", category=ConvergenceWarning, module="sklearn")
        try:
            model.fit(masked_response_function, image_row, sample_weight=sample_weights_row)
            data_out = model.predict(masked_response_function)
            em = model.coef_
            score_data = model.score(masked_response_function, image_row)
        except ConvergenceWarning:
            self._unconverged_rows.append(row_index)
            em = np.zeros((self._num_slits * self._num_deps), dtype=np.float32)
            data_out = np.zeros(self._overlappogram_width, dtype=np.float32)
            score_data = -999

        return row_index, em, data_out, score_data

    def _progress_indicator(self, future):
        """used in multithreading to track progress of inversion"""
        with self._thread_count_lock:
            if not future.cancelled():
                self._completed_row_count += 1
                self._progress_bar.update(1)

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
                warm_start=False,
            )
            self._models.append(enet_model)
            future = self.executors[-1].submit(self._invert_image_row, row_index, i)
            future.add_done_callback(self._progress_indicator)
            self.futures[future] = (row_index, i)

    def _collect_results(self, mode_switch_thread_count, model_config, alpha, rho):
        for future in concurrent.futures.as_completed(self.futures):
            row_index, em, data_out, score_data = future.result()
            for slit_num in range(self._num_slits):
                if self._smooth_over == "dependence":
                    slit_em = em[slit_num * self._num_deps : (slit_num + 1) * self._num_deps]
                else:
                    slit_em = em[slit_num :: self._num_slits]
                self._em_data[row_index, slit_num, :] = slit_em
            self._inversion_prediction[row_index, :] = data_out
            self._row_scores[row_index] = score_data

            rows_remaining = self.total_row_count - self._completed_row_count

            if rows_remaining < mode_switch_thread_count and self._mode == InversionMode.HYBRID:
                self._switch_to_row_inversion(model_config, alpha, rho)
                break

    def _start_row_inversion(self, model_config, alpha, rho, num_threads):
        self.executors = [concurrent.futures.ThreadPoolExecutor(max_workers=num_threads)]

        self.futures = {}
        self._models = []
        for i, row_index in enumerate(range(self._detector_row_range[0], self._detector_row_range[1])):
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
                warm_start=False,  # warm start doesn't make sense in the row version
            )
            self._models.append(enet_model)
            future = self.executors[-1].submit(self._invert_image_row, row_index, i)
            future.add_done_callback(self._progress_indicator)
            self.futures[future] = (row_index, i)

    def _start_chunk_inversion(self, model_config, alpha, rho, num_threads):
        starts = np.arange(
            self._detector_row_range[0],
            self._detector_row_range[1],
            (self._detector_row_range[1] - self._detector_row_range[0]) / num_threads,
        ).astype(int)
        ends = np.append(starts[1:], self._detector_row_range[1])

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

            new_futures = {
                self.executors[-1].submit(self._invert_image_row, row, chunk_index): (row, chunk_index)
                for row in range(start, end)
            }
            for future in new_futures:
                future.add_done_callback(self._progress_indicator)

            self.futures.update(new_futures)

    def _initialize_with_overlappogram(self, overlappogram):
        self._overlappogram = overlappogram

        if self._overlappogram.uncertainty is None:
            warnings.warn("Running in weightless mode since no weights array was provided.", NoWeightsWarnings)

        if self._detector_row_range is None:
            self._detector_row_range = (0, overlappogram.data.shape[0])
        self.total_row_count = self._detector_row_range[1] - self._detector_row_range[0]

        # correct for exposure time
        try:
            image_exposure_time = self._overlappogram.meta["IMG_EXP"]
        except KeyError:
            image_exposure_time = 1.0
        self._overlappogram /= image_exposure_time

        self._overlappogram.data[np.where(self._overlappogram.data < 0.0)] = 0.0

        # initialize all results cubes
        self._overlappogram_height, self._overlappogram_width = self._overlappogram.data.shape
        self._em_data = np.zeros((self._overlappogram_height, self._num_slits, self._num_deps), dtype=np.float32)
        self._inversion_prediction = np.zeros((self._overlappogram_height, self._overlappogram_width), dtype=np.float32)
        self._row_scores = np.zeros((self._overlappogram_height, 1), dtype=np.float32)

    def invert(
        self,
        overlappogram: NDCube,
        model_config,
        alpha,
        rho,
        num_threads: int = 1,
        mode_switch_thread_count: int = 0,
        mode: InversionMode = InversionMode.HYBRID,
    ) -> (NDCube, NDCube, np.ndarray, list[int]):
        self._initialize_with_overlappogram(overlappogram)

        self._mode = mode

        self._progress_bar = tqdm(total=self.total_row_count, unit='row', delay=1, leave=False)

        self._models = []
        self._completed_row_count = 0

        if self._mode == InversionMode.HYBRID:
            self._start_chunk_inversion(model_config, alpha, rho, num_threads)
            # Collect results during the chunk stage
            self._collect_results(mode_switch_thread_count, model_config, alpha, rho)
            # Collect results during the row stage, if there is a mode change
            self._collect_results(mode_switch_thread_count, model_config, alpha, rho)
        elif self._mode == InversionMode.CHUNKED:
            self._start_chunk_inversion(model_config, alpha, rho, num_threads)
            # mode never switches since count=0
            self._collect_results(0, model_config, alpha, rho)
        elif self._mode == InversionMode.ROW:
            self._start_row_inversion(model_config, alpha, rho, num_threads)
            self._collect_results(np.inf, model_config, alpha, rho)
        else:
            raise InvalidInversionModeError("Invalid InversionMode.")

        for executor in self.executors:
            executor.shutdown()

        self._progress_bar.close()

        out_wcs = wcs.WCS(naxis=2)

        return (
            NDCube(data=np.transpose(self._em_data, (2, 0, 1)), wcs=out_wcs, meta=self._response_meta),
            NDCube(data=self._inversion_prediction, wcs=out_wcs, meta=self._response_meta),
            self._row_scores,
            self._unconverged_rows,
        )
