import glob
import math
import os
import os.path
import typing as tp
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import astropy.wcs
import dateutil
import numpy as np
import numpy.ma as ma
import pandas as pd
from astropy.io import fits
from astropy.table import Table
from PIL import Image
from scipy.io import loadmat

# Launch T0 (seconds since beginning of year) and timestamp.
launch_t0 = 18296410.2713640
launch_timestamp = "2021-07-30T18:22:21Z"

# Pixel deltas (y, x).
pixel_8_deltas = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
pixel_16_deltas = [
    (-2, -2),
    (-2, -1),
    (-2, 0),
    (-2, 1),
    (-2, 2),
    (-1, -2),
    (-1, 2),
    (0, -2),
    (0, 2),
    (1, -2),
    (1, 2),
    (2, -2),
    (2, -1),
    (2, 0),
    (2, 1),
    (2, 2),
]
pixel_16_box_deltas = [
    (-2, -2),
    (-2, -1),
    (-2, 0),
    (-2, 1),
    (-2, 2),
    (-1, -2),
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (-1, 2),
    (0, -2),
    (0, -1),
    (0, 1),
    (0, 2),
    (1, -2),
    (1, -1),
    (1, 0),
    (1, 1),
    (1, 2),
    (2, -2),
    (2, -1),
    (2, 0),
    (2, 1),
    (2, 2),
]
pixel_16_adjacent_deltas = [
    (-2, -2),
    (-2, -1),
    (-2, 0),
    (-2, 1),
    (-2, 2),
    (-1, 2),
    (0, 2),
    (1, 2),
    (2, 2),
    (2, 1),
    (2, 0),
    (2, -1),
    (2, -2),
    (1, -2),
    (0, -2),
    (-1, -2),
]
pixel_8_adjacent_deltas = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (-1, 0),
]
pixel_16_adjacent_wrap_deltas = [
    (-1, -2),
    (-2, -2),
    (-2, -1),
    (-2, 0),
    (-2, 1),
    (-2, 2),
    (-1, 2),
    (0, 2),
    (1, 2),
    (2, 2),
    (2, 1),
    (2, 0),
    (2, -1),
    (2, -2),
    (1, -2),
    (0, -2),
    (-1, -2),
]


def calc_analog_cal(raw_values):
    return (raw_values / 1024.0) * 5.0


def calc_cold_block_cal(values):
    return (52.37 * calc_analog_cal(values)) - 157.55


def calc_ccd_holder_cal(values):
    return (
        (0.5513 * calc_analog_cal(values) ** 3)
        + (1.3523 * calc_analog_cal(values) ** 2)
        + (29.942 * calc_analog_cal(values))
        - 116.85
    )


@dataclass(order=True)
class MaGIXSDataProducts:
    """
    Inversion for overlap-a-gram data.

    Attributes
    ----------

    Returns
    -------
    None.

    """

    def __post_init__(self):
        self.active_pixels_height: np.uit32 = 1024
        self.active_pixels_width: np.uint32 = 2048
        self.pixel_fov_width = 2.8
        self.ll_gain = 2.6
        self.lr_gain = 2.6
        self.ul_gain = 2.6
        self.ur_gain = 2.6
        self.ll_bias = 0.0
        self.lr_bias = 0.0
        self.ul_bias = 0.0
        self.ur_bias = 0.0

    def remove_bias(self, image: np.ndarray) -> np.ndarray:
        """
        Blank pixels are used to calculate the bias and remove bias from image.
        This algorithm using the 35 inner most columns of the blank pixels and
        the rows of the blank pixels that contains the rows of the active image
        pixels.

        Parameters
        ----------
        image : np.ndarray
            Image with bias.

        Returns
        -------
        image : np.ndarray
            Image with bias removed.

        """
        height, width = np.shape(image)

        quad_height = height // 2
        quad_width = width // 2

        # Calculate bias for each quadrant.
        self.ll_bias = np.mean(image[8:quad_height, 15:50])
        self.lr_bias = np.mean(image[8:quad_height, width - 50 : width - 15])
        self.ul_bias = np.mean(image[quad_height : height - 8, 15:50])
        self.ur_bias = np.mean(image[quad_height : height - 8, width - 50 : width - 15])

        # Subtract bias for each quadrant.
        image[8:quad_height, 50:quad_width] -= self.ll_bias
        image[8:quad_height, quad_width : width - 50] -= self.lr_bias
        image[quad_height : height - 8, 50:quad_width] -= self.ul_bias
        image[quad_height : height - 8, quad_width : width - 50] -= self.ur_bias

        return image

    def remove_bias_by_row(self, image: np.ndarray) -> np.ndarray:
        """
        Blank pixels are used to calculate the bias and remove bias from image.
        This algorithm using the 35 inner most columns of the blank pixels and
        the rows of the blank pixels that contains the rows of the active image
        pixels.

        Parameters
        ----------
        image : np.ndarray
            Image with bias.

        Returns
        -------
        image : np.ndarray
            Image with bias removed.

        """
        height, width = np.shape(image)

        quad_height = height // 2
        quad_width = width // 2

        # Calculate bias for each quadrant.
        self.ll_bias = np.mean(image[8:quad_height, 15:50])
        self.lr_bias = np.mean(image[8:quad_height, width - 50 : width - 15])
        self.ul_bias = np.mean(image[quad_height : height - 8, 15:50])
        self.ur_bias = np.mean(image[quad_height : height - 8, width - 50 : width - 15])

        # Subtract bias for each quadrant.
        image[8:quad_height, 50:quad_width] -= np.mean(
            image[8:quad_height, 15:50], axis=1
        )[:, None]
        image[8:quad_height, quad_width : width - 50] -= np.mean(
            image[8:quad_height, width - 50 : width - 15], axis=1
        )[:, None]
        image[quad_height : height - 8, 50:quad_width] -= np.mean(
            image[quad_height : height - 8, 15:50], axis=1
        )[:, None]
        image[quad_height : height - 8, quad_width : width - 50] -= np.mean(
            image[quad_height : height - 8, width - 50 : width - 15], axis=1
        )[:, None]

        return image

    def remove_inactive_pixels(self, image: np.ndarray) -> np.ndarray:
        """
        Inactive pixels (i.e. blank and overscan pixels) are removed from the
        image.

        Parameters
        ----------
        image : np.ndarray
            Image with inactive pixels.

        Returns
        -------
        active_image : np.ndarray
            Image with inactive pixels removed.

        """
        height, width = np.shape(image)
        # print("rm inact h, w = ", height, width)

        # quad_height = height // 2
        active_height = height - (8 * 2)
        quad_width = width // 2
        active_width = width - ((50 + 2) * 2)
        active_half_width = active_width // 2

        active_image = np.zeros((active_height, active_width), dtype=image.dtype)

        # Left side.
        active_image[0:active_height, 0:active_half_width] = image[
            8 : active_height + 8, 50 : 50 + active_half_width
        ]

        # Right side.
        active_image[0:active_height, active_half_width:active_width] = image[
            8 : active_height + 8, quad_width + 2 : width - 50
        ]

        return active_image

    def adjust_gain(self, image: np.ndarray) -> np.ndarray:
        """
        Adjust the gain of the image.

        Parameters
        ----------
        image : np.ndarray
            Image that does not gain adjustment.

        Returns
        -------
        image : np.ndarray
            Image with gain adjusted.

        """
        height, width = np.shape(image)

        quad_height = height // 2
        quad_width = width // 2

        image[0:quad_height, 0:quad_width] *= self.ll_gain
        image[0:quad_height, quad_width:width] *= self.lr_gain
        image[quad_height:height, 0:quad_width] *= self.ul_gain
        image[quad_height:height, quad_width:width] *= self.ur_gain

        return image

    def replace_bad_pixels(
        self, image: np.ndarray, bad_pixel_mask: np.ndarray
    ) -> [np.ndarray, pd.DataFrame]:
        """
        Replace bad pixels in image.  The algorithm uses the median of the
        surrounding pixels to replace the bad pixel value.

        Parameters
        ----------
        image : np.ndarray
            Image to replace bad pixels.
        bad_pixel_mask : np.ndarray
            Mask indicating which pixels are bad.

        Returns
        -------
        image: np.ndarray
            Image with bad pixels replaced.
        original_values: pd.DataFrame
            Bad pixel replaced values (y, x, value).

        """
        height, width = np.shape(image)
        bad_pixel_list = np.where(bad_pixel_mask != 0)
        if len(bad_pixel_list[0]) > 0:
            # data = np.zeros((len(bad_pixel_list[0]),), dtype=[("y", "i4"), ("x", "i4"), ("value", "f4")])
            data = np.zeros(
                (len(bad_pixel_list[0]),),
                dtype=[("y", np.int32), ("x", np.int32), ("value", np.float32)],
            )
            original_values = pd.DataFrame(data)
            for c in range(len(bad_pixel_list[0])):
                adjacent_pixel_list = []
                for i in [-1, 0, 1]:
                    for j in [-1, 0, 1]:
                        # check if location is within boundary
                        if (
                            0 <= bad_pixel_list[0][c] + i < height
                            and 0 <= bad_pixel_list[1][c] + j < width
                        ):
                            adjacent_pixel_list.append(
                                image[
                                    bad_pixel_list[0][c] + i, bad_pixel_list[1][c] + j
                                ]
                            )
                median_value = np.median(adjacent_pixel_list)
                original_value = image[bad_pixel_list[0][c], bad_pixel_list[1][c]]
                image[bad_pixel_list[0][c], bad_pixel_list[1][c]] = median_value
                original_values.loc[c, ["y", "x", "value"]] = [
                    bad_pixel_list[0][c],
                    bad_pixel_list[1][c],
                    original_value,
                ]

            return image, original_values
        else:  # noqa: RET505
            return image, pd.DataFrame(columns=["y", "x", "value"])

    def create_bad_pixel_mask(
        self,
        image_list: list,
        pre_master_dark_file: str,
        post_master_dark_file: str,
        sigma: np.float32,
        percent_threshold: np.float32,
        bad_pixel_mask_file: str,
    ):
        """
        Creates a bad pixel mask.  This algorithm uses values from the entire image.
        The algorithm for determining bad pixels is any values above the mean - (sigma * standard deviation)
        For any values below the mean + (sigma * standard deviation where the percent threshold is met.

        Parameters
        ----------
        image_list : list
            Dark image filenames used to detect bad pixels of CCD sensor.
        pre_master_dark_file : str
            Pre-Master Dark file.
        post_master_dark_file : str
            Post-Master Dark file.
         sigma : np.float
            Sigma.
        percent_threshold : np.float
            Percentage threshold of marked pixels to be declared bad.
        bad_pixel_mask_file : str
            Filename for bad pixel mask.

        Returns
        -------
        None.

        """
        number_images = len(image_list)
        image_cube = np.zeros(
            (number_images, self.active_pixels_height, self.active_pixels_width),
            dtype=np.float32,
        )

        with fits.open(pre_master_dark_file) as dark_hdul:
            pre_master_dark = dark_hdul[0].data.astype(np.float32).copy()
            pre_dark_temp = dark_hdul[0].header["ADCTEMP6"]
        with fits.open(post_master_dark_file) as dark_hdul:
            post_master_dark = dark_hdul[0].data.astype(np.float32).copy()
            post_dark_temp = dark_hdul[0].header["ADCTEMP6"]

        img_seq_num_list = []
        first_image = True
        for index in range(number_images):
            with fits.open(image_list[index]) as image_hdul:
                if first_image is True:
                    camera_id = image_hdul[0].header["CAM_ID"]
                    camera_sn = image_hdul[0].header["CAM_SN"]
                    first_image = False
                img_seq_num_list.append(image_hdul[0].header["IMG_ISN"])
                (
                    image,
                    image_header,
                    bad_pixels_replaced_values,
                ) = self.create_adjusted_light(
                    image_hdul[0].data,
                    image_hdul[0].header,
                    pre_master_dark,
                    pre_dark_temp,
                    post_master_dark,
                    post_dark_temp,
                    None,
                )
                image_cube[index, :, :] = image

        # Despike.
        image_cube[
            np.where(
                image_cube
                > (
                    np.nanmedian(image_cube, axis=0, keepdims=True)
                    + (sigma * np.nanstd(image_cube, axis=0, keepdims=True))
                )
            )
        ] = np.nan
        # print("despike_len = ", np.count_nonzero(np.isnan(image_cube)))

        bad_pixel_mask = np.zeros(
            (self.active_pixels_height, self.active_pixels_width), dtype=np.float32
        )
        bad_pixels = np.zeros(
            (number_images, self.active_pixels_height, self.active_pixels_width),
            dtype=np.float32,
        )

        bad_pixels[np.where(np.isnan(image_cube))] = np.nan
        for index in range(number_images):
            bad_pixels[index, :, :][
                np.where(
                    image_cube[index, :, :]
                    < np.nanmean(image_cube[index, :, :], axis=(0, 1))
                    - (sigma * np.nanstd(image_cube[index, :, :], axis=(0, 1)))
                )
            ] = 1
            bad_pixels[index, :, :][
                np.where(
                    image_cube[index, :, :]
                    > np.nanmean(image_cube[index, :, :], axis=(0, 1))
                    + (sigma * np.nanstd(image_cube[index, :, :], axis=(0, 1)))
                )
            ] = 1

        # Set dark and hot pixels where number is equal to or greater than percent threshold
        bad_pixel_mask[
            np.where(np.nanmean(bad_pixels, axis=0) >= percent_threshold)
        ] = 1

        bad_pixel_mask_header = fits.Header()
        bad_pixel_mask_header["CAM_ID"] = (camera_id, "Camera ID")
        bad_pixel_mask_header["CAM_SN"] = (camera_sn, "Camera Serial Number")
        bad_pixel_mask_header["LEVEL"] = ("0.1", "Data Product Level")
        image_sequence_numbers = self.create_image_sequence_number_list(
            img_seq_num_list
        )
        bad_pixel_mask_header["IMGSEQNM"] = (
            image_sequence_numbers,
            "Image Sequence Numbers",
        )
        bad_pixel_mask_header["SIGMA"] = (sigma, "Sigma")
        bad_pixel_mask_header["PCNTTHLD"] = (percent_threshold, "Percent Threshold")
        bad_pixel_mask_header["PREMDARK"] = (
            os.path.basename(pre_master_dark_file),
            "Pre Master Dark",
        )
        bad_pixel_mask_header["PSTMDARK"] = (
            os.path.basename(post_master_dark_file),
            "Post Master Dark",
        )
        bad_pixel_mask_header["INTPTEMP"] = ("ADCTEMP6", "Interpolation Temperature")

        # Create output directory.
        os.makedirs(os.path.dirname(bad_pixel_mask_file), exist_ok=True)

        hdu = fits.PrimaryHDU(data=bad_pixel_mask, header=bad_pixel_mask_header)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(bad_pixel_mask_file, overwrite=True)

    def create_bad_pixel_mask_by_tap(
        self,
        image_list: list,
        sigma: np.float32,
        percent_threshold: np.float32,
        bad_pixel_mask_file: str,
    ):
        """
        Creates a bad pixel mask.  This algorithm uses values from the entire image.
        The algorithm for determining bad pixels is any values above the mean - (sigma * standard deviation)
        For any values below the median + (sigma * standard deviation where the percent threshold is met.

        Parameters
        ----------
        image_list : list
            Dark image filenames used to detect bad pixels of CCD sensor.
        pre_master_dark_file : str
            Pre-Master Dark file.
        post_master_dark_file : str
            Post-Master Dark file.
         sigma : np.float
            Sigma.
        percent_threshold : np.float
            Percentage threshold of marked pixels to be declared bad.
        bad_pixel_mask_file : str
            Filename for bad pixel mask.

        Returns
        -------
        None.

        """
        number_images = len(image_list)
        image_cube = np.zeros(
            (number_images, self.active_pixels_height, self.active_pixels_width),
            dtype=np.float32,
        )

        img_seq_num_list = []
        first_image = True
        for index in range(number_images):
            with fits.open(image_list[index]) as image_hdul:
                if first_image is True:
                    camera_id = image_hdul[0].header["CAM_ID"]
                    camera_sn = image_hdul[0].header["CAM_SN"]
                    first_image = False
                img_seq_num_list.append(image_hdul[0].header["IMG_ISN"])
                image = self.remove_bias(image_hdul[0].data.astype(np.float32))
                image = self.remove_inactive_pixels(image)
                image_cube[index, :, :] = image

        # Despike.
        image_cube[
            np.where(
                image_cube
                > (
                    np.nanmedian(image_cube, axis=0, keepdims=True)
                    + (sigma * np.nanstd(image_cube, axis=0, keepdims=True))
                )
            )
        ] = np.nan
        # print("despike_len = ", np.count_nonzero(np.isnan(image_cube)))

        bad_pixel_mask = np.zeros(
            (self.active_pixels_height, self.active_pixels_width), dtype=np.float32
        )
        bad_pixels = np.zeros(
            (number_images, self.active_pixels_height, self.active_pixels_width),
            dtype=np.float32,
        )

        quad_height = self.active_pixels_height // 2
        quad_width = self.active_pixels_width // 2

        bad_pixels[np.where(np.isnan(image_cube))] = np.nan
        for index in range(number_images):
            bad_pixels[index, 0:quad_height, 0:quad_width][
                np.where(
                    image_cube[index, 0:quad_height, 0:quad_width]
                    < np.nanmedian(image_cube[index, 0:quad_height, 0:quad_width])
                    - (
                        sigma
                        * np.nanstd(image_cube[index, 0:quad_height, 0:quad_width])
                    )
                )
            ] = 1
            bad_pixels[index, 0:quad_height, quad_width : self.active_pixels_width][
                np.where(
                    image_cube[
                        index, 0:quad_height, quad_width : self.active_pixels_width
                    ]
                    < np.nanmedian(
                        image_cube[
                            index, 0:quad_height, quad_width : self.active_pixels_width
                        ]
                    )
                    - (
                        sigma
                        * np.nanstd(
                            image_cube[
                                index,
                                0:quad_height,
                                quad_width : self.active_pixels_width,
                            ]
                        )
                    )
                )
            ] = 1
            bad_pixels[index, quad_height : self.active_pixels_height, 0:quad_width][
                np.where(
                    image_cube[
                        index, quad_height : self.active_pixels_height, 0:quad_width
                    ]
                    < np.nanmedian(
                        image_cube[
                            index, quad_height : self.active_pixels_height, 0:quad_width
                        ]
                    )
                    - (
                        sigma
                        * np.nanstd(
                            image_cube[
                                index,
                                quad_height : self.active_pixels_height,
                                0:quad_width,
                            ]
                        )
                    )
                )
            ] = 1
            bad_pixels[
                index,
                quad_height : self.active_pixels_height,
                quad_width : self.active_pixels_width,
            ][
                np.where(
                    image_cube[
                        index,
                        quad_height : self.active_pixels_height,
                        quad_width : self.active_pixels_width,
                    ]
                    < np.nanmedian(
                        image_cube[
                            index,
                            quad_height : self.active_pixels_height,
                            quad_width : self.active_pixels_width,
                        ]
                    )
                    - (
                        sigma
                        * np.nanstd(
                            image_cube[
                                index,
                                quad_height : self.active_pixels_height,
                                quad_width : self.active_pixels_width,
                            ]
                        )
                    )
                )
            ] = 1
            bad_pixels[index, 0:quad_height, 0:quad_width][
                np.where(
                    image_cube[index, 0:quad_height, 0:quad_width]
                    > np.nanmedian(image_cube[index, 0:quad_height, 0:quad_width])
                    + (
                        sigma
                        * np.nanstd(image_cube[index, 0:quad_height, 0:quad_width])
                    )
                )
            ] = 1
            bad_pixels[index, 0:quad_height, quad_width : self.active_pixels_width][
                np.where(
                    image_cube[
                        index, 0:quad_height, quad_width : self.active_pixels_width
                    ]
                    > np.nanmedian(
                        image_cube[
                            index, 0:quad_height, quad_width : self.active_pixels_width
                        ]
                    )
                    + (
                        sigma
                        * np.nanstd(
                            image_cube[
                                index,
                                0:quad_height,
                                quad_width : self.active_pixels_width,
                            ]
                        )
                    )
                )
            ] = 1
            bad_pixels[index, quad_height : self.active_pixels_height, 0:quad_width][
                np.where(
                    image_cube[
                        index, quad_height : self.active_pixels_height, 0:quad_width
                    ]
                    > np.nanmedian(
                        image_cube[
                            index, quad_height : self.active_pixels_height, 0:quad_width
                        ]
                    )
                    + (
                        sigma
                        * np.nanstd(
                            image_cube[
                                index,
                                quad_height : self.active_pixels_height,
                                0:quad_width,
                            ]
                        )
                    )
                )
            ] = 1
            bad_pixels[
                index,
                quad_height : self.active_pixels_height,
                quad_width : self.active_pixels_width,
            ][
                np.where(
                    image_cube[
                        index,
                        quad_height : self.active_pixels_height,
                        quad_width : self.active_pixels_width,
                    ]
                    > np.nanmedian(
                        image_cube[
                            index,
                            quad_height : self.active_pixels_height,
                            quad_width : self.active_pixels_width,
                        ]
                    )
                    + (
                        sigma
                        * np.nanstd(
                            image_cube[
                                index,
                                quad_height : self.active_pixels_height,
                                quad_width : self.active_pixels_width,
                            ]
                        )
                    )
                )
            ] = 1

        # Set dark and hot pixels where number is equal to or greater than percent threshold
        bad_pixel_mask[
            np.where(np.nanmean(bad_pixels, axis=0) >= percent_threshold)
        ] = 1

        bad_pixel_mask_header = fits.Header()
        bad_pixel_mask_header["CAM_ID"] = (camera_id, "Camera ID")
        bad_pixel_mask_header["CAM_SN"] = (camera_sn, "Camera Serial Number")
        bad_pixel_mask_header["LEVEL"] = ("0.1", "Data Product Level")
        image_sequence_numbers = self.create_image_sequence_number_list(
            img_seq_num_list
        )
        bad_pixel_mask_header["IMGSEQNM"] = (
            image_sequence_numbers,
            "Image Sequence Numbers",
        )
        bad_pixel_mask_header["SIGMA"] = (sigma, "Sigma")
        bad_pixel_mask_header["PCNTTHLD"] = (percent_threshold, "Percent Threshold")

        # Create output directory.
        os.makedirs(os.path.dirname(bad_pixel_mask_file), exist_ok=True)

        hdu = fits.PrimaryHDU(data=bad_pixel_mask, header=bad_pixel_mask_header)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(bad_pixel_mask_file, overwrite=True)

    def create_master_dark(
        self, image_list: list, sigma: np.float32, master_dark_file: str
    ):
        """
        Creates a master dark image.  Spikes are ignored in the creation of the
        master dark.  The algorithm for spike detection is median + (sigma * standard deviation).
        The image housekeeping parameters are averaged and stored with the master dark.

        Parameters
        ----------
        image_list : list
            Dark image filenames used to create master dark.
        sigma : np.float
            Sigma.
        master_dark_file : str
            Filename for master dark.

        Returns
        -------
        None.

        """
        number_images = len(image_list)
        image_cube = np.zeros(
            (number_images, self.active_pixels_height, self.active_pixels_width),
            dtype=np.float32,
        )
        master_dark = np.zeros(
            (self.active_pixels_height, self.active_pixels_width), dtype=np.float32
        )

        summed_image_exposure_time = 0.0
        summed_measured_exposure_time = 0.0
        summed_fpga_temp = 0.0
        summed_fpga_vint = 0.0
        summed_fpga_vaux = 0.0
        summed_fpga_vbrm = 0.0
        summed_adc_temp = np.zeros(8)
        img_seq_num_list = []

        first_image = True
        for index in range(number_images):
            with fits.open(image_list[index]) as image_hdul:
                if first_image is True:
                    camera_id = image_hdul[0].header["CAM_ID"]
                    camera_sn = image_hdul[0].header["CAM_SN"]
                    first_image = False
                summed_image_exposure_time += image_hdul[0].header["IMG_EXP"]
                summed_measured_exposure_time += image_hdul[0].header["MEAS_EXP"]
                summed_fpga_temp += image_hdul[0].header["FPGATEMP"]
                summed_fpga_vint += image_hdul[0].header["FPGAVINT"]
                summed_fpga_vaux += image_hdul[0].header["FPGAVAUX"]
                summed_fpga_vbrm += image_hdul[0].header["FPGAVBRM"]
                for t in range(8):
                    summed_adc_temp[t] += image_hdul[0].header[f"ADCTEMP{t+1}"]
                img_seq_num_list.append(image_hdul[0].header["IMG_ISN"])
                image = self.remove_bias(image_hdul[0].data.astype(np.float32))
                image = self.remove_inactive_pixels(image)
                image_cube[index, :, :] = image

        # Despike.
        image_cube[
            np.where(
                image_cube
                > (
                    np.nanmedian(image_cube, axis=0)
                    + (sigma * np.nanstd(image_cube, axis=0))
                )
            )
        ] = np.nan

        master_dark = np.nanmean(image_cube, axis=0)

        master_dark_header = fits.Header()
        master_dark_header["CAM_ID"] = (camera_id, "Camera ID")
        master_dark_header["CAM_SN"] = (camera_sn, "Camera Serial Number")
        master_dark_header["IMG_EXP"] = (
            summed_image_exposure_time / number_images,
            "Exposure (seconds)",
        )
        master_dark_header["MEAS_EXP"] = (
            summed_measured_exposure_time / number_images,
            "Measured Exposure (seconds)",
        )
        master_dark_header["FPGATEMP"] = (
            summed_fpga_temp / number_images,
            "FPGA Temperature (degC)",
        )
        master_dark_header["FPGAVINT"] = (
            summed_fpga_vint / number_images,
            "FPGA VccInt Voltage (volts)",
        )
        master_dark_header["FPGAVAUX"] = (
            summed_fpga_vaux / number_images,
            "FPGA VccAux (volts)",
        )
        master_dark_header["FPGAVBRM"] = (
            summed_fpga_vbrm / number_images,
            "FPGA VccBram (volts)",
        )
        for t in range(8):
            master_dark_header.append(
                (
                    f"ADCTEMP{t+1}",
                    summed_adc_temp[t] / number_images,
                    f"ADC Temperature{t+1} (degC",
                ),
                end=True,
            )

        master_dark_header["LEVEL"] = ("0.2", "Data Product Level")
        image_sequence_numbers = self.create_image_sequence_number_list(
            img_seq_num_list
        )
        master_dark_header["IMGSEQNM"] = (
            image_sequence_numbers,
            "Image Sequence Numbers",
        )
        master_dark_header["SIGMA"] = (sigma, "Sigma")

        # Create output directory.
        os.makedirs(os.path.dirname(master_dark_file), exist_ok=True)

        hdu = fits.PrimaryHDU(data=master_dark, header=master_dark_header)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(master_dark_file, overwrite=True)

    def create_level0_5_images(self, image_list: list, output_dir: str):
        """
        Creates Level 0.5 images with corrected timestamps.  Level 0.5 header information is added.

        Parameters
        ----------
        image_list : list
            List of Level 0 images.
        output_dir : str
            Directory to write Level 0.5 images.

        Returns
        -------
        None.

        """
        number_images = len(image_list)

        launch_year = datetime(2021, 1, 1, tzinfo=timezone.utc)
        launch_date = dateutil.parser.isoparse("2021-07-30T18:22:21Z")
        exp_launch_time = launch_date - launch_year
        t0_delta = (
            launch_t0 - timedelta(days=1).total_seconds()
        ) - exp_launch_time.total_seconds()
        # Correct for launch signal trigger plus verification time.
        t0_delta = t0_delta - 1.1

        for index in range(number_images):
            with fits.open(image_list[index]) as image_hdul:
                image_date = dateutil.parser.isoparse(image_hdul[0].header["IMG_TS"])
                measured_exposure = image_hdul[0].header["MEAS_EXP"]
                image_hdul[0].header["LEVEL"] = ("0.5", "Data Product Level")
                image_hdul[0].header["EXPTIME"] = (measured_exposure, "seconds")
                date_obs_image_date = (
                    image_date
                    + timedelta(seconds=t0_delta)
                    - timedelta(seconds=measured_exposure)
                )
                image_hdul[0].header["DATE_OBS"] = (
                    date_obs_image_date.isoformat(timespec="milliseconds").replace(
                        "+00:00", "Z"
                    ),
                    "Date Observation",
                )
                t_obs_image_date = date_obs_image_date + timedelta(
                    seconds=(measured_exposure / 2.0)
                )
                image_hdul[0].header["T_OBS"] = (
                    t_obs_image_date.isoformat(timespec="milliseconds").replace(
                        "+00:00", "Z"
                    ),
                    "Telescope Observation",
                )
                image_hdul[0].header["TELESCOP"] = ("MaGIXS", "Telescope")
                image_hdul[0].header["INSTRUME"] = ("MaGIXS", "Instrument")
                image_hdul[0].header["IMGACQST"] = ("", "Image Acquisition State")
                image_hdul[0].header["CCDHLDR"] = (0.0, "CCD Holder Temperature (degC)")
                image_hdul[0].header["COLDBLOC"] = (
                    0.0,
                    "Cold Block Temperature (degC)",
                )
                image_hdul[0].header["UNITS"] = ("DN", "Units")

                image_timestamp = date_obs_image_date.isoformat(
                    timespec="milliseconds"
                ).replace("+00:00", "Z")
                # Create output directory.
                os.makedirs(output_dir, exist_ok=True)
                output_file = (
                    output_dir
                    + "magixs_L0.5_"
                    + image_timestamp.replace(":", ".")
                    + ".fits"
                )
                hdu = fits.PrimaryHDU(
                    data=image_hdul[0].data, header=image_hdul[0].header
                )
                hdulist = fits.HDUList([hdu])
                hdulist.writeto(output_file, overwrite=True)

    def update_level0_5_image_acquisition_state(
        self,
        input_dir: str,
        image_sequence_number_list: list,
        image_acquisition_state: str,
    ):
        """
        Updates Level 0.5 images with the Image Acquisition State.

        Parameters
        ----------
        input_dir : str
            Directory of Level 0.5 images to update.
        image_sequence_number_list : list
            List of Image Sequence Numbers for state.
        image_acquisition_state : str
            Image Acquistion State (e.g. 'Dark', 'Shutter Opening', 'Pointing',
            'Light', and 'Shutter Closing').

        Returns
        -------
        None.

        """
        os.chdir(input_dir)
        for file in glob.glob("*.fits"):
            with fits.open(file) as image_hdul:
                img_seq_num = image_hdul[0].header["IMG_ISN"]
                if img_seq_num in image_sequence_number_list:
                    image_hdul[0].header["IMGACQST"] = image_acquisition_state
                    hdu = fits.PrimaryHDU(
                        data=image_hdul[0].data, header=image_hdul[0].header
                    )
                    hdulist = fits.HDUList([hdu])
                    hdulist.writeto(file, overwrite=True)

    def update_level0_5_temperatures(self, input_dir: str, matlab_file: str):
        """
        Updates Level 0.5 CCD holder and cold block temperatures.

        Parameters
        ----------
        input_dir : str
           Directory of Level 0.5 images to update..
        matlab_file : str
            Matlab filename containing the temperatures.  This is a NSROC file.

        Returns
        -------
        None.

        """
        matlab_data = loadmat(matlab_file)
        ccd_param = matlab_data["ccdt"]
        ccd_param_time = matlab_data["ccdt_Time"]
        ccd_param_time -= timedelta(days=1).total_seconds()
        ccd_param_cal = calc_ccd_holder_cal(ccd_param)
        ccd_param_cal = np.array(ccd_param_cal).flatten()
        ccd_param_time = np.array(ccd_param_time).flatten()
        cold_block_param = matlab_data["cldblkt"]
        cold_block_param_time = matlab_data["cldblkt_Time"]
        cold_block_param_time -= timedelta(days=1).total_seconds()
        cold_block_param_cal = calc_cold_block_cal(cold_block_param)
        cold_block_param_cal = np.array(cold_block_param_cal).flatten()
        cold_block_param_time = np.array(cold_block_param_time).flatten()

        launch_year = datetime(2021, 1, 1, tzinfo=timezone.utc)

        os.chdir(input_dir)
        for file in glob.glob("*.fits"):
            with fits.open(file) as image_hdul:
                image_date = dateutil.parser.isoparse(image_hdul[0].header["DATE_OBS"])
                image_delta = (image_date - launch_year).total_seconds()
                index = np.argmax(ccd_param_time > image_delta)
                image_hdul[0].header["CCDHLDR"] = (ccd_param_cal[index], "degC")
                index = np.argmax(cold_block_param_time > image_delta)
                image_hdul[0].header["COLDBLOC"] = (cold_block_param_cal[index], "degC")
                hdu = fits.PrimaryHDU(
                    data=image_hdul[0].data, header=image_hdul[0].header
                )
                hdulist = fits.HDUList([hdu])
                hdulist.writeto(file, overwrite=True)

    def create_level1_0_images(
        self,
        image_list: list,
        pre_master_dark_file: str,
        post_master_dark_file: str,
        bad_pixel_mask_file: tp.Union[str, None],
        output_dir: str,
    ):
        """
        Creates Level 1.0 images.  If bad pixels replaced, replaced values and their
        locations are stored as a table in the image FITS file.

        Parameters
        ----------
        image_list : list
            List of raw image filenames (typically Level 0.5).
        pre_master_dark_file : str
            Pre-Master Dark filename.
        post_master_dark_file : str
            Post-Master Dark filename.
        bad_pixel_mask_file : tp.Union[str, None]
            Bad Pixel Mask filenanme.
        output_dir : str
            Directory to write Level 1.0 images.

        Returns
        -------
        None.

        """
        number_images = len(image_list)

        with fits.open(pre_master_dark_file) as dark_hdul:
            pre_master_dark = dark_hdul[0].data.astype(np.float32).copy()
            pre_dark_temp = dark_hdul[0].header["ADCTEMP6"]
        with fits.open(post_master_dark_file) as dark_hdul:
            post_master_dark = dark_hdul[0].data.astype(np.float32).copy()
            post_dark_temp = dark_hdul[0].header["ADCTEMP6"]
        if bad_pixel_mask_file is not None:
            with fits.open(bad_pixel_mask_file) as bad_pixel_mask_hdul:
                bad_pixel_mask = bad_pixel_mask_hdul[0].data.astype(np.float32).copy()
        else:
            bad_pixel_mask = None

        despike_table_hdu = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="y", format="J"),
                fits.Column(name="x", format="J"),
                fits.Column(name="value", format="E"),
            ]
        )
        wcs_table_hdu = fits.BinTableHDU.from_columns(
            [
                fits.Column(name="wavelength", format="E"),
                fits.Column(name="plate_scale_x", format="E"),
                fits.Column(name="pixel_x", format="E"),
                fits.Column(name="solar_coord_x", format="E"),
                fits.Column(name="plate_scale_y", format="E"),
                fits.Column(name="pixel_y", format="E"),
                fits.Column(name="solar_coord_y", format="E"),
                fits.Column(name="roll", format="E"),
            ]
        )
        for index in range(number_images):
            with fits.open(image_list[index]) as image_hdul:
                (
                    image,
                    image_header,
                    bad_pixels_replaced_values,
                ) = self.create_adjusted_light(
                    image_hdul[0].data,
                    image_hdul[0].header,
                    pre_master_dark,
                    pre_dark_temp,
                    post_master_dark,
                    post_dark_temp,
                    bad_pixel_mask,
                )
                image_header["PREMDARK"] = (
                    os.path.basename(pre_master_dark_file),
                    "Pre Master Dark",
                )
                image_header["PSTMDARK"] = (
                    os.path.basename(post_master_dark_file),
                    "Post Master Dark",
                )
                image_header["INTPTEMP"] = ("ADCTEMP6", "Interpolation Temperature")

                # Create output directory.
                os.makedirs(output_dir, exist_ok=True)
                image_date = dateutil.parser.isoparse(image_hdul[0].header["DATE_OBS"])
                image_timestamp = image_date.isoformat(timespec="milliseconds").replace(
                    "+00:00", "Z"
                )
                output_file = (
                    output_dir
                    + "magixs_L1.0_"
                    + image_timestamp.replace(":", ".")
                    + ".fits"
                )
                fits_hdu = fits.PrimaryHDU(data=image, header=image_header)
                bad_pixels_replaced_dict = {
                    "y": np.int32,
                    "x": np.int32,
                    "value": np.float32,
                }
                bad_pixels_replaced_values = bad_pixels_replaced_values.astype(
                    bad_pixels_replaced_dict
                )
                bad_pixels_table = Table.from_pandas(bad_pixels_replaced_values)
                hdu_list = fits.HDUList(
                    [
                        fits_hdu,
                        fits.table_to_hdu(bad_pixels_table),
                        despike_table_hdu,
                        wcs_table_hdu,
                    ]
                )
                hdu_list.writeto(output_file, overwrite=True)

    def create_adjusted_light(
        self,
        image: np.ndarray,
        level0_header: astropy.io.fits.header.Header,
        pre_master_dark: np.ndarray,
        pre_master_dark_temp: np.float32,
        post_master_dark: np.ndarray,
        post_master_dark_temp: np.float32,
        bad_pixel_mask: tp.Union[np.ndarray, None] = None,
    ) -> [np.ndarray, astropy.io.fits.header.Header, pd.DataFrame]:
        """
        Creates an adjusted light image.  The bias is removed, the inactive pixels
        are removed, the master dark is removed, the gain is adjusted, and
        optionally the bad pixels are replaced.  The master dark is removed using a
        linear interpolation between the pre and post master darks and the images's
        analog temperature.  Level 1.0 header information is added.

        Parameters
        ----------
        image : np.ndarray
            Light image to adjust.
        level0_header : tp.Union[astropy.io.fits.header.Header, None]
            Level 0 header of light image.  If None, new header is created.
        master_dark : np.ndarray
            Master dark image to remove from light image.
        bad_pixel_mask : tp.Union[np.ndarray, None], optional
            Mask of bad pixels to replace in light image. The default is None.
        adjusted_light_file : tp.Union[str, None], optional
            Filename for adjusted light image. The default is None.

        Returns
        -------
        adjusted_image : np.ndarray
            Adjusted image.
        level1_header : astropy.io.fits.header.Header
            Level 1.0 header for the adjusted image.
        bad_pixels_replaced_values: pd.DataFrame
            Bad pixel replaced values (y, x, value).

        """
        image_height, image_width = np.shape(image)
        adjusted_image = np.zeros(
            (self.active_pixels_height, self.active_pixels_width), dtype=np.float32
        )

        image_temp = level0_header["ADCTEMP6"]

        level1_header = level0_header

        level1_header["LEVEL"] = "1.0"

        adjusted_image = self.remove_bias(image.astype(np.float32))
        level1_header["DEBIASED"] = (1, "Bias Removed")
        adjusted_image = self.remove_inactive_pixels(adjusted_image)
        master_dark_scale = (image_temp - pre_master_dark_temp) / (
            post_master_dark_temp - pre_master_dark_temp
        )
        # print(master_dark_scale)
        scaled_master_dark = pre_master_dark.astype(np.float32) + (
            master_dark_scale
            * (post_master_dark.astype(np.float32) - pre_master_dark.astype(np.float32))
        )
        adjusted_image -= scaled_master_dark.astype(np.float32)
        # print("post", adjusted_image[544, 1517])
        level1_header["DEDARKED"] = (1, "Dark Removed")
        adjusted_image = self.adjust_gain(adjusted_image)
        level1_header["GAINADJ"] = (1, "Gain Adjusted")
        if bad_pixel_mask is not None:
            adjust_image, bad_pixels_replaced_values = self.replace_bad_pixels(
                adjusted_image, bad_pixel_mask
            )
            # print("bad pixels replaced values =", bad_pixels_replaced_values)
            level1_header["BDPIXRPL"] = (1, "Bad Pixels Replaced")
        else:
            bad_pixels_replaced_values = pd.DataFrame(columns=["y", "x", "value"])
            level1_header["BDPIXRPL"] = (0, "Bad Pixels Replaced")
        # print("final", adjusted_image[544, 1517])
        # print("")

        level1_header["DESPIKED"] = (0, "Despiked")
        level1_header["ABSORADJ"] = (0, "Absorption Adjusted")

        level1_header["UNITS"] = ("Electrons", "Units")

        level1_header["BIAS1"] = (self.ll_bias, "Bias Tap E (Lower Left)")
        level1_header["BIAS2"] = (self.lr_bias, "Bias Tap F (Lower Right)")
        level1_header["BIAS3"] = (self.ur_bias, "Bias Tap G (Upper Right)")
        level1_header["BIAS4"] = (self.ul_bias, "Bias Tap H (Upper Left)")
        level1_header["GAIN1"] = (self.ll_gain, "Gain Tap E (Lower Left)")
        level1_header["GAIN2"] = (self.lr_gain, "Gain Tap F (Lower Right)")
        level1_header["GAIN3"] = (self.ur_gain, "Gain Tap G (Upper Right)")
        level1_header["GAIN4"] = (self.ul_gain, "Gain Tap H (Upper Left)")

        return adjusted_image, level1_header, bad_pixels_replaced_values

    def create_level1_5_images(
        self,
        image_list: list,
        despike_areas: list,
        despike_sigma: np.float32,
        output_dir: str,
    ):
        """
        Creates Level 1.5 images that are ready for inversion.

        Parameters
        ----------
        image_list : list
            List of Level 1.0 filenames.
        despike_areas : list
            Despike areas.  This is a list containing the box coordinates and their thresholds.
        despike_sigma : np.float32
            Despike sigma to determine when a pixel value is considered a spike.
        output_dir : str
           Directory to write Level 1.5 images.

        Returns
        -------
        None.

        """
        number_images = len(image_list)

        for index in range(number_images):
            with fits.open(image_list[index]) as image_hdul:
                (
                    image,
                    image_header,
                    spike_replaced_values,
                ) = self.create_inversion_ready_light_with_mask(
                    image_hdul[0].data,
                    image_hdul[0].header,
                    despike_areas,
                    despike_sigma,
                )
                # Create output directory.
                os.makedirs(output_dir, exist_ok=True)
                image_date = dateutil.parser.isoparse(image_hdul[0].header["DATE_OBS"])
                image_timestamp = image_date.isoformat(timespec="milliseconds").replace(
                    "+00:00", "Z"
                )
                output_file = (
                    output_dir
                    + "magixs_L1.5_"
                    + image_timestamp.replace(":", ".")
                    + ".fits"
                )
                image_hdul[0].data = image
                image_hdul[0].header = image_header
                spike_dict = {"y": np.int32, "x": np.int32, "value": np.float32}
                spike_replaced_values = spike_replaced_values.astype(spike_dict)
                # print(spike_replaced_values)
                spike_table = Table.from_pandas(spike_replaced_values)
                spike_table_hdu = fits.table_to_hdu(spike_table)
                image_hdul[2].data = spike_table_hdu.data
                image_hdul[2].header = spike_table_hdu.header
                image_hdul.writeto(output_file, overwrite=True)

    def despike_pixel_with_mask(
        self,
        despiked_image: np.ndarray,
        y_pixel: np.float32,
        x_pixel: np.float32,
        despike_threshold: np.float32,
        despike_sigma: np.float32,
        spikes: list,
    ) -> [np.ndarray, pd.DataFrame]:
        """


        Parameters
        ----------
        despiked_image : np.ndarray
            DESCRIPTION.
        y_pixel : np.float32
            DESCRIPTION.
        x_pixel : np.float32
            DESCRIPTION.
        despike_threshold : np.float32
            DESCRIPTION.
        despike_sigma : np.float32
            DESCRIPTION.
        spikes : list
            DESCRIPTION.

        Returns
        -------
        despiked_image : TYPE
            DESCRIPTION.
        spike_replaced_values : TYPE
            DESCRIPTION.

        """
        # print("despiking pixel (", y_pixel, ", ", x_pixel, ")")

        spike_replaced_values = pd.DataFrame(columns=["y", "x", "value"])

        surrounding_pixels_values = []
        surrounding_pixels_list = []
        skip = False
        skip_last = False
        # This algorithm skips pixels on either side of a pixel that is greater than the threshold.
        for index, delta in enumerate(pixel_16_adjacent_deltas):
            y = y_pixel + delta[0]
            x = x_pixel + delta[1]
            # print("y, x", y, x)
            # If the first pixel is greater than threshold, skip last.
            if (index == (len(pixel_16_adjacent_deltas) - 1)) and skip_last:
                continue
            if skip:
                # If adjacent pixels are greater than threshold, skip again.
                if (y, x) not in spikes:
                    # if (despiked_image[y, x] < despike_threshold):
                    skip = False
                continue
            if (y >= 0 and y < self.active_pixels_height) and (
                x >= 0 and x < self.active_pixels_width
            ):
                if (y, x) not in spikes and despiked_image[(y, x)] is not ma.masked:
                    # if (despiked_image[y, x] < despike_threshold):
                    surrounding_pixels_values += [despiked_image[y, x]]
                    surrounding_pixels_list += [(y, x)]
                else:
                    if surrounding_pixels_values:
                        del surrounding_pixels_values[-1]
                        del surrounding_pixels_list[-1]
                    skip = True
                    if index == 0:
                        skip_last = True

        for index, delta in enumerate(pixel_8_adjacent_deltas):
            y = y_pixel + delta[0]
            x = x_pixel + delta[1]
            if (y, x) in spikes:
                for i in range(index * 2, (index * 2) + 3):
                    try:
                        list_y = y_pixel + pixel_16_adjacent_wrap_deltas[i][0]
                        list_x = x_pixel + pixel_16_adjacent_wrap_deltas[i][1]
                        list_index = surrounding_pixels_list.index((list_y, list_x))
                        del surrounding_pixels_list[list_index]
                    except:  # noqa: E722 # TODO figure out what exception was expected
                        pass

        # if len(surrounding_pixels_values) > 1:
        if len(surrounding_pixels_list) > 1:
            surrounding_pixels_values = []
            for coords in surrounding_pixels_list:
                surrounding_pixels_values += [despiked_image[coords]]
            if despiked_image[y_pixel, x_pixel] > np.median(
                surrounding_pixels_values
            ) + (despike_sigma * np.std(surrounding_pixels_values)):
                # despiked_image[y_pixel, x_pixel] = np.mean(surrounding_pixels_values)
                spike_replaced_values.loc[len(spike_replaced_values.index)] = [
                    y_pixel,
                    x_pixel,
                    despiked_image[y_pixel, x_pixel],
                ]
                # print(y_pixel, x_pixel, despiked_image[y_pixel, x_pixel])
                despiked_image[y_pixel, x_pixel] = np.median(surrounding_pixels_values)
                # print("despike_pixel =", np.mean(surrounding_pixels_values))
                # print("despike_pixel =", np.median(surrounding_pixels_values))

                for delta in pixel_8_deltas:
                    # for delta in pixel_16_deltas:
                    y = y_pixel + delta[0]
                    x = x_pixel + delta[1]
                    if (
                        (y >= 0 and y < self.active_pixels_height)
                        and (x >= 0 and x < self.active_pixels_width)
                        and (despiked_image[y, x] < despike_threshold)
                        and despiked_image[(y, x)] is not ma.masked
                    ):
                        (
                            despiked_image,
                            area_spike_values_replaced,
                        ) = self.despike_pixel_with_mask(
                            despiked_image,
                            y,
                            x,
                            despike_threshold,
                            despike_sigma,
                            spikes,
                        )
                        spike_replaced_values = spike_replaced_values.append(
                            area_spike_values_replaced, ignore_index=True
                        )
        elif len(surrounding_pixels_list) == 1:
            if (
                despiked_image[y_pixel, x_pixel]
                > despiked_image[surrounding_pixels_list[0]]
            ):
                spike_replaced_values.loc[len(spike_replaced_values.index)] = [
                    y_pixel,
                    x_pixel,
                    despiked_image[y_pixel, x_pixel],
                ]
                # print(y_pixel, x_pixel, despiked_image[y_pixel, x_pixel])
                despiked_image[y_pixel, x_pixel] = despiked_image[
                    surrounding_pixels_list[0]
                ]
        else:
            surrounding_pixels_values = []
            for index, delta in enumerate(pixel_16_adjacent_deltas):
                y = y_pixel + delta[0]
                x = x_pixel + delta[1]
                if (
                    (y >= 0 and y < self.active_pixels_height)
                    and (x >= 0 and x < self.active_pixels_width)
                    and (despiked_image[y, x] < despike_threshold)
                    and despiked_image[(y, x)] is not ma.masked
                ):
                    surrounding_pixels_values += [despiked_image[y, x]]
            if len(surrounding_pixels_values) > 0:
                spike_replaced_values.loc[len(spike_replaced_values.index)] = [
                    y_pixel,
                    x_pixel,
                    despiked_image[y_pixel, x_pixel],
                ]
                # print(y_pixel, x_pixel, despiked_image[y_pixel, x_pixel])
                despiked_image[y_pixel, x_pixel] = np.median(surrounding_pixels_values)

        return despiked_image, spike_replaced_values

    def despike_with_mask(
        self,
        image: ma.masked_array,
        despike_threshold: np.float32,
        despike_sigma: np.float32,
    ) -> [np.ndarray, pd.DataFrame]:
        """


        Parameters
        ----------
        image : ma.masked_array
            DESCRIPTION.
        despike_threshold : np.float32
            DESCRIPTION.
        despike_sigma : np.float32
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        spike_replaced_values : TYPE
            DESCRIPTION.

        """
        despiked_image = image
        height, width = np.shape(image)
        # print("image h,w", height, width)
        spike_replaced_values = pd.DataFrame(columns=["y", "x", "value"])
        spikes = list(zip(*np.ma.where(despiked_image >= despike_threshold)))
        original_spikes = spikes.copy()
        spike_values = [despiked_image[i] for i in spikes]
        # print("Number of spikes =", len(spike_values))
        while spike_values:
            min_spike_index = np.argmin(spike_values)
            y_pixel = spikes[min_spike_index][0]
            x_pixel = spikes[min_spike_index][1]
            # print("despiking pixel (", y_pixel, ", ", x_pixel, "), value", spike_values[min_spike_index])
            despiked_image, area_spike_replaced_values = self.despike_pixel_with_mask(
                despiked_image,
                y_pixel,
                x_pixel,
                despike_threshold,
                despike_sigma,
                original_spikes,
            )
            spike_replaced_values = spike_replaced_values.append(
                area_spike_replaced_values, ignore_index=True
            )
            del spikes[min_spike_index]
            del spike_values[min_spike_index]

        return despiked_image.data, spike_replaced_values

    def create_inversion_ready_light_with_mask(
        self,
        image: np.ndarray,
        level1_header: astropy.io.fits.header.Header,
        despike_areas: list,
        despike_sigma: np.float32,
    ) -> [np.ndarray, astropy.io.fits.header.Header, pd.DataFrame]:
        """


        Parameters
        ----------
        image : np.ndarray
            DESCRIPTION.
        level1_header : astropy.io.fits.header.Header
            DESCRIPTION.
        despike_areas : list
            DESCRIPTION.
        despike_sigma : np.float32
            DESCRIPTION.

        Returns
        -------
        inversion_ready_image : TYPE
            DESCRIPTION.
        level1_5_header : TYPE
            DESCRIPTION.
        spike_replaced_values : TYPE
            DESCRIPTION.

        """
        image_height, image_width = np.shape(image)
        inversion_ready_image = np.zeros(
            (self.active_pixels_height, self.active_pixels_width), dtype=np.float32
        )
        spike_replaced_values = pd.DataFrame(columns=["y", "x", "value"])
        mask = np.zeros(
            (self.active_pixels_height, self.active_pixels_width), dtype=np.bool
        )

        inversion_ready_image = image.astype(np.float32)

        level1_5_header = level1_header
        level1_5_header["LEVEL"] = ("1.5", "Data Product Level")
        level1_header["DESPIKED"] = (1, "Despiked")

        for despike_area in despike_areas:
            mask[:, :] = True
            despike_threshold = despike_area[1]
            for area in despike_area[0]:
                y0 = area[0][0]
                y1 = area[0][1]
                x0 = area[1][0]
                x1 = area[1][1]
                mask[y0:y1, x0:x1] = False

            masked_image = ma.masked_array(
                inversion_ready_image, mask=mask, fill_value=np.nan
            )
            inversion_ready_image, area_spike_replaced_values = self.despike_with_mask(
                masked_image, despike_threshold, despike_sigma
            )

            spike_replaced_values = spike_replaced_values.append(
                area_spike_replaced_values, ignore_index=True
            )

        return inversion_ready_image, level1_5_header, spike_replaced_values

    def update_level1_pointing(
        self, image_list: list, level1_wcs_table: str, solar_fov_coords: str
    ):
        number_images = len(image_list)

        # Read Level 1 WCS table information.
        wcs_table = pd.read_excel(level1_wcs_table, index_col=None)
        rows, columns = wcs_table.shape
        # Read solar FOV coordinates.
        fov_table = pd.read_excel(solar_fov_coords, index_col=None)
        rows, columns = fov_table.shape

        for index in range(number_images):
            with fits.open(image_list[index]) as image_hdul:
                try:
                    data_product_level = image_hdul[0].header["LEVEL"]
                    data_product_level.rstrip()
                except KeyError:
                    data_product_level = ""
                assert data_product_level == "1.0" or data_product_level == "1.5"
                image_hdul[0].header["SFOVLLX"] = (
                    fov_table.values[0, 0],
                    "Slot FOV Lower Left X",
                )
                image_hdul[0].header["SFOVLLY"] = (
                    fov_table.values[0, 1],
                    "Slot FOV Lower Left Y",
                )
                image_hdul[0].header["SFOVULX"] = (
                    fov_table.values[0, 2],
                    "Slot FOV Upper Left X",
                )
                image_hdul[0].header["SFOVULY"] = (
                    fov_table.values[0, 3],
                    "Slot FOV Upper Left Y",
                )
                image_hdul[0].header["SFOVLRX"] = (
                    fov_table.values[0, 4],
                    "Slot FOV Lower Right X",
                )
                image_hdul[0].header["SFOVLRY"] = (
                    fov_table.values[0, 5],
                    "Slot FOV Lower Right Y",
                )
                image_hdul[0].header["SFOVURX"] = (
                    fov_table.values[0, 6],
                    "Slot FOV Upper Right X",
                )
                image_hdul[0].header["SFOVURY"] = (
                    fov_table.values[0, 7],
                    "Slot FOV Upper Right Y",
                )
                image_hdul[0].header["ROLL"] = (wcs_table.values[0, 7], "degrees")
                pointing_table = Table.from_pandas(wcs_table)
                pointing_table_hdu = fits.table_to_hdu(pointing_table)
                image_hdul[3].data = pointing_table_hdu.data
                image_hdul[3].header = pointing_table_hdu.header

                image_hdul.writeto(image_list[index], overwrite=True)

    def update_sun_radius(
        self, image_list: list, sun_radius_observed: np.float32, sun_radius: np.float32
    ):
        """
        Updates the sun radius information in the primary header.

        Parameters
        ----------
        image_list : list
            List of FITS files to update.
        sun_radius_observed : np.float32
            The observed sun radius in arcsecs.
        sun_radius : np.float32
            The sun radius in arcsecs.

        Returns
        -------
        None.

        """
        number_images = len(image_list)

        for index in range(number_images):
            with fits.open(image_list[index]) as image_hdul:
                image_hdul[0].header["RSUN_OBS"] = (sun_radius_observed, "arcsecs")
                image_hdul[0].header["R_SUN"] = (sun_radius, "arcsecs")
                image_hdul.writeto(image_list[index], overwrite=True)

    def create_level2_1_summed_image(self, image_list: list, output_dir: str):
        """
        Creates a Level 1.5 summed image for a Level 2.1 inversion.
        All light images are summed and normalized.

        Parameters
        ----------
        image_list : list
            List of Level 1.5 filenames.
        output_dir : str
           Directory to write the Level 1.5 summed image.

        Returns
        -------
        None.

        """
        # Create output directory.
        os.makedirs(output_dir, exist_ok=True)
        # image_list.sort()
        # print(image_list)
        light_summed_image = np.zeros((1024, 2048), dtype=np.float64)
        summed_image_exposure_time = 0.0
        spike_replaced_values = pd.DataFrame(columns=["y", "x", "value"])
        level2_0_header = fits.Header()
        level2_0_header["LEVEL"] = "1.5"
        running_summed_length = len(image_list)
        for i in range(running_summed_length):
            with fits.open(image_list[i]) as image_hdul:
                light_summed_image += image_hdul[0].data.astype(np.float64)
                if i == 0:
                    level2_0_header["DATE_OBS"] = (
                        image_hdul[0].header["DATE_OBS"],
                        "Date Observation",
                    )
                    level2_0_header["T_OBS"] = (
                        image_hdul[0].header["T_OBS"],
                        "Telescope Observation",
                    )
                    level2_0_header["TELESCOP"] = (
                        image_hdul[0].header["TELESCOP"],
                        "Telescope",
                    )
                    level2_0_header["INSTRUME"] = (
                        image_hdul[0].header["INSTRUME"],
                        "Instrument",
                    )
                    level2_0_header["DEBIASED"] = (
                        image_hdul[0].header["DEBIASED"],
                        "Bias Removed",
                    )
                    level2_0_header["DEDARKED"] = (
                        image_hdul[0].header["DEDARKED"],
                        "Dark Removed",
                    )
                    level2_0_header["GAINADJ"] = (
                        image_hdul[0].header["GAINADJ"],
                        "Gain Adjusted",
                    )
                    level2_0_header["BDPIXRPL"] = (
                        image_hdul[0].header["BDPIXRPL"],
                        "Bad Pixels Replaced",
                    )
                    level2_0_header["DESPIKED"] = (
                        image_hdul[0].header["DESPIKED"],
                        "Despiked",
                    )
                    level2_0_header["ABSORADJ"] = (
                        image_hdul[0].header["ABSORADJ"],
                        "Absorption Adjusted",
                    )
                    level2_0_header["UNITS"] = ("Electrons s-1", "Units")
                    image_timestamp_begin = image_hdul[0].header["DATE_OBS"]
                if i == (running_summed_length - 1):
                    image_timestamp_end = image_hdul[0].header["DATE_OBS"]
                image_exposure_time = image_hdul[0].header["IMG_EXP"]
                summed_image_exposure_time += image_exposure_time
                image_spike_replaced_values = pd.DataFrame(image_hdul[2].data)
                spike_replaced_values = spike_replaced_values.append(
                    image_spike_replaced_values, ignore_index=True
                )
                if i == (running_summed_length - 1):
                    # Normalize data.
                    light_summed_image /= summed_image_exposure_time
                    light_summed_image[np.where(light_summed_image < 0.0)] = 0.0
                    level2_0_header["EXPTIME"] = summed_image_exposure_time
                    keywords_exist = True
                    try:
                        lower_left_x = image_hdul[0].header["SFOVLLX"]
                        lower_left_y = image_hdul[0].header["SFOVLLY"]
                        upper_left_x = image_hdul[0].header["SFOVULX"]
                        upper_left_y = image_hdul[0].header["SFOVULY"]
                        lower_right_x = image_hdul[0].header["SFOVLRX"]
                        lower_right_y = image_hdul[0].header["SFOVLRY"]
                        upper_right_x = image_hdul[0].header["SFOVURX"]
                        upper_right_y = image_hdul[0].header["SFOVURY"]
                        roll = image_hdul[0].header["ROLL"]
                    except KeyError:
                        keywords_exist = False
                    if keywords_exist:
                        level2_0_header["SFOVLLX"] = (
                            lower_left_x,
                            "SLOT FOV Lower Left X",
                        )
                        level2_0_header["SFOVLLY"] = (
                            lower_left_y,
                            "SLOT FOV Lower Left Y",
                        )
                        level2_0_header["SFOVULX"] = (
                            upper_left_x,
                            "SLOT FOV Upper Left X",
                        )
                        level2_0_header["SFOVULY"] = (
                            upper_left_y,
                            "SLOT FOV Upper Left Y",
                        )
                        level2_0_header["SFOVLRX"] = (
                            lower_right_x,
                            "SLOT FOV Lower Right X",
                        )
                        level2_0_header["SFOVLRY"] = (
                            lower_right_y,
                            "SLOT FOV Lower Right Y",
                        )
                        level2_0_header["SFOVURX"] = (
                            upper_right_x,
                            "SLOT FOV Upper Right X",
                        )
                        level2_0_header["SFOVURY"] = (
                            upper_right_y,
                            "SLOT FOV Upper Right Y",
                        )
                        level2_0_header["ROLL"] = (roll, "degrees")
                    keywords_exist = True
                    try:
                        sun_radius_observed = image_hdul[0].header["RSUN_OBS"]
                        sun_radius = image_hdul[0].header["R_SUN"]
                    except KeyError:
                        keywords_exist = False
                    if keywords_exist:
                        level2_0_header["RSUN_OBS"] = (sun_radius_observed, "arcsecs")
                        level2_0_header["R_SUN"] = (sun_radius, "arcsecs")
                    hdu = fits.PrimaryHDU(
                        data=light_summed_image.astype(np.float32),
                        header=level2_0_header,
                    )
                    spike_dict = {"y": np.int32, "x": np.int32, "value": np.float32}
                    spike_replaced_values = spike_replaced_values.astype(spike_dict)
                    despike_table = Table.from_pandas(spike_replaced_values)
                    depike_table_hdu = fits.table_to_hdu(despike_table)
                    hdu_list = fits.HDUList(
                        [
                            hdu,
                            image_hdul[1].copy(),
                            depike_table_hdu,
                            image_hdul[3].copy(),
                        ]
                    )
        index_slice = image_timestamp_begin.find(".")
        image_timestamp_begin = image_timestamp_begin[:index_slice]
        image_timestamp_begin = image_timestamp_begin.replace(":", ".")
        index_slice = image_timestamp_end.find("T")
        index_slice += 1
        image_timestamp_end = image_timestamp_end[index_slice:]
        index_slice = image_timestamp_end.find(".")
        image_timestamp_end = image_timestamp_end[:index_slice]
        image_timestamp_end = image_timestamp_end.replace(":", ".")
        image_output_file = (
            output_dir
            + "magixs_L1.5_"
            + image_timestamp_begin
            + "_"
            + image_timestamp_end
            + "_summed_image.fits"
        )
        hdu_list.writeto(image_output_file, overwrite=True)

    def create_level2_3_running_summed_images(
        self, image_list: list, running_summed_length: np.int32, output_dir: str
    ):
        """
        Creates Level 1.5 running summed images (i.e. 0 to n, 1 to n+1, etc.) for Level 2.3 inversions.

        Parameters
        ----------
        image_list : list
            List of Level 1.5 filenames.
        running_summed_length : np.int32
            Number of images to sum.
        output_dir : str
           Directory to write Level 1.5 running summed images.

        Returns
        -------
        None.

        """
        # Create output directory.
        os.makedirs(output_dir, exist_ok=True)
        image_list.sort()
        assert len(image_list) > running_summed_length
        num_summed_images = len(image_list) - running_summed_length + 1
        light_summed_image = np.zeros((1024, 2048), dtype=np.float64)
        for j in range(num_summed_images):
            light_summed_image[:, :] = 0.0
            summed_image_exposure_time = 0.0
            spike_replaced_values = pd.DataFrame(columns=["y", "x", "value"])
            level2_0_header = fits.Header()
            level2_0_header["LEVEL"] = "1.5"
            for i in range(running_summed_length):
                with fits.open(image_list[j + i]) as image_hdul:
                    light_summed_image += image_hdul[0].data.astype(np.float64)
                    if i == 0:
                        level2_0_header["DATE_OBS"] = (
                            image_hdul[0].header["DATE_OBS"],
                            "Date Observation",
                        )
                        level2_0_header["T_OBS"] = (
                            image_hdul[0].header["T_OBS"],
                            "Telescope Observation",
                        )
                        level2_0_header["TELESCOP"] = (
                            image_hdul[0].header["TELESCOP"],
                            "Telescope",
                        )
                        level2_0_header["INSTRUME"] = (
                            image_hdul[0].header["INSTRUME"],
                            "Instrument",
                        )
                        level2_0_header["DEBIASED"] = (
                            image_hdul[0].header["DEBIASED"],
                            "Bias Removed",
                        )
                        level2_0_header["DEDARKED"] = (
                            image_hdul[0].header["DEDARKED"],
                            "Dark Removed",
                        )
                        level2_0_header["GAINADJ"] = (
                            image_hdul[0].header["GAINADJ"],
                            "Gain Adjusted",
                        )
                        level2_0_header["BDPIXRPL"] = (
                            image_hdul[0].header["BDPIXRPL"],
                            "Bad Pixels Replaced",
                        )
                        level2_0_header["DESPIKED"] = (
                            image_hdul[0].header["DESPIKED"],
                            "Despiked",
                        )
                        level2_0_header["ABSORADJ"] = (
                            image_hdul[0].header["ABSORADJ"],
                            "Absorption Adjusted",
                        )
                        level2_0_header["UNITS"] = ("Electrons s-1", "Units")
                        image_timestamp_begin = image_hdul[0].header["DATE_OBS"]
                    if i == (running_summed_length - 1):
                        image_timestamp_end = image_hdul[0].header["DATE_OBS"]
                    image_exposure_time = image_hdul[0].header["IMG_EXP"]
                    summed_image_exposure_time += image_exposure_time
                    image_spike_replaced_values = pd.DataFrame(image_hdul[2].data)
                    spike_replaced_values = spike_replaced_values.append(
                        image_spike_replaced_values, ignore_index=True
                    )
                    if i == (running_summed_length - 1):
                        # Normalize data.
                        light_summed_image /= summed_image_exposure_time
                        light_summed_image[np.where(light_summed_image < 0.0)] = 0.0
                        level2_0_header["EXPTIME"] = summed_image_exposure_time
                        keywords_exist = True
                        try:
                            lower_left_x = image_hdul[0].header["SFOVLLX"]
                            lower_left_y = image_hdul[0].header["SFOVLLY"]
                            upper_left_x = image_hdul[0].header["SFOVULX"]
                            upper_left_y = image_hdul[0].header["SFOVULY"]
                            lower_right_x = image_hdul[0].header["SFOVLRX"]
                            lower_right_y = image_hdul[0].header["SFOVLRY"]
                            upper_right_x = image_hdul[0].header["SFOVURX"]
                            upper_right_y = image_hdul[0].header["SFOVURY"]
                            roll = image_hdul[0].header["ROLL"]
                        except KeyError:
                            keywords_exist = False
                        if keywords_exist:
                            level2_0_header["SFOVLLX"] = (
                                lower_left_x,
                                "SLOT FOV Lower Left X",
                            )
                            level2_0_header["SFOVLLY"] = (
                                lower_left_y,
                                "SLOT FOV Lower Left Y",
                            )
                            level2_0_header["SFOVULX"] = (
                                upper_left_x,
                                "SLOT FOV Upper Left X",
                            )
                            level2_0_header["SFOVULY"] = (
                                upper_left_y,
                                "SLOT FOV Upper Left Y",
                            )
                            level2_0_header["SFOVLRX"] = (
                                lower_right_x,
                                "SLOT FOV Lower Right X",
                            )
                            level2_0_header["SFOVLRY"] = (
                                lower_right_y,
                                "SLOT FOV Lower Right Y",
                            )
                            level2_0_header["SFOVURX"] = (
                                upper_right_x,
                                "SLOT FOV Upper Right X",
                            )
                            level2_0_header["SFOVURY"] = (
                                upper_right_y,
                                "SLOT FOV Upper Right Y",
                            )
                            level2_0_header["ROLL"] = (roll, "degrees")
                        keywords_exist = True
                        try:
                            sun_radius_observed = image_hdul[0].header["RSUN_OBS"]
                            sun_radius = image_hdul[0].header["R_SUN"]
                        except KeyError:
                            keywords_exist = False
                        if keywords_exist:
                            level2_0_header["RSUN_OBS"] = (
                                sun_radius_observed,
                                "arcsecs",
                            )
                            level2_0_header["R_SUN"] = (sun_radius, "arcsecs")
                        hdu = fits.PrimaryHDU(
                            data=light_summed_image.astype(np.float32),
                            header=level2_0_header,
                        )
                        spike_dict = {"y": np.int32, "x": np.int32, "value": np.float32}
                        spike_replaced_values = spike_replaced_values.astype(spike_dict)
                        despike_table = Table.from_pandas(spike_replaced_values)
                        depike_table_hdu = fits.table_to_hdu(despike_table)
                        hdu_list = fits.HDUList(
                            [
                                hdu,
                                image_hdul[1].copy(),
                                depike_table_hdu,
                                image_hdul[3].copy(),
                            ]
                        )
            index_slice = image_timestamp_begin.find(".")
            image_timestamp_begin = image_timestamp_begin[:index_slice]
            image_timestamp_begin = image_timestamp_begin.replace(":", ".")
            index_slice = image_timestamp_end.find("T")
            index_slice += 1
            image_timestamp_end = image_timestamp_end[index_slice:]
            index_slice = image_timestamp_end.find(".")
            image_timestamp_end = image_timestamp_end[:index_slice]
            image_timestamp_end = image_timestamp_end.replace(":", ".")
            image_output_file = (
                output_dir
                + "magixs_L1.5_"
                + image_timestamp_begin
                + "_"
                + image_timestamp_end
                + "_summed_image.fits"
            )
            hdu_list.writeto(image_output_file, overwrite=True)

    # def perform_level2_0_elasticnet_inversion(
    #     self,
    #     image_list: list,
    #     rsp_func_cube_file: str,
    #     rsp_dep_name: str,
    #     rsp_dep_list: tp.Union[list, None],
    #     solution_fov_width: np.int32,
    #     smooth_over: str,
    #     field_angle_range: tp.Union[list, None],
    #     image_mask_file: tp.Union[str, None],
    #     level: str,
    #     detector_row_range: tp.Union[list, None],
    #     output_dir: str,
    # ):
    #     """
    #     Performs inversion of Level 2.x images using ElasticNet.
    #
    #     Parameters
    #     ----------
    #     image_list : list
    #         List of Level 2.0 filenames.
    #     rsp_func_cube_file: str
    #         Filename of response function cube.
    #     rsp_dep_name: str
    #         Response dependence name (e.g. 'ion' or 'logt').
    #     rsp_dep_list: list, optional
    #         List of dependence items.  If None, use all dependence values.
    #     solution_fov_width: np.int32
    #         Solution field-of-view width.  1 (all field angles), 2 (every other one), etc.  The default is 1.
    #     smooth_over: str, optional
    #         Inversion smoothing (i.e. 'spatial' or 'dependence').  The default is 'spatial'.
    #     field_angle_range: list, optional
    #         Beginning and ending field angles to invert over.  If None, use all field angles.
    #     image_mask_file : str, optional
    #        Mask to apply to image and response functions for inversion.
    #     level: str
    #         Level value for FITS keyword LEVEL.
    #     detector_row_range: list, optional
    #         Beginning and ending row numbers to invert.  If None, invert all rows.  The default is None.
    #     output_dir : str
    #        Directory to write Level 2.x EM data cubes and model predicted data.
    #
    #     Returns
    #     -------
    #     None.
    #
    #     """
    #     num_images = len(image_list)
    #     if num_images > 0:
    #         for index in range(len(image_list)):
    #             print("Image ", index)
    #             # Read noisy image.
    #             print("Inverting:", image_list[index])
    #
    #             inversion = Inversion(
    #                 rsp_func_cube_file=rsp_func_cube_file,
    #                 rsp_dep_name=rsp_dep_name,
    #                 rsp_dep_list=rsp_dep_list,
    #                 solution_fov_width=solution_fov_width,
    #                 smooth_over=smooth_over,
    #                 field_angle_range=field_angle_range,
    #             )
    #
    #             inversion.initialize_input_data(image_list[index], image_mask_file)
    #
    #             alpha = 1e-5 * 3
    #             rho = 0.1
    #             # alpha = 1e-3
    #             # rho = 0.8
    #             model = enet(
    #                 alpha=alpha,
    #                 l1_ratio=rho,
    #                 max_iter=50000,
    #                 precompute=True,
    #                 positive=True,
    #                 fit_intercept=True,
    #                 selection="cyclic",
    #             )
    #             inv_model = enet_model(model)
    #
    #             basename = os.path.splitext(os.path.basename(image_list[index]))[0]
    #             basename_components = basename.split("_")
    #             basename = basename.replace(basename_components[1], "L" + level)
    #             print(basename)
    #
    #             start = time.time()
    #             inversion.invert(
    #                 inv_model,
    #                 output_dir,
    #                 output_file_prefix=basename,
    #                 output_file_postfix="",
    #                 level=level,
    #                 detector_row_range=detector_row_range,
    #             )
    #             end = time.time()
    #             print("Inversion Time =", end - start)

    # def perform_level2_0_lassolars_inversion(
    #     self,
    #     image_list: list,
    #     rsp_func_cube_file: str,
    #     rsp_dep_name: str,
    #     rsp_dep_list: tp.Union[list, None],
    #     solution_fov_width: np.int32,
    #     smooth_over: str,
    #     field_angle_range: tp.Union[list, None],
    #     image_mask_file: tp.Union[str, None],
    #     level: str,
    #     detector_row_range: tp.Union[list, None],
    #     output_dir: str,
    # ):
    #     """
    #     Performs inversion of Level 2.x images using Lasso Lars.
    #
    #     Parameters
    #     ----------
    #     image_list : list
    #         List of Level 2.0 filenames.
    #     rsp_func_cube_file: str
    #         Filename of response function cube.
    #     rsp_dep_name: str
    #         Response dependence name (e.g. 'ion' or 'logt').
    #     rsp_dep_list: list, optional
    #         List of dependence items.  If None, use all dependence values.
    #     solution_fov_width: np.int32
    #         Solution field-of-view width.  1 (all field angles), 2 (every other one), etc.  The default is 1.
    #     smooth_over: str, optional
    #         Inversion smoothing (i.e. 'spatial' or 'dependence').  The default is 'spatial'.
    #     field_angle_range: list, optional
    #         Beginning and ending field angles to invert over.  If None, use all field angles.
    #     image_mask_file : str, optional
    #        Mask to apply to image and response functions for inversion.
    #     level: str
    #         Level value for FITS keyword LEVEL.
    #     detector_row_range: list, optional
    #         Beginning and ending row numbers to invert.  If None, invert all rows.  The default is None.
    #     output_dir : str
    #        Directory to write Level 2.x EM data cubes and model predicted data.
    #
    #     Returns
    #     -------
    #     None.
    #
    #     """
    #     num_images = len(image_list)
    #     if num_images > 0:
    #         for index in range(len(image_list)):
    #             print("Image ", index)
    #             # Read noisy image.
    #             print("Inverting:", image_list[index])
    #
    #             inversion = Inversion(
    #                 rsp_func_cube_file=rsp_func_cube_file,
    #                 rsp_dep_name=rsp_dep_name,
    #                 rsp_dep_list=rsp_dep_list,
    #                 solution_fov_width=solution_fov_width,
    #                 smooth_over=smooth_over,
    #                 field_angle_range=field_angle_range,
    #             )
    #
    #             inversion.initialize_input_data(image_list[index], image_mask_file)
    #
    #             # alpha = 1e-5 * 3
    #             alpha = 6e-3
    #             model = llars(
    #                 alpha=alpha,
    #                 max_iter=50000,
    #                 normalize=False,
    #                 precompute=True,
    #                 positive=True,
    #                 fit_intercept=True,
    #             )
    #             inv_model = llars_model(model)
    #
    #             basename = os.path.splitext(os.path.basename(image_list[index]))[0]
    #             basename_components = basename.split("_")
    #             basename = basename.replace(basename_components[1], "L" + level)
    #             print(basename)
    #
    #             start = time.time()
    #             inversion.invert(
    #                 inv_model,
    #                 output_dir,
    #                 output_file_prefix=basename,
    #                 output_file_postfix="",
    #                 level=level,
    #                 detector_row_range=detector_row_range,
    #             )
    #             end = time.time()
    #             print("Inversion Time =", end - start)

    def create_level2_0_spectrally_pure_images(
        self, image_list: list, gnt_file: str, rsp_dep_list: list, output_dir: str
    ):
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
                # print(np.shape(gnt_data_values))
                num_gnts, num_gnt_deps = np.shape(gnt_data_values)
                gnt_dep_list = gnt_hdul[1].data["logt"]
                # print(gnt_dep_list)
                try:
                    ion_wavelength_table_format = gnt_hdul[0].header["IWTBLFMT"]
                    if ion_wavelength_table_format == "ion@wavelength":
                        ion_wavelength_name = "ion_wavelength"
                    else:
                        ion_wavelength_name = ion_wavelength_table_format
                    # print(ion_wavelength_name)

                    ion_wavelength_values = gnt_hdul[2].data[ion_wavelength_name]
                    # print(ion_wavelength_values)
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
                        # print(np.around(dep, decimals=2))
                        try:
                            index = np.where(gnt_dep_list == np.around(dep, decimals=2))
                            # print(index[0][0])
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
                            gnt_data_cube = np.zeros(
                                (image_height, num_slits, num_gnts), dtype=np.float64
                            )
                        else:
                            gnt_data_cube = np.transpose(
                                gnt_data_cube.astype(np.float32), axes=(1, 2, 0)
                            )
                            gnt_data_cube[:, :, :] = 0.0
                        for gnt_num in range(num_gnts):
                            gnt_image = (
                                em_data_cube[:, :, 0:num_rsp_deps]
                                * 10**26
                                * gnt_values[gnt_num, 0:num_rsp_deps]
                            ).sum(axis=2)
                            gnt_data_cube[:, :, gnt_num] = gnt_image
                        basename = os.path.splitext(
                            os.path.basename(image_list[index])
                        )[0]
                        # print(type(basename))
                        slice_index = basename.find("_em_data_cube")
                        # print(type(basename))
                        postfix_val = basename.split("_x")
                        postfix_val = postfix_val[1]
                        # print(postfix_val)
                        basename = basename[:slice_index]
                        basename += (
                            "_spectrally_pure_data_cube_x" + postfix_val + ".fits"
                        )
                        gnt_data_cube_file = output_dir + basename
                        # Transpose data (wavelength, y, x).  Readable by ImageJ.
                        gnt_data_cube = np.transpose(
                            gnt_data_cube.astype(np.float32), axes=(2, 0, 1)
                        )
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
                        col1 = fits.Column(
                            name="index", format="1I", array=gnt_index_list
                        )
                        col2 = fits.Column(
                            name=ion_wavelength_name,
                            format="15A",
                            array=ion_wavelength_values,
                        )
                        table_hdu = fits.BinTableHDU.from_columns([col1, col2])
                        em_hdul[1].data = table_hdu.data
                        em_hdul[1].header = table_hdu.header
                        em_hdul.writeto(gnt_data_cube_file, overwrite=True)

    def update_level2_pointing(
        self, image_list: list, level1_wcs_table: str, solar_fov_coords: str
    ):
        number_images = len(image_list)

        # Read Level 1 WCS table information.
        wcs_table = pd.read_excel(level1_wcs_table, index_col=None)
        rows, columns = wcs_table.shape
        # Read solar FOV coordinates.
        fov_table = pd.read_excel(solar_fov_coords, index_col=None)
        rows, columns = fov_table.shape

        for index in range(number_images):
            with fits.open(image_list[index]) as image_hdul:
                num_deps, image_height, image_width = np.shape(image_hdul[0].data)
                # print(image_height, image_width, num_deps)
                try:
                    fa_cdelt = float(image_hdul[0].header["FA_CDELT"])
                    drow_min = float(image_hdul[0].header["DROW_MIN"])
                    drow_max = float(image_hdul[0].header["DROW_MAX"])
                    roll = float(wcs_table.values[0, 7])
                    # Create Level 2 WCS table.
                    level_2_x_wcs_table = wcs_table.iloc[0:1].copy()
                    level_2_x_wcs_table.iloc[[0], [0]] = 0.0
                    level_2_x_wcs_table.iloc[[0], [1]] = fa_cdelt
                    # calc_x_pixel = divmod(image_width, 2)
                    # level_2_x_wcs_table.iloc[[0], [2]] = calc_x_pixel[0]
                    calc_x_pixel = (image_width - 1) / 2.0
                    level_2_x_wcs_table.iloc[[0], [2]] = calc_x_pixel
                    # calc_y_pixel = divmod(((drow_max + 1) - drow_min), 2)
                    # level_1_y_pixel = level_2_x_wcs_table.values[0, 5]
                    calc_y_pixel = (drow_max - drow_min) / 2.0
                    level_1_y_pixel = level_2_x_wcs_table.values[0, 5]
                    level_2_x_wcs_table.iloc[[0], [5]] = drow_min + calc_y_pixel
                    level_2_x_wcs_table.iloc[[0], [3]] = level_2_x_wcs_table.values[
                        0, 3
                    ] - math.cos(90 - roll) * (
                        2.8 * (level_1_y_pixel - level_2_x_wcs_table.values[0, 5])
                    )
                    level_2_x_wcs_table.iloc[[0], [6]] = level_2_x_wcs_table.values[
                        0, 6
                    ] - math.sin(90 - roll) * (
                        2.8 * (level_1_y_pixel - level_2_x_wcs_table.values[0, 5])
                    )
                except KeyError:
                    # Create empty Level 2 WCS table.
                    level_2_x_wcs_table = wcs_table.iloc[0:0].copy()
                # print(level_2_x_wcs_table)
                try:
                    data_product_level = image_hdul[0].header["LEVEL"]
                    data_product_level.rstrip()
                except KeyError:
                    data_product_level = ""
                assert (
                    data_product_level == "2.1"
                    or data_product_level == "2.3"
                    or data_product_level == "2.4"
                )
                image_hdul[0].header["SFOVLLX"] = (
                    fov_table.values[0, 0],
                    "Slot FOV Lower Left X",
                )
                image_hdul[0].header["SFOVLLY"] = (
                    fov_table.values[0, 1],
                    "Slot FOV Lower Left Y",
                )
                image_hdul[0].header["SFOVULX"] = (
                    fov_table.values[0, 2],
                    "Slot FOV Upper Left X",
                )
                image_hdul[0].header["SFOVULY"] = (
                    fov_table.values[0, 3],
                    "Slot FOV Upper Left Y",
                )
                image_hdul[0].header["SFOVLRX"] = (
                    fov_table.values[0, 4],
                    "Slot FOV Lower Right X",
                )
                image_hdul[0].header["SFOVLRY"] = (
                    fov_table.values[0, 5],
                    "Slot FOV Lower Right Y",
                )
                image_hdul[0].header["SFOVURX"] = (
                    fov_table.values[0, 6],
                    "Slot FOV Upper Right X",
                )
                image_hdul[0].header["SFOVURY"] = (
                    fov_table.values[0, 7],
                    "Slot FOV Upper Right Y",
                )
                image_hdul[0].header["ROLL"] = (wcs_table.values[0, 7], "degrees")
                # Add/Update binary table.
                try:
                    pointing_table = Table.from_pandas(level_2_x_wcs_table)
                    pointing_table_hdu = fits.table_to_hdu(pointing_table)
                    image_hdul[2].data = pointing_table_hdu.data
                    image_hdul[2].header = pointing_table_hdu.header
                    image_hdul.writeto(image_list[index], overwrite=True)
                except IndexError:
                    pointing_table = Table.from_pandas(level_2_x_wcs_table)
                    pointing_table_hdu = fits.table_to_hdu(pointing_table)
                    image_hdul.append(pointing_table_hdu)
                    image_hdul.writeto(image_list[index], overwrite=True)

    def create_level3_0_color_color_plot(
        self,
        spectrally_pure_data_cube: str,
        wavelength_list: list,
        saturation: float,
        lambda_scale: float,
        output_plot_filename: str,
    ):
        """
        Creates Level 3.0 color-color plot.

        Parameters
        ----------
        spectrally_pure_data_cube : str
            Spectrally Pure data cube filename.
        wavelength_list : list
            Wavelength list.
        saturation : float
           Saturation.
        lambda_scale: float
            Lambda scale.
        output_plot_filename : str
           Output plot filename.

        Returns
        -------
        None.

        """
        top = 255.0

        assert len(wavelength_list) == 3 or len(wavelength_list) == 4
        print(wavelength_list)

        with fits.open(spectrally_pure_data_cube) as spdc_hdul:
            try:
                dep_name = spdc_hdul[0].header["DEPNAME"]
                wavelengths = spdc_hdul[1].data[dep_name]
                print(wavelengths)
            except KeyError:
                wavelengths = []

            num_gnts, image_height, num_slits = np.shape(spdc_hdul[0].data)
            try:
                solution_scale = spdc_hdul[0].header["SLTNFOV"]
            # except Exception as e:
            except:  # noqa: E722 # TODO figure out what exception was expected
                solution_scale = 1.0

            dep_data = np.zeros(
                (len(wavelength_list), image_height, num_slits), dtype=np.float32
            )
            for i in range(len(wavelength_list)):
                for j in range(len(wavelengths)):
                    if wavelength_list[i] == wavelengths[j]:
                        dep_data[i, :, :] = spdc_hdul[0].data[j, :, :]
                        # dep_data[i, :, :] = spdc_hdul[0].data[j, :, :] / 10**26
                        print(
                            wavelength_list[i], np.max(dep_data[i]), np.min(dep_data[i])
                        )
                        break
            # dep_data[:,:, :] = dep_data[:,:, :] / 10**26
            # average_slits = np.average(dep_data, axis = 0)

            for dep_index in range(len(wavelength_list)):
                # dep_data[dep_index] = dep_data[dep_index] - average_slits
                print(np.max(dep_data[dep_index]), np.min(dep_data[dep_index]))
                dep_data[dep_index] = np.maximum(
                    np.minimum(dep_data[dep_index], saturation), 0.0
                )

            max_value = np.amax(dep_data)

            if len(wavelength_list) == 4:
                slit_data = dep_data[3]
                # slit_data = slit_data * (max_value / slit_max_value)
                # max_value = np.max(slit_data)
                min_value = np.min(slit_data)
                print("yellow", max_value, min_value)
                y_channel = np.maximum(
                    np.minimum(
                        (
                            (top + 0.9999)
                            * (slit_data - min_value)
                            / (max_value - min_value)
                        ).astype(np.int16),
                        top,
                    ),
                    0,
                )

            slit_data = dep_data[0]
            # slit_data = slit_data * (max_value / slit_max_value)
            # max_value = np.max(slit_data)
            min_value = np.min(slit_data)
            print("red", max_value, min_value)
            r_channel = np.maximum(
                np.minimum(
                    (
                        (top + 0.9999)
                        * (slit_data - min_value)
                        / (max_value - min_value)
                    ).astype(np.int16),
                    top,
                ),
                0,
            )
            if len(wavelength_list) == 4:
                r_channel += y_channel
            r = Image.fromarray(r_channel.astype(np.uint8), mode=None)
            slit_data = dep_data[1]
            # slit_data = slit_data * (max_value / slit_max_value)
            # max_value = np.max(slit_data)
            min_value = np.min(slit_data)
            print("green", max_value, min_value)
            g_channel = np.maximum(
                np.minimum(
                    (
                        (top + 0.9999)
                        * (slit_data - min_value)
                        / (max_value - min_value)
                    ).astype(np.int16),
                    top,
                ),
                0,
            )
            if len(wavelength_list) == 4:
                g_channel += y_channel
            g = Image.fromarray(g_channel.astype(np.uint8), mode=None)
            slit_data = dep_data[2]
            # slit_data = slit_data * (max_value / slit_max_value)
            # max_value = np.max(slit_data)
            min_value = np.min(slit_data)
            print("blue", max_value, min_value)
            b_channel = np.maximum(
                np.minimum(
                    (
                        (top + 0.9999)
                        * (slit_data - min_value)
                        / (max_value - min_value)
                    ).astype(np.int16),
                    top,
                ),
                0,
            )
            b = Image.fromarray(b_channel.astype(np.uint8), mode=None)

            rgb_image = Image.merge("RGB", (r, g, b))
            scaled_image = rgb_image.resize(
                (
                    int(rgb_image.width * solution_scale * lambda_scale),
                    int(rgb_image.height),
                )
            )
            scaled_image.show()
            scaled_image.save(output_plot_filename)

    def create_summed_noisy_images(
        self,
        image_list: list,
        num_noisy_images: np.int32,
        output_dir_path: str,
        output_file_post_fix: str = "_Summed_Image",
    ):
        summed_image = np.zeros((1024, 2048), dtype=np.float32)
        summed_image_exposure_time = 0.0

        for i in range(len(image_list)):
            image_hdul = fits.open(image_list[i])
            image_exposure_time = image_hdul[0].header["IMG_EXP"]
            summed_image_exposure_time += image_exposure_time
            summed_image += image_hdul[0].data

        # Create output directory.
        os.makedirs(output_dir_path, exist_ok=True)

        summed_image[np.where(summed_image < 0.0)] = 0.0

        summed_image_filename = (
            output_dir_path + "Original" + output_file_post_fix + ".fits"
        )
        level1_5_header = fits.Header()
        level1_5_header["IMG_EXP"] = (summed_image_exposure_time, "Image Exposure")
        level1_5_header["MEAS_EXP"] = (
            summed_image_exposure_time,
            "Measurement Exposure",
        )
        hdu = fits.PrimaryHDU(data=summed_image, header=level1_5_header)
        hdu.writeto(summed_image_filename, overwrite=True)

        self.create_noisy_images(
            summed_image_filename,
            num_noisy_images,
            output_dir_path,
            output_file_post_fix,
        )

    def create_noisy_images(
        self,
        image_file: str,
        num_noisy_images: np.int32,
        output_dir_path: str,
        output_file_post_fix: str,
    ):
        image_hdul = fits.open(image_file)
        image = image_hdul[0].data
        try:
            image_exposure_time = image_hdul[0].header["IMG_EXP"]
        except:  # noqa: E722 # TODO figure out what exception was expected
            image_exposure_time = 1.0

        # Create output directory.
        os.makedirs(output_dir_path, exist_ok=True)

        noisy_image = image.copy() / image_exposure_time
        hdu = fits.PrimaryHDU(noisy_image.astype(np.float32))
        noisy_image_filename = output_dir_path + "Run0" + output_file_post_fix + ".fits"
        hdu.writeto(noisy_image_filename, overwrite=True)

        for i in range(num_noisy_images):
            # Create and save poisson noisy image.
            rng = np.random.default_rng()
            noisy_image = rng.poisson(image)
            noisy_image = noisy_image / image_exposure_time
            hdu = fits.PrimaryHDU(noisy_image.astype(np.float32))
            noisy_image_filename = (
                output_dir_path + f"Run{i+1}" + output_file_post_fix + ".fits"
            )
            hdu.writeto(noisy_image_filename, overwrite=True)

    def create_image_sequence_number_list(self, numbers: list) -> str:
        number_list = ""
        sorted_numbers = sorted(numbers)
        for num in sorted_numbers:
            if not number_list:
                first_in_sequence = num
                last_number = num
                number_list += str(last_number)
            else:
                if num == (last_number + 1):
                    last_number = num
                elif num > (last_number + 1):
                    if last_number == (first_in_sequence + 1):
                        number_list += f", {last_number}"
                    elif last_number > (first_in_sequence + 1):
                        number_list += f"-{last_number}"
                    first_in_sequence = num
                    last_number = num
                    number_list += f", {last_number}"
                else:
                    first_in_sequence = num
                    last_number = num
                    number_list += f", {last_number}"
        if number_list:
            if last_number == (first_in_sequence + 1):
                number_list += f", {last_number}"
            elif last_number > (first_in_sequence + 1):
                number_list += f"-{last_number}"
        return number_list

    def create_image_list(
        self, input_dir: str, image_sequence_number_list: list
    ) -> list:
        image_list = []
        os.chdir(input_dir)
        for file in glob.glob("*.fits"):
            with fits.open(file) as image_hdul:
                img_seq_num = image_hdul[0].header["IMG_ISN"]
                if img_seq_num in image_sequence_number_list:
                    image_list.append(file)
        return image_list
