import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from astropy.io import fits
from sklearn.linear_model import LassoLarsCV


@dataclass(order=True)
class Train:
    """
    Inversion for overlap-a-gram data.

    Attributes
    ----------
    pixel_fov_width: np.float64
        Pixel field-of-view width in arcsecs.
    solution_fov_width: np.float64
        Solution field-of-view width in arcsecs.  Should be a multiple of pixel_fov_width.
    slit_fov_width: np.float64
        Slit field-of-view width in arcsecs.
    rsp_dep_name: str
        Response dependence name (e.g. 'ion' or 'logt').
    rsp_dep_list: str
        List of dependence items.  Each item is in the response filename.
    rsp_dep_file_fmt: str
        Path including format of filename (e.g. 'logt_{:.1f}.txt').
    rsp_dep_desc_fmt: str, optional
        FITS binary table column format (e.g. '10A' or '1E').  THe default is "" and the format is autogenererated.
    smooth_over: str, optional
        Inversion smoothing (i.e. 'spatial' or 'dependence').  THe default is 'spatial'.
    image_mask : str, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    """

    pixel_fov_width: np.float64
    solution_fov_width: np.float64
    slit_fov_width: np.float64
    rsp_dep_name: str
    rsp_dep_list: list
    rsp_dep_file_fmt: str
    rsp_dep_desc_fmt: str = ""
    smooth_over: str = "spatial"

    def __post_init__(self):
        # Calculate number of slits
        calc_num_slits = divmod(self.slit_fov_width, self.solution_fov_width)
        self.num_slits = int(calc_num_slits[0])
        if calc_num_slits[1] > 0.0:
            self.num_slits += 1
        if self.num_slits % 2 == 0.0:
            self.num_slits += 1
        # print("number slits =", self.num_slits)
        self.half_slits = divmod(self.num_slits, 2)
        # print("half slits =", self.half_slits)
        # calc_shift_width = divmod(self.solution_fov_width, self.pixel_fov_width)
        # self.slit_shift_width = int(round(calc_shift_width[0]))
        self.slit_shift_width = int(
            round(self.solution_fov_width / self.pixel_fov_width)
        )
        # print("slit shift width =", self.slit_shift_width)

        self.image_height = 0
        self.image_width = 0
        # Read response files and create response matrix
        response_files = [self.rsp_dep_file_fmt.format(i) for i in self.rsp_dep_list]
        # print("Response files =", response_files)
        self.num_response_files = len(response_files)
        assert self.num_response_files > 0
        # print("num rsp files =", self.num_response_files)
        self.groups = np.zeros(self.num_slits * self.num_response_files, dtype=int)
        response_count = 0
        for index in range(len(response_files)):
            # Read file
            dep_em_data = pd.read_csv(response_files[index], delim_whitespace=True)
            if index == 0:
                self.pixels = dep_em_data.iloc[:, 0].values
                self.wavelengths = dep_em_data.iloc[:, 1].values
                self.wavelength_width = len(self.wavelengths)
                self.response_function = np.zeros(
                    (self.num_response_files * self.num_slits, self.wavelength_width)
                )
            em = dep_em_data.iloc[:, 2].values
            if self.smooth_over == "dependence":
                # Smooth over dependence.
                slit_count = 0
                for slit_num in range(-self.half_slits[0], self.half_slits[0] + 1):
                    slit_shift = slit_num * self.slit_shift_width
                    if slit_shift < 0:
                        slit_em = np.pad(em, (0, -slit_shift), mode="constant")[
                            -slit_shift:
                        ]
                    elif slit_shift > 0:
                        slit_em = np.pad(em, (slit_shift, 0), mode="constant")[
                            :-slit_shift
                        ]
                    else:
                        slit_em = em
                    self.response_function[
                        (self.num_response_files * slit_count) + response_count, :
                    ] = slit_em
                    self.groups[
                        (self.num_response_files * slit_count) + response_count
                    ] = index
                    slit_count += 1
                response_count += 1
            else:
                self.smooth_over = "spatial"
                # Smooth over spatial.
                for slit_num in range(-self.half_slits[0], self.half_slits[0] + 1):
                    slit_shift = slit_num * self.slit_shift_width
                    if slit_shift < 0:
                        slit_em = np.pad(em, (0, -slit_shift), mode="constant")[
                            -slit_shift:
                        ]
                    elif slit_shift > 0:
                        slit_em = np.pad(em, (slit_shift, 0), mode="constant")[
                            :-slit_shift
                        ]
                    else:
                        slit_em = em
                    self.response_function[response_count, :] = slit_em
                    self.groups[response_count] = index
                    response_count += 1

        print("groups =", self.groups)
        # print("response count =", response_count)
        self.response_function = self.response_function.transpose()

        if self.rsp_dep_desc_fmt == "":
            max_dep_len = len(max(self.rsp_dep_list, key=len))
            self.rsp_dep_desc_fmt = str(max_dep_len) + "A"

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
        image_height, image_width = np.shape(image_hdul[0].data)
        # Verify image width equals number of wavelengths in dependence files.
        assert image_width == self.wavelength_width
        self.image = image_hdul[0].data
        # print("image (h, w) =", image_height, image_width)
        self.image_width = image_width
        self.image_height = image_height
        self.input_image = os.path.basename(input_image)

        if image_mask is not None:
            # Read mask
            mask_hdul = fits.open(image_mask)
            mask_height, mask_width = np.shape(mask_hdul[0].data)
            self.image_mask = mask_hdul[0].data
            if len(np.where(image_mask == 0)) == 0:
                self.image_mask = None
        else:
            # self.image_mask = np.ones((image_height, image_width))
            self.image_mask = None

    def invert(self):
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

        Returns
        -------
        None.

        """
        # for image_row_number in range(self.image_height):
        # for image_row_number in range(180, 184):
        for image_row_number in range(240, 250):
            # for image_row_number in range(30, 34):
            # for image_row_number in range(530, 534):
            # if (image_row_number % 10 == 0):
            #     print("image row number =", image_row_number)
            print("image row number =", image_row_number)
            image_row = self.image[image_row_number, :]
            # masked_rsp_func = self.response_function
            # if self.image_mask is not None:
            #     mask_row = self.image_mask[image_row_number,:]
            #     mask_pixels = np.where(mask_row == 0)
            #     if len(mask_pixels) > 0:
            #         image_row[mask_pixels] = 0
            #         masked_rsp_func = self.response_function.copy()
            #         masked_rsp_func[mask_pixels, :] = 0

            # alphas = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 0.1]
            # alphas = np.arange(0.1, 1.0, 0.1)
            # ratios = [0.05, 0.1, 0.15, 0.2]
            model = LassoLarsCV(
                max_iter=50000,
                precompute=True,
                normalize=True,
                positive=True,
                fit_intercept=True,
            )
            model.fit(self.response_function, image_row)
            # print("model =", model)
            print("alpha: %f" % model.alpha_)
            # print('l1_ratio: %f' % model.l1_ratio_)
            print("intercept: %f" % model.intercept_)
            print("n_iter: %f" % model.n_iter_)

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
        header.append(("INPUTIMG", self.input_image, "Input Image"), end=True)
        header.append(("PIXELFOV", self.pixel_fov_width, "Pixel FOV Width"), end=True)
        header.append(
            ("SLTNFOV", self.solution_fov_width, "Solution FOV Width"), end=True
        )
        header.append(("SLITFOV", self.slit_fov_width, "Slit FOV Width"), end=True)
        header.append(("DEPNAME", self.rsp_dep_name, "Dependence Name"), end=True)
        header.append(("SMTHOVER", self.smooth_over, "Smooth Over"), end=True)
