import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from astropy.io import fits


@dataclass(order=True)
class Inversion:
    '''
    Inversion for overlap-a-gram data.

    Attributes
    ----------
    pixel_fov_width: np.float32
        Pixel field-of-view width in arcsecs.
    solution_fov_width: np.float32
        Solution field-of-view width in arcsecs.  Should be a multiple of pixel_fov_width.
    slit_fov_width: np.float32
        Slit field-of-view width in arcsecs.
    rsp_dep_name: str
        Response dependence name (e.g. 'ion' or 'logt').
    rsp_dep_list: list
        List of dependence items.  Each item is in the response filename.
    rsp_dep_file_fmt: str
        Path including format of filename (e.g. 'logt_{:.1f}.txt').
    rsp_dep_desc_fmt: str, optional
        FITS binary table column format (e.g. '10A' or '1E').  The default is "" and the format is autogenererated.
    smooth_over: str, optional
        Inversion smoothing (i.e. 'spatial' or 'dependence').  The default is 'spatial'.

    Returns
    -------
    None.

    '''
    pixel_fov_width: np.float32
    solution_fov_width: np.float32
    slit_fov_width: np.float32
    rsp_dep_name: str
    rsp_dep_list: list
    rsp_dep_file_fmt: str
    rsp_dep_desc_fmt: str = ''
    smooth_over: str = 'spatial'
    def __post_init__(self):
        # Calculate number of slits
        calc_num_slits = divmod(self.slit_fov_width, self.solution_fov_width)
        self.num_slits = int(calc_num_slits[0])
        if calc_num_slits[1] > 0.0:
            self.num_slits += 1
        if self.num_slits % 2 == 0.0:
            self.num_slits += 1
        #print("number slits =", self.num_slits)
        self.half_slits = divmod(self.num_slits, 2)
        #print("half slits =", self.half_slits)
        # calc_shift_width = divmod(self.solution_fov_width, self.pixel_fov_width)
        # self.slit_shift_width = int(round(calc_shift_width[0]))
        self.slit_shift_width = int(round(self.solution_fov_width / self.pixel_fov_width))
        #print("slit shift width =", self.slit_shift_width)

        self.image_height = 0
        self.image_width = 0
        # Read response files and create response matrix
        response_files = [self.rsp_dep_file_fmt.format(i) for i in self.rsp_dep_list]
        #print("Response files =", response_files)
        self.num_response_files = len(response_files)
        assert(self.num_response_files > 0)
        #print("num rsp files =", self.num_response_files)
        response_count = 0
        for index in range(len(response_files)):
            # Read file
            dep_em_data = pd.read_csv(response_files[index], delim_whitespace=True)
            if index == 0:
                self.pixels = dep_em_data.iloc[:, 0].values
                self.wavelengths = dep_em_data.iloc[:, 1].values
                self.wavelength_width = len(self.wavelengths)
                self.response_function = np.zeros((self.num_response_files * self.num_slits, self.wavelength_width), dtype=np.float32)
            em = dep_em_data.iloc[:, 2].values
            # TEMPORARY PATCH!!!
            em[-1] = 0.0
            if self.smooth_over == 'dependence':
                # Smooth over dependence.
                slit_count = 0
                for slit_num in range(-self.half_slits[0], self.half_slits[0] + 1):
                    slit_shift = slit_num * self.slit_shift_width
                    if slit_shift < 0:
                        slit_em = np.pad(em, (0, -slit_shift), mode='constant')[-slit_shift:]
                    elif slit_shift > 0:
                        slit_em = np.pad(em, (slit_shift, 0), mode='constant')[:-slit_shift]
                    else:
                        slit_em = em
                    self.response_function[(self.num_response_files * slit_count) + response_count, :] = slit_em
                    slit_count += 1
                response_count += 1
            else:
                self.smooth_over = 'spatial'
                # Smooth over spatial.
                for slit_num in range(-self.half_slits[0], self.half_slits[0] + 1):
                    slit_shift = slit_num * self.slit_shift_width
                    if slit_shift < 0:
                        slit_em = np.pad(em, (0, -slit_shift), mode='constant')[-slit_shift:]
                    elif slit_shift > 0:
                        slit_em = np.pad(em, (slit_shift, 0), mode='constant')[:-slit_shift]
                    else:
                        slit_em = em
                    self.response_function[response_count, :] = slit_em
                    response_count += 1

        #print("response count =", response_count)
        self.response_function = self.response_function.transpose()

        if self.rsp_dep_desc_fmt == '':
            max_dep_len = len(max(self.rsp_dep_list, key=len))
            self.rsp_dep_desc_fmt = str(max_dep_len) + 'A'

    def initialize_input_data(self, input_image: str, image_mask: str = None):
        '''
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

        '''
        # Read image
        image_hdul = fits.open(input_image)
        image_height, image_width = np.shape(image_hdul[0].data)
        print(image_height, image_width)
        # Verify image width equals number of wavelengths in dependence files.
        assert image_width == self.wavelength_width
        self.image = image_hdul[0].data
        self.image_header = image_hdul[0].header
        #print("image (h, w) =", image_height, image_width)
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
                #print("mask (h, w) =", mask_height, mask_width)
                assert image_height == mask_height and image_width == mask_width and self.wavelength_width == self.image_width
        else:
            #self.image_mask = np.ones((image_height, image_width), dtype=np.float32)
            self.image_mask = None

    def invert(self, model, output_dir: str,
               output_file_prefix: str = '',
               output_file_postfix: str = ''):
        '''
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

        '''
        # Verify input data has been initialized.
        assert self.image_width != 0 and self.image_height != 0
        em_data_cube = np.zeros((self.image_height, self.num_slits, self.num_response_files), dtype=np.float32)
        inverted_data = np.zeros((self.image_height, self.image_width), dtype=np.float32)
        for image_row_number in range(self.image_height):
            if (image_row_number % 10 == 0):
                print("image row number =", image_row_number)
            image_row = self.image[image_row_number,:]
            masked_rsp_func = self.response_function
            if self.image_mask is not None:
                mask_row = self.image_mask[image_row_number,:]
                mask_pixels = np.where(mask_row == 0)
                if len(mask_pixels) > 0:
                    image_row[mask_pixels] = 0
                    masked_rsp_func = self.response_function.copy()
                    masked_rsp_func[mask_pixels, :] = 0.0
            # # If image has zero pixel values, zero out corresponding response function pixels.
            # zero_image_pixels = np.where(image_row == 0.0)
            # if len(zero_image_pixels) > 0:
            #     masked_rsp_func = masked_rsp_func.copy()
            #     masked_rsp_func[zero_image_pixels, :] = 0.0

            # masked_rsp_func2 = preprocessing.MinMaxScaler().fit_transform(masked_rsp_func)
            # em, data_out = model.invert(masked_rsp_func2, image_row)
            em, data_out = model.invert(masked_rsp_func, image_row)

            for slit_num in range(self.num_slits):
                if self.smooth_over == 'dependence':
                    slit_em = em[slit_num * self.num_response_files:(slit_num + 1) * self.num_response_files]
                else:
                    slit_em = em[slit_num::self.num_slits]
                em_data_cube[image_row_number, slit_num, :] = slit_em

            inverted_data[image_row_number, :] = data_out

        # Create output directory.
        os.makedirs(output_dir, exist_ok=True)

        # Save EM data cube.
        base_filename = output_file_prefix
        if len(output_file_prefix) > 0 and output_file_prefix[-1] != '_':
            base_filename += '_'
        base_filename += 'em_data_cube'
        if len(output_file_postfix) > 0 and output_file_postfix[0] != '_':
            base_filename += '_'
        base_filename += output_file_postfix
        em_data_cube_file = output_dir + base_filename + '.fits'
        # Transpose data (wavelength, y, x).  Readable by ImageJ.
        em_data_cube = np.transpose(em_data_cube, axes=(2, 0, 1))
        em_data_cube_header = self.image_header.copy()
        self.__add_fits_keywords(em_data_cube_header)
        model.add_fits_keywords(em_data_cube_header)
        hdu = fits.PrimaryHDU(data = em_data_cube, header = em_data_cube_header)
        index = np.arange(len(self.rsp_dep_list))
        # Add binary table.
        col1 = fits.Column(name='index', format='1I', array=index)
        col2 = fits.Column(name=self.rsp_dep_name, format=self.rsp_dep_desc_fmt, array=self.rsp_dep_list)
        table_hdu = fits.BinTableHDU.from_columns([col1, col2])
        hdulist = fits.HDUList([hdu, table_hdu])
        hdulist.writeto(em_data_cube_file, overwrite=True)

        # Save model predicted data.
        base_filename = output_file_prefix
        if len(output_file_prefix) > 0 and output_file_prefix[-1] != '_':
            base_filename += '_'
        base_filename += 'model_predicted_data'
        if len(output_file_postfix) > 0 and output_file_postfix[0] != '_':
            base_filename += '_'
        base_filename += output_file_postfix
        data_file = output_dir + base_filename + ".fits"
        model_predicted_data_header = self.image_header.copy()
        self.__add_fits_keywords(model_predicted_data_header)
        model.add_fits_keywords(model_predicted_data_header)
        hdu = fits.PrimaryHDU(data = inverted_data, header = model_predicted_data_header)
        # Add binary table.
        col1 = fits.Column(name='pixel', format='1I', array=self.pixels)
        col2 = fits.Column(name='wavelength', format='1E', array=self.wavelengths)
        table_hdu = fits.BinTableHDU.from_columns([col1, col2])
        hdulist = fits.HDUList([hdu, table_hdu])
        hdulist.writeto(data_file, overwrite=True)

    def __add_fits_keywords(self, header):
        '''
        Add FITS keywords to FITS header.

        Parameters
        ----------
        header : class 'astropy.io.fits.hdu.image.PrimaryHDU'.
            FITS header.

        Returns
        -------
        None.

        '''
        header.append(('INPUTIMG', self.input_image, 'Input Image'), end=True)
        header.append(('PIXELFOV', self.pixel_fov_width, 'Pixel FOV Width'), end=True)
        header.append(('SLTNFOV', self.solution_fov_width, 'Solution FOV Width'), end=True)
        header.append(('SLITFOV', self.slit_fov_width, 'Slit FOV Width'), end=True)
        header.append(('DEPNAME', self.rsp_dep_name, 'Dependence Name'), end=True)
        header.append(('SMTHOVER', self.smooth_over, 'Smooth Over'), end=True)
