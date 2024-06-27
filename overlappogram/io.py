from __future__ import annotations

from datetime import datetime

from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube

from .error import InvalidDataFormatError

__all__ = ["load_overlappogram", "load_response_cube", "save_em_cube", "save_spectral_cube", "save_prediction"]


RESPONSE_HEADER_KEYS = ['DATE',
                        'VALUE',
                        'FIELDANG',
                        'RSP_DATE',
                        'DEPNAME',
                        'ABUNDANC',
                        'ELECDIST']


def load_overlappogram(image_path: str,
                       weights_path: str | None = None,
                       mask_path: str | None = None) -> NDCube:
    with fits.open(image_path) as image_hdul:
        image = image_hdul[0].data
        header = image_hdul[0].header
        wcs = WCS(image_hdul[0].header)

    if weights_path is None:
        uncertainty = None
    else:
        with fits.open(weights_path) as weights_hdul:
            weights = weights_hdul[0].data
            uncertainty = StdDevUncertainty(1 / weights)

    if mask_path is None:
        mask = None
    else:
        with fits.open(mask_path) as mask_hdul:
            mask = mask_hdul[0].data

    return NDCube(image, wcs=wcs, uncertainty=uncertainty, mask=mask, meta=dict(header))


def load_response_cube(path: str) -> NDCube:
    with fits.open(path) as hdul:
        response = hdul[0].data
        header = hdul[0].header
        wcs = WCS(hdul[0].header)
        temperatures = hdul[1].data
        field_angles = hdul[2].data
    meta = dict(header)
    meta.update({"temperatures": temperatures, "field_angles": field_angles})
    for key in RESPONSE_HEADER_KEYS:
        meta[key] = header[key]
    return NDCube(response, wcs=wcs, meta=meta)


def save_em_cube(cube: NDCube, path: str, overwrite: bool = True) -> None:
    if "temperatures" not in cube.meta:
        raise InvalidDataFormatError("Temperatures are missing form the EM cube's metadata.")

    hdul = fits.HDUList([fits.PrimaryHDU(cube.data),
                         fits.BinTableHDU(cube.meta['temperatures']),
                         # todo : make sure this is the right temperatures and not just a pass through
                         ])
    for key in RESPONSE_HEADER_KEYS:
        hdul[0].header[key] = cube.meta[key]
    hdul[0].header['DATE'] = datetime.now().isoformat()
    hdul.writeto(path, overwrite=overwrite)


def save_prediction(prediction: NDCube, path: str, overwrite: bool = True) -> None:
    if "temperatures" not in prediction.meta:
        raise InvalidDataFormatError("Temperatures are missing form the prediction cube's metadata.")

    hdul = fits.HDUList([fits.PrimaryHDU(prediction.data),
                         fits.BinTableHDU(prediction.meta['temperatures']),
                         # todo : make sure this is the right temperatures and not just a pass through
                         ])
    for key in RESPONSE_HEADER_KEYS:
        hdul[0].header[key] = prediction.meta[key]
    hdul[0].header['DATE'] = datetime.now().isoformat()
    hdul.writeto(path, overwrite=overwrite)


def save_spectral_cube(spectral_cube: NDCube, path: str, overwrite: bool = True) -> None:
    if "temperatures" not in spectral_cube.meta:
        raise InvalidDataFormatError("Temperatures are missing form the spectral cube's metadata.")

    if "ions" not in spectral_cube.meta:
        raise InvalidDataFormatError("Ions are missing form the spectral cube's metadata.")

    hdul = fits.HDUList([fits.PrimaryHDU(spectral_cube.data),
                         fits.BinTableHDU(spectral_cube.meta['temperatures']),
                         fits.BinTableHDU(spectral_cube.meta['ions']),
                         # todo : make sure this is the right temperatures and not just a pass through
                         ])
    for key in RESPONSE_HEADER_KEYS:
        hdul[0].header[key] = spectral_cube.meta[key]
    hdul[0].header['DATE'] = datetime.now().isoformat()
    hdul.writeto(path, overwrite=overwrite)
