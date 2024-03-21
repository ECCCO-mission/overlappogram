from astropy.io import fits
from astropy.nddata import StdDevUncertainty
from astropy.wcs import WCS
from ndcube import NDCube

__all__ = ["load_overlappogram", "load_response_cube", "save_em_cube", "save_spectral_cube", "save_prediction"]


def load_overlappogram(image_path, weights_path) -> NDCube:
    with fits.open(image_path) as image_hdul:
        image = image_hdul[0].data
        header = image_hdul[0].header
        wcs = WCS(image_hdul[0].header)
    with fits.open(weights_path) as weights_hdul:
        weights = weights_hdul[0].data
    return NDCube(image, wcs=wcs, uncertainty=StdDevUncertainty(1 / weights), meta=dict(header))


def load_response_cube(path) -> NDCube:
    with fits.open(path) as hdul:
        response = hdul[0].data
        header = hdul[0].header
        wcs = WCS(hdul[0].header)
        temperatures = hdul[1].data
        field_angles = hdul[2].data
    meta = dict(header)
    meta.update({"temperatures": temperatures, "field_angles": field_angles})
    return NDCube(response, wcs=wcs, meta=meta)


def save_em_cube(cube, path, overwrite=True) -> None:
    fits.writeto(path, cube, overwrite=overwrite)


def save_prediction(prediction, path, overwrite=True) -> None:
    fits.writeto(path, prediction, overwrite=overwrite)


def save_spectral_cube(spectral_cube, path, overwrite=True) -> None:
    fits.writeto(path, spectral_cube, overwrite=overwrite)
