import numpy as np
from astropy.io import fits


def create_poisson_noisy_image(
    image: np.ndarray, output_image_file: str, exposure_time: float
):
    """


    Parameters
    ----------
    image : np.ndarray
        Original image.
    output_image_file : str
        Output file for noisy image.
    exposure_time : float
        Exposure time in seconds.

    Returns
    -------
    None.

    """
    # Create and save poisson noisy image.
    rng = np.random.default_rng()
    noisy_image = rng.poisson(image * exposure_time) / exposure_time
    hdu = fits.PrimaryHDU(noisy_image)
    hdu.header.append(("EXPTIME", noisy_image, "Exposure Time (seconds)"), end=True)
    hdu.writeto(output_image_file, overwrite=True)
