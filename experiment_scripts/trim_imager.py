# remove the imager from the response, weights, and overlappogram
from astropy.io import fits

response = (
    "data/D16Feb2024_eccco_response_feldman_m_el_with_tables_s_i_lw_coopersun.fits"
)
weights = "data/eccco_is_lw_forwardmodel_sample_weights_psf4pix_el.fits"
image = "data/eccco_is_lw_forwardmodel_thermal_response_psf4pix_el.fits"


response_hdul = fits.open(response)
image_hdul = fits.open(image)
weights_hdul = fits.open(weights)

response_hdul_img_norm = response_hdul.copy()
response_hdul_img_norm[0].data = response_hdul[0].data[:, :, :4096]
response_hdul_img_norm.writeto("data/response_only_spectra.fits", overwrite=True)
fits.writeto(
    "data/forward_model_only_spectra.fits",
    image_hdul[0].data[:, :4096],
    header=image_hdul[0].header,
    overwrite=True,
)
fits.writeto(
    "data/weights_only_spectra.fits",
    weights_hdul[0].data[:, :4096],
    header=weights_hdul[0].header,
    overwrite=True,
)
