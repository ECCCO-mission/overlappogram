import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

output_dir = "output/photons/plots/"
ground_truth_path = "output/photons/ground_truth_spectrally_pure_data_cube_reshaped.fits"
dep_ref = "data/ECCCO_speedtest_runs/eccco_sp_puremaps_psf4pix_inelectrons_cm3perspersr_with_tables.fits"
output_paths = ["output/photons/combined_ECCCO_trim_sw_lw_s_i_scaled_spectrally_pure_data_cube_x2_1.0_0.1_wpsf.fits",
                "output/photons/combined_ECCCO_trim_sw_lw_s_i_scaled_spectrally_pure_data_cube_x2_1.0_0.01_wpsf.fits",
                "output/photons/combined_ECCCO_trim_sw_lw_s_i_scaled_spectrally_pure_data_cube_x2_1.0_0.2_wpsf.fits",
                "output/photons/combined_ECCCO_trim_sw_lw_s_i_scaled_spectrally_pure_data_cube_x2_1.0_0.005_wpsf.fits"]

with fits.open(dep_ref) as hdul:
    dep_table = hdul[2].data

for output_path in output_paths:
    short_path = os.path.splitext(os.path.basename(output_path))[0]
    ground_truth = fits.getdata(ground_truth_path)
    output_data = fits.getdata(output_path)

    for i in range(len(dep_table)):
        gt = np.flipud(ground_truth[i].T)
        gt = gaussian_filter(gt, 4 / (2 * np.sqrt(2 * np.log(2))))

        vmin, vmax = 0, np.nanpercentile(ground_truth[i], 99.9)
        interval = vmax * 0.5

        # initial image comparison
        fig, axs = plt.subplots(ncols=3, figsize=(30, 10))
        im1 = axs[0].imshow(gt, vmin=vmin, vmax=vmax, origin='lower')
        axs[0].set_title("Ground truth")
        axs[1].imshow(output_data[i], vmin=vmin, vmax=vmax, origin='lower')
        axs[1].set_title("Output")
        im2 = axs[2].imshow(gt - output_data[i],
                            origin='lower', cmap='seismic', vmin=-interval, vmax=interval)
        axs[2].set_title("Ground truth - output")
        for ax in axs:
            ax.set_aspect(0.5)
        fig.colorbar(im1, ax=axs[:2])
        fig.colorbar(im2, ax=axs[2])
        fig.suptitle(dep_table[i][1])
        fig.savefig(output_dir + f"spectral_pure_{dep_table[i][1]}_{short_path}.png")
        plt.close()

        # scatter plots
        fig, ax = plt.subplots()
        ax.plot(gt.flatten(), output_data[i].flatten(), '.', ms=1)
        ax.plot([0, np.max(ground_truth[i])], [0, np.max(ground_truth[i])], 'r-')
        ax.set_title(dep_table[i][1])
        ax.set_xlabel("Ground truth")
        ax.set_ylabel("Unfolded recreation")
        fig.savefig(output_dir + f"scatter_{dep_table[i][1]}_{short_path}.png")
        plt.close()

        # histogram
        Z, xedges, yedges = np.histogram2d(gt.flatten(), output_data[i].flatten(), 50)
        fig, ax = plt.subplots()
        im = ax.pcolormesh(xedges, yedges, np.log10(Z.T + 1E-3))
        ax.plot([0, np.max(ground_truth[i])], [0, np.max(ground_truth[i])], 'r-')
        ax.set_title(dep_table[i][1])
        fig.colorbar(im, ax=ax)
        fig.savefig(output_dir + f"histogram_{dep_table[i][1]}_{short_path}.png")
        plt.close()
