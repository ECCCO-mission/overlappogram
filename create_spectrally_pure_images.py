import numpy as np

from magixs_data_products import MaGIXSDataProducts

dir_path = "output/photons/"

mdp = MaGIXSDataProducts()
image_list = ["output/photons/combined_ECCCO_trim_sw_lw_s_i_scaled_em_data_cube_x2_1.0_0.2_wpsf.fits",
              "output/photons/combined_ECCCO_trim_sw_lw_s_i_scaled_em_data_cube_x2_1.0_0.1_wpsf.fits",
              "output/photons/combined_ECCCO_trim_sw_lw_s_i_scaled_em_data_cube_x2_1.0_0.01_wpsf.fits",
              "output/photons/combined_ECCCO_trim_sw_lw_s_i_scaled_em_data_cube_x2_1.0_0.005_wpsf.fits"]
gnt_file = "data/ECCCO_speedtest_runs/master_gnt_eccco_inelectrons_cm3perspersr_with_tables.fits"
gnt_file = "data/ECCCO_speedtest_runs/master_gnt_eccco_inphotons_cm3persperpix_with_tables.fits"

rsp_dep_list = np.round((np.arange(57, 68+1, 1) / 10.0), decimals=1)
mdp.create_level2_0_spectrally_pure_images(image_list, gnt_file, rsp_dep_list, dir_path)
