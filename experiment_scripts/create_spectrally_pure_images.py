import numpy as np

from magixs_data_products import MaGIXSDataProducts

dir_path = "output/test/"

mdp = MaGIXSDataProducts()
# image_list = ["output/photons/combined_ECCCO_trim_sw_lw_s_i_scaled_em_data_cube_x2_1.0_0.2_wpsf.fits",
#               "output/photons/combined_ECCCO_trim_sw_lw_s_i_scaled_em_data_cube_x2_1.0_0.1_wpsf.fits",
#               "output/photons/combined_ECCCO_trim_sw_lw_s_i_scaled_em_data_cube_x2_1.0_0.01_wpsf.fits",
#               "output/photons/combined_ECCCO_trim_sw_lw_s_i_scaled_em_data_cube_x2_1.0_0.005_wpsf.fits"]
#image_list = ["/Users/jhughes/Desktop/ECCCO_unfolding_share/output/combined_ECCCO_trim_sw_lw_s_i_scaled_em_data_cube_x2_1.0_0.2_wpsf.fits"]
image_list = ["data/ECCCO_speedtest_runs/combined_ECCCO_sw_lw_s_i_scaled_em_data_cube_x2_1.0_0.005_wpsf.fits"]
# gnt_file = "data/ECCCO_speedtest_runs/master_gnt_eccco_inelectrons_cm3perspersr_with_tables.fits"
gnt_file = "data/ECCCO_speedtest_runs/master_gnt_eccco_inphotons_cm3persperpix_with_tables.fits"

rsp_dep_list = [5.7, 5.8, 5.9, 6.0 , 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7, 6.8] #np.round((np.arange(57, 68+1, 1) / 10.0), decimals=1)
mdp.create_level2_0_spectrally_pure_images(image_list, gnt_file, rsp_dep_list, dir_path)
