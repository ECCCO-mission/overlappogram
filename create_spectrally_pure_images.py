import numpy as np

from magixs_data_products import MaGIXSDataProducts

dir_path = "output/full_data_fewer_temps/"

mdp = MaGIXSDataProducts()
image_list = ["output/full_data_fewer_temps/combined_ECCCO_sw_lw_s_i_scaled_em_data_cube_x2_1.0_0.01_wpsf.fits"]
gnt_file = "data/ECCCO_speedtest_runs/master_gnt_eccco_inelectrons_cm3perspersr_with_tables.fits"

rsp_dep_list = np.round((np.arange(56, 68, 1) / 10.0), decimals=1)
mdp.create_level2_0_spectrally_pure_images(image_list, gnt_file, rsp_dep_list, dir_path)
