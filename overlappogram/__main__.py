import os

NUMPY_TREAD_COUNT = 1
os.environ["OMP_NUM_THREADS"] = str(NUMPY_TREAD_COUNT)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUMPY_TREAD_COUNT)
os.environ["MKL_NUM_THREADS"] = str(NUMPY_TREAD_COUNT)

import argparse  # noqa: E402
import time  # noqa: E402

import toml  # noqa: E402

from overlappogram.inversion import Inverter  # noqa: E402

parser = argparse.ArgumentParser(description="Inverts overlappograms")

parser.add_argument("config", help="path to the toml configuration")
args = parser.parse_args()

with open(args.config) as f:
    config = toml.load(f)

inversion = Inverter(
    rsp_func_cube_file=config["paths"]["response"],
    rsp_dep_name=config["inversion"]["response_dependency_name"],
    rsp_dep_list=config["inversion"]["response_dependency_list"],
    solution_fov_width=config["inversion"]["solution_fov_width"],
    smooth_over=config["inversion"]["smooth_over"],
    field_angle_range=config["inversion"]["field_angle_range"],
    detector_row_range=config["inversion"]["detector_row_range"],
)

inversion.initialize_input_data(config['paths']['overlappogram'], None, config["paths"]["weights"])

em_cube_paths = []
for alpha in config["model"]["alphas"]:
    for rho in config["model"]["rhos"]:
        start = time.time()

        postfix = (
                "x"
                + str(config["inversion"]["solution_fov_width"])
                + "_"
                + str(rho * 10)
                + "_"
                + str(alpha)
                + "_wpsf"
        )
        # todo : handle IO since invert returns a tuple not the paths
        em_cube_paths.append(
            inversion.invert(
                config["model"],
                alpha,
                rho,
                num_threads=config["execution"]["num_threads"],
                mode_switch_thread_count=config['execution']['mode_switch_thread_count']
            )
        )

        end = time.time()
        print("Inversion Time =", end - start)

# TODO: enable IO again
# if config["output"]["make_spectral"]:
#     mdp = MaGIXSDataProducts()
#     mdp.create_level2_0_spectrally_pure_images(
#         em_cube_paths,
#         config["paths"]["gnt"],
#         config["inversion"]["response_dependency_list"],
#         config["output"]["directory"],
#     )
