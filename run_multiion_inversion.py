import os

NUMPY_TREAD_COUNT = 1
os.environ["OMP_NUM_THREADS"] = str(NUMPY_TREAD_COUNT)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUMPY_TREAD_COUNT)
os.environ["MKL_NUM_THREADS"] = str(NUMPY_TREAD_COUNT)

import argparse  # noqa: E402
import time  # noqa: E402

import toml  # noqa: E402

from magixs_data_products import MaGIXSDataProducts  # noqa: E402
from overlappogram.inversion_field_angles import Inversion  # noqa: E402


def run_inversion(image_path, config: dict):
    inversion = Inversion(
        rsp_func_cube_file=config["paths"]["response"],
        rsp_dep_name=config["inversion"]["response_dependency_name"],
        rsp_dep_list=config["inversion"]["response_dependency_list"],
        solution_fov_width=config["inversion"]["solution_fov_width"],
        smooth_over=config["inversion"]["smooth_over"],
        field_angle_range=config["inversion"]["field_angle_range"],
    )

    inversion.initialize_input_data(image_path, None, config["paths"]["weights"])

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
            em_cube_paths.append(
                inversion.multiprocessing_invert(
                    config["model"],
                    alpha,
                    rho,
                    config["output"]["directory"],
                    num_threads=config["execution"]["num_threads"],
                    output_file_prefix=config["output"]["prefix"],
                    output_file_postfix=postfix,
                    detector_row_range=config["inversion"]["detector_row_range"],
                    score=True,
                )
            )

            end = time.time()
            print("Inversion Time =", end - start)
    return em_cube_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inverts overlappograms")

    parser.add_argument("path")
    parser.add_argument("config")
    args = parser.parse_args()

    with open(args.config) as f:
        config = toml.load(f)

    em_cube_paths = run_inversion(args.path, config)

    if config["output"]["make_spectral"]:
        mdp = MaGIXSDataProducts()
        mdp.create_level2_0_spectrally_pure_images(
            em_cube_paths,
            config["paths"]["gnt"],
            config["inversion"]["response_dependency_list"],
            config["output"]["directory"],
        )
