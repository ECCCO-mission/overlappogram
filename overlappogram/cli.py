from __future__ import annotations

import os

NUMPY_THREAD_COUNT = 1
os.environ["OMP_NUM_THREADS"] = str(NUMPY_THREAD_COUNT)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUMPY_THREAD_COUNT)
os.environ["MKL_NUM_THREADS"] = str(NUMPY_THREAD_COUNT)

import time  # noqa: E402

import click  # noqa: E402
import toml  # noqa: E402

from overlappogram.inversion import MODE_MAPPING, Inverter  # noqa: E402
from overlappogram.io import load_overlappogram  # noqa: E402
from overlappogram.io import save_em_cube  # noqa: E402
from overlappogram.io import save_spectral_cube  # noqa: E402
from overlappogram.io import load_response_cube, save_prediction  # noqa: E402
from overlappogram.spectral import create_spectrally_pure_images  # noqa: E402


@click.command()
@click.argument("config")
def unfold(config):
    """Unfold an overlappogram given a configuration toml file.

    See https://eccco-mission.github.io/overlappogram/configuration.html for the configuration file format.
    """

    with open(config) as f:
        config = toml.load(f)

    os.makedirs(config["output"]["directory"], exist_ok=True)  # make sure output directory exists

    overlappogram = load_overlappogram(config["paths"]["overlappogram"],
                                       config["paths"]["weights"] if 'weights' in config['paths'] else None,
                                       config["paths"]["mask"] if 'mask' in config['paths'] else None)
    response_cube = load_response_cube(config["paths"]["response"])

    inversion = Inverter(
        response_cube,
        solution_fov_width=config["inversion"]["solution_fov_width"],
        response_dependency_list=config["inversion"]["response_dependency_list"],
        smooth_over=config["inversion"]["smooth_over"],
        field_angle_range=config["inversion"]["field_angle_range"],
        detector_row_range=config["inversion"]["detector_row_range"],
    )

    for alpha in config["model"]["alphas"]:
        for rho in config["model"]["rhos"]:
            print(80*"-")
            print(f"Beginning inversion for alpha={alpha}, rho={rho}.")
            start = time.time()
            em_cube, prediction, scores, unconverged_rows = inversion.invert(
                    overlappogram,
                    config["model"],
                    alpha,
                    rho,
                    num_threads=config["execution"]["num_threads"],
                    mode_switch_thread_count=config["execution"]["mode_switch_thread_count"],
                    mode=MODE_MAPPING.get(config['execution']['mode'], 'invalid')
            )
            end = time.time()
            print(
                f"Inversion for alpha={alpha}, rho={rho} took",
                int(end - start),
                f"seconds; {len(unconverged_rows)} unconverged rows",
            )

            postfix = (
                "x" + str(config["inversion"]["solution_fov_width"]) + "_" + str(rho * 10) + "_" + str(alpha) + "_wpsf"
            )
            save_em_cube(
                em_cube,
                os.path.join(config["output"]["directory"],
                             f"{config['output']['prefix']}_emcube_{postfix}.fits"),
                config["output"]["overwrite"],
            )

            save_prediction(
                prediction,
                os.path.join(config["output"]["directory"],
                             f"{config['output']['prefix']}_prediction_{postfix}.fits"),
                overwrite=config["output"]["overwrite"],
            )

            scores_path = os.path.join(config["output"]["directory"],
                                       f"{config['output']['prefix']}_scores_{postfix}.txt")
            with open(scores_path, 'w') as f:
                f.write("\n".join(scores.flatten().astype(str).tolist()))

            if config["output"]["make_spectral"]:
                spectral_images = create_spectrally_pure_images(
                    [em_cube], config["paths"]["gnt"], config["inversion"]["response_dependency_list"]
                )
                save_spectral_cube(
                    spectral_images,
                    os.path.join(
                        config["output"]["directory"], f"{config['output']['prefix']}_spectral_{postfix}.fits"
                    ),
                    overwrite=config["output"]["overwrite"],
                )
