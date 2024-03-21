import os

NUMPY_TREAD_COUNT = 1
os.environ["OMP_NUM_THREADS"] = str(NUMPY_TREAD_COUNT)
os.environ["OPENBLAS_NUM_THREADS"] = str(NUMPY_TREAD_COUNT)
os.environ["MKL_NUM_THREADS"] = str(NUMPY_TREAD_COUNT)

import time  # noqa: E402

import click  # noqa: E402
import toml  # noqa: E402

from overlappogram.inversion import Inverter  # noqa: E402
from overlappogram.io import load_overlappogram  # noqa: E402
from overlappogram.io import load_response_cube  # noqa: E402


@click.command()
@click.argument("config")
def unfold(config):
    """ Unfold an overlappogram given a configuration toml file."""  # TODO improve message

    with open(config) as f:
        config = toml.load(f)

    os.makedirs(config['output']['directory'], exist_ok=True)  # make sure output directory exists

    overlappogram = load_overlappogram(config['paths']['overlappogram'],  config['paths']['weights'])
    response_cube = load_response_cube(config['paths']['response'])

    inversion = Inverter(
        response_cube,
        solution_fov_width=config["inversion"]["solution_fov_width"],
        smooth_over=config["inversion"]["smooth_over"],
        field_angle_range=config["inversion"]["field_angle_range"],
        detector_row_range=config["inversion"]["detector_row_range"],
    )

    for alpha in config["model"]["alphas"]:
        for rho in config["model"]["rhos"]:
            start = time.time()
            inversion.invert(
                    overlappogram,
                    config["model"],
                    alpha,
                    rho,
                    num_threads=config["execution"]["num_threads"],
                    mode_switch_thread_count=config['execution']['mode_switch_thread_count']
                )
            end = time.time()
            print(f"Inversion Time for alpha={alpha}, rho={rho}:", end - start)

            # TODO write out results
            # postfix = (
            #         "x"
            #         + str(config["inversion"]["solution_fov_width"])
            #         + "_"
            #         + str(rho * 10)
            #         + "_"
            #         + str(alpha)
            #         + "_wpsf"
            # )

            if config['output']['make_spectral']:
                pass
                #spectral_images = create_spectrally_pure_images(overlappogram, None, None, None)
                # TODO write out spectral images
