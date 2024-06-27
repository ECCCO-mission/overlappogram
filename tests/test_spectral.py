from __future__ import annotations

import os

import toml
from ndcube import NDCube

from overlappogram.inversion import MODE_MAPPING, Inverter
from overlappogram.io import (load_overlappogram, load_response_cube,
                              save_spectral_cube)
from overlappogram.spectral import create_spectrally_pure_images

TEST_PATH = os.path.dirname(os.path.realpath(__file__))


def test_create_spectrally_pure_images(tmp_path):
    response_path = os.path.join(TEST_PATH, "test_response.fits")
    overlappogram_path = os.path.join(TEST_PATH, "test_overlappogram.fits")
    weights_path = os.path.join(TEST_PATH, "test_weights.fits")
    config_path = os.path.join(TEST_PATH, "test_config.toml")
    gnt_path = os.path.join(TEST_PATH, "test_gnt.fits")

    response = load_response_cube(response_path)
    overlappogram = load_overlappogram(overlappogram_path, weights_path)
    with open(config_path) as f:
        config = toml.load(f)

    inversion = Inverter(
        response,
        solution_fov_width=config["inversion"]["solution_fov_width"],
        response_dependency_list=config["inversion"]["response_dependency_list"],
        smooth_over=config["inversion"]["smooth_over"],
        field_angle_range=config["inversion"]["field_angle_range"],
        detector_row_range=config["inversion"]["detector_row_range"],
    )

    em_cube, prediction, scores, unconverged_rows = inversion.invert(
        overlappogram,
        config["model"],
        3E-5,
        0.1,
        num_threads=config["execution"]["num_threads"],
        mode_switch_thread_count=config["execution"]["mode_switch_thread_count"],
        mode=MODE_MAPPING.get(config['execution']['mode'], 'invalid')
    )

    spectrally_pure_image = create_spectrally_pure_images([em_cube],
                                                          gnt_path,
                                                          config["inversion"]["response_dependency_list"])
    assert isinstance(spectrally_pure_image, NDCube)

    save_spectral_cube(spectrally_pure_image, str(tmp_path / "spectrally_pure.fits"))

    assert os.path.isfile(tmp_path / "spectrally_pure.fits")


def test_create_spectrally_pure_images_dep_none():
    """why does this fail"""
    pass
