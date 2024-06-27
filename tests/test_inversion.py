from __future__ import annotations

import os

import numpy as np
import pytest
import toml
from ndcube import NDCube

from overlappogram.error import InvalidInversionModeError
from overlappogram.inversion import InversionMode, Inverter
from overlappogram.io import (load_overlappogram, load_response_cube,
                              save_em_cube, save_prediction)

TEST_PATH = os.path.dirname(os.path.realpath(__file__))


def test_create_inverter():
    response_path = os.path.join(TEST_PATH, "test_response.fits")
    response = load_response_cube(response_path)
    inverter = Inverter(response)
    assert isinstance(inverter, Inverter)
    assert inverter.is_inverted == False  # noqa E712, acceptable for a test


@pytest.mark.parametrize("inversion_mode", [InversionMode.ROW, InversionMode.CHUNKED, InversionMode.HYBRID])
@pytest.mark.parametrize("is_weighted", [True, False])
def test_inversion_runs(tmp_path, inversion_mode, is_weighted):
    response_path = os.path.join(TEST_PATH, "test_response.fits")
    overlappogram_path = os.path.join(TEST_PATH, "test_overlappogram.fits")
    weights_path = os.path.join(TEST_PATH, "test_weights.fits") if is_weighted else None
    config_path = os.path.join(TEST_PATH, "test_config.toml")

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
        mode=inversion_mode
    )

    assert isinstance(em_cube, NDCube)
    assert isinstance(prediction, NDCube)
    assert isinstance(scores, np.ndarray)
    assert isinstance(unconverged_rows, list)

    save_em_cube(em_cube, str(tmp_path / "em.fits"))
    save_prediction(prediction, str(tmp_path / "prediction.fits"))

    assert os.path.isfile(tmp_path / "em.fits")
    assert os.path.isfile(tmp_path / "prediction.fits")


def test_inversion_invalid_inversion_mode_fails():
    response_path = os.path.join(TEST_PATH, "test_response.fits")
    overlappogram_path = os.path.join(TEST_PATH, "test_overlappogram.fits")
    weights_path = os.path.join(TEST_PATH, "test_weights.fits")
    config_path = os.path.join(TEST_PATH, "test_config.toml")

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

    with pytest.raises(InvalidInversionModeError):
        em_cube, prediction, scores, unconverged_rows = inversion.invert(
            overlappogram,
            config["model"],
            3E-5,
            0.1,
            num_threads=config["execution"]["num_threads"],
            mode_switch_thread_count=config["execution"]["mode_switch_thread_count"],
            mode="row"
        )
