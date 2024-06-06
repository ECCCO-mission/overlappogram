import os

import numpy as np
from astropy.nddata import StdDevUncertainty
from ndcube import NDCube

from overlappogram.io import load_overlappogram

TEST_PATH = os.path.dirname(os.path.realpath(__file__))


def test_load_overlappogram():
    data_path = os.path.join(TEST_PATH, "test_overlappogram.fits")
    weights_path = os.path.join(TEST_PATH, "test_weights.fits")
    overlappogram = load_overlappogram(data_path, weights_path)
    assert isinstance(overlappogram, NDCube)
    assert isinstance(overlappogram.data, np.ndarray)
    assert isinstance(overlappogram.uncertainty, StdDevUncertainty)
    assert isinstance(overlappogram.uncertainty.array, np.ndarray)


def test_load_overlappogram_without_weights():
    data_path = os.path.join(TEST_PATH, "test_overlappogram.fits")
    overlappogram = load_overlappogram(data_path, weights_path=None)
    assert isinstance(overlappogram, NDCube)
    assert overlappogram.uncertainty is None


def test_load_response_cube():
    pass
