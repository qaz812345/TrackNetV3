"""Tests for heatmap generation."""

import numpy as np
import pytest
from tracknet_golf.data.heatmap import make_heatmap, make_zero_heatmap


def test_binary_disk_peak_at_center():
    hm = make_heatmap(64, 64, cx=32, cy=32, sigma=3.0, mode="binary_disk")
    assert hm[32, 32] == 1.0


def test_binary_disk_outside_sigma_is_zero():
    hm = make_heatmap(64, 64, cx=32, cy=32, sigma=3.0, mode="binary_disk")
    assert hm[0, 0] == 0.0


def test_binary_disk_shape():
    hm = make_heatmap(512, 288, cx=100, cy=200, sigma=2.5, mode="binary_disk")
    assert hm.shape == (512, 288)
    assert hm.dtype == np.float32


def test_gaussian_peak_at_center():
    hm = make_heatmap(64, 64, cx=32.0, cy=32.0, sigma=3.0, mode="gaussian")
    assert abs(hm[32, 32] - 1.0) < 1e-5


def test_gaussian_decays_from_center():
    hm = make_heatmap(64, 64, cx=32.0, cy=32.0, sigma=3.0, mode="gaussian")
    assert hm[32, 32] > hm[32, 35]
    assert hm[32, 35] > hm[32, 40]


def test_zero_heatmap_shape_and_values():
    hm = make_zero_heatmap(512, 288)
    assert hm.shape == (512, 288)
    assert hm.max() == 0.0
    assert hm.dtype == np.float32


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        make_heatmap(64, 64, cx=32, cy=32, sigma=2.5, mode="invalid_mode")
