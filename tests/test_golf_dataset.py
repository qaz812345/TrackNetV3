"""Tests for GolfBallTrajectoryDataset using a synthetic 16-row manifest."""

import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from tracknet_golf.data.dataset import GolfBallTrajectoryDataset
from tracknet_golf.data.manifest import MANIFEST_COLUMNS


# ------------------------------------------------------------------ fixtures

def _make_fake_manifest_with_images(tmpdir: Path, n_frames: int = 16) -> pd.DataFrame:
    """Create a manifest CSV and matching dummy JPEG files."""
    rows = []
    for i in range(n_frames):
        img_name = f"frame_{i:06d}.jpg"
        img_path = tmpdir / img_name
        # Save a small random image
        arr = np.random.randint(0, 255, (1920, 1080, 3), dtype=np.uint8)
        Image.fromarray(arr).save(str(img_path))

        rows.append({
            "sample_id": f"shotX:{i}",
            "shot_id": "shotX",
            "image_id": i,
            "image_name": img_name,
            "image_path": str(img_path),
            "frame_index": i,
            "rel_frame": i,
            "proposed_rel_frame": i - 5,
            "orig_width": 1080,
            "orig_height": 1920,
            "x_px": 540.0,
            "y_px": 960.0,
            "visibility_label": "sharp",
            "visible": 1,
            "usable_for_training": "yes",
            "split": "train",
        })
    return pd.DataFrame(rows, columns=MANIFEST_COLUMNS)


@pytest.fixture(scope="module")
def fake_manifest_dir(tmp_path_factory):
    tmpdir = tmp_path_factory.mktemp("frames")
    df = _make_fake_manifest_with_images(tmpdir, n_frames=16)
    return df


# ------------------------------------------------------------------ tests

def test_dataset_length_no_step(fake_manifest_dir):
    df = fake_manifest_dir
    ds = GolfBallTrajectoryDataset(df, input_height=64, input_width=36, seq_len=8, sliding_step=1)
    # 16 frames, seq_len=8, step=1 → 16 - 8 + 1 = 9 windows
    assert len(ds) == 9


def test_dataset_length_with_step(fake_manifest_dir):
    df = fake_manifest_dir
    ds = GolfBallTrajectoryDataset(df, input_height=64, input_width=36, seq_len=8, sliding_step=4)
    # windows at 0, 4, 8 → 3
    assert len(ds) == 3


def test_sample_image_tensor_shape_no_bg(fake_manifest_dir):
    df = fake_manifest_dir
    ds = GolfBallTrajectoryDataset(df, input_height=64, input_width=36, seq_len=8, sliding_step=1, bg_mode="")
    sample = ds[0]
    # seq_len * 3 channels
    assert sample["images"].shape == (24, 64, 36)


def test_sample_image_tensor_shape_concat_bg(fake_manifest_dir):
    df = fake_manifest_dir
    bg = np.zeros((64, 36, 3), dtype=np.uint8)
    ds = GolfBallTrajectoryDataset(
        df, input_height=64, input_width=36, seq_len=8, sliding_step=1,
        bg_mode="concat", bg_frames={"shotX": bg}
    )
    sample = ds[0]
    # (seq_len + 1) * 3 channels
    assert sample["images"].shape == (27, 64, 36)


def test_sample_heatmap_tensor_shape(fake_manifest_dir):
    df = fake_manifest_dir
    ds = GolfBallTrajectoryDataset(df, input_height=64, input_width=36, seq_len=8, sliding_step=1)
    sample = ds[0]
    assert sample["heatmaps"].shape == (8, 64, 36)


def test_visible_heatmap_has_peak(fake_manifest_dir):
    df = fake_manifest_dir
    ds = GolfBallTrajectoryDataset(
        df, input_height=64, input_width=36, seq_len=8, sliding_step=1,
        sigma=2.5, target_mode="binary_disk"
    )
    sample = ds[0]
    # All frames are visible, heatmaps should not be all zeros
    for i in range(8):
        assert sample["heatmaps"][i].max() > 0


def test_invisible_heatmap_is_zero():
    """Frames with visible=0 should produce all-zero heatmaps."""
    rows = []
    for i in range(8):
        rows.append({
            "sample_id": f"shotY:{i}",
            "shot_id": "shotY",
            "image_id": i,
            "image_name": "does_not_exist.jpg",
            "image_path": "/does_not_exist/frame.jpg",
            "frame_index": i,
            "rel_frame": i,
            "proposed_rel_frame": i,
            "orig_width": 1080,
            "orig_height": 1920,
            "x_px": 0.0,
            "y_px": 0.0,
            "visibility_label": "",
            "visible": 0,
            "usable_for_training": "no",
            "split": "train",
        })
    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    ds = GolfBallTrajectoryDataset(df, input_height=64, input_width=36, seq_len=8)
    sample = ds[0]
    for i in range(8):
        assert sample["heatmaps"][i].max() == 0.0


def test_sample_metadata(fake_manifest_dir):
    df = fake_manifest_dir
    ds = GolfBallTrajectoryDataset(df, input_height=64, input_width=36, seq_len=8, sliding_step=1)
    sample = ds[0]
    assert sample["shot_id"] == "shotX"
    assert len(sample["frame_index"]) == 8
    assert len(sample["visible"]) == 8
    assert len(sample["x_orig"]) == 8
    assert len(sample["y_orig"]) == 8


def test_coordinate_scaling(fake_manifest_dir):
    df = fake_manifest_dir
    ds = GolfBallTrajectoryDataset(df, input_height=64, input_width=36, seq_len=8)
    sample = ds[0]
    # orig: 540/1080 * 36 = 18.0, 960/1920 * 64 = 32.0
    assert abs(sample["x_input"][0] - 18.0) < 1e-3
    assert abs(sample["y_input"][0] - 32.0) < 1e-3
