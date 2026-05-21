"""Unit tests for manifest builder and split generator."""

import tempfile
import textwrap
from pathlib import Path

import pytest

from tracknet_golf.data.splits import assign_shot_splits
from tracknet_golf.data.manifest import build_manifest, save_manifest, load_manifest, MANIFEST_COLUMNS

SAMPLE_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="utf-8"?>
    <annotations>
      <version>1.1</version>
      <image id="0" name="shotA/frames/trajectory/rel_+000__proposed_rel_-005__frame_003046.jpg" width="1080" height="1920">
        <points label="ball" source="manual" occluded="0" points="534.28,1624.03" z_order="0">
          <attribute name="visibility">sharp</attribute>
          <attribute name="usable_for_training">yes</attribute>
        </points>
      </image>
      <image id="1" name="shotA/frames/trajectory/rel_+001__proposed_rel_-004__frame_003047.jpg" width="1080" height="1920">
        <points label="ball" source="manual" occluded="0" points="560.10,1600.50" z_order="0">
          <attribute name="visibility">blurred</attribute>
          <attribute name="usable_for_training">yes</attribute>
        </points>
      </image>
      <image id="2" name="shotB/frames/trajectory/rel_+000__proposed_rel_+000__frame_000001.jpg" width="1080" height="1920">
        <points label="ball" source="manual" occluded="0" points="100.0,200.0" z_order="0">
          <attribute name="visibility">streak</attribute>
          <attribute name="usable_for_training">no</attribute>
        </points>
      </image>
      <image id="3" name="shotC/frames/trajectory/rel_+000__proposed_rel_+000__frame_000001.jpg" width="1080" height="1920">
      </image>
    </annotations>
""")


def _write_xml(xml_str: str) -> str:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, encoding="utf-8") as f:
        f.write(xml_str)
        return f.name


# ------------------------------------------------------------------ splits

def test_split_assignment_deterministic():
    shots = [f"shot{i}" for i in range(10)]
    m1 = assign_shot_splits(shots, seed=13)
    m2 = assign_shot_splits(shots, seed=13)
    assert m1 == m2


def test_split_by_shot_not_frame():
    shots = ["A", "A", "A", "B", "B", "C"]
    m = assign_shot_splits(shots, train_ratio=0.5, val_ratio=0.25, seed=42)
    # All rows for the same shot_id must land in the same split
    assert m["A"] in ("train", "val", "test")
    assert m["B"] in ("train", "val", "test")
    assert m["C"] in ("train", "val", "test")


def test_split_ratios_approximate():
    shots = [f"shot{i}" for i in range(100)]
    m = assign_shot_splits(shots, train_ratio=0.8, val_ratio=0.1, seed=13)
    counts = {"train": 0, "val": 0, "test": 0}
    for v in m.values():
        counts[v] += 1
    assert counts["train"] == 80
    assert counts["val"] == 10
    assert counts["test"] == 10


# ------------------------------------------------------------------ manifest

def test_manifest_column_order():
    xml = _write_xml(SAMPLE_XML)
    df, _ = build_manifest(xml, frame_root="/nonexistent", seed=13)
    assert list(df.columns) == MANIFEST_COLUMNS


def test_manifest_row_count():
    xml = _write_xml(SAMPLE_XML)
    df, _ = build_manifest(xml, frame_root="/nonexistent", seed=13)
    assert len(df) == 4


def test_manifest_shot_split_consistency():
    xml = _write_xml(SAMPLE_XML)
    df, _ = build_manifest(xml, frame_root="/nonexistent", seed=13)
    # Both rows for shotA must be in the same split
    shotA_splits = df[df["shot_id"] == "shotA"]["split"].unique()
    assert len(shotA_splits) == 1


def test_manifest_missing_file_count():
    xml = _write_xml(SAMPLE_XML)
    _, summary = build_manifest(xml, frame_root="/nonexistent", seed=13)
    # All 4 files don't exist under /nonexistent
    assert summary["missing_file_count"] == 4


def test_manifest_summary_keys():
    xml = _write_xml(SAMPLE_XML)
    _, summary = build_manifest(xml, frame_root="/nonexistent", seed=13)
    expected_keys = {
        "total_frames", "total_shots", "frames_by_split", "shots_by_split",
        "visibility_counts", "usable_counts", "missing_file_count"
    }
    assert expected_keys.issubset(summary.keys())


def test_manifest_total_shots():
    xml = _write_xml(SAMPLE_XML)
    _, summary = build_manifest(xml, frame_root="/nonexistent", seed=13)
    assert summary["total_shots"] == 3  # shotA, shotB, shotC


def test_manifest_save_and_load_roundtrip():
    xml = _write_xml(SAMPLE_XML)
    df, _ = build_manifest(xml, frame_root="/nonexistent", seed=13)
    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
        out = f.name
    save_manifest(df, out)
    df2 = load_manifest(out)
    assert len(df2) == len(df)
    assert list(df2.columns) == MANIFEST_COLUMNS


def test_manifest_usable_no_preserved():
    xml = _write_xml(SAMPLE_XML)
    df, _ = build_manifest(xml, frame_root="/nonexistent", seed=13)
    shotB_row = df[df["shot_id"] == "shotB"].iloc[0]
    assert shotB_row["usable_for_training"] == "no"
