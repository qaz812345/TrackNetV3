"""Unit tests for tracknet_golf.data.cvat_parser."""

import textwrap
import xml.etree.ElementTree as ET
from io import StringIO
from pathlib import Path
import tempfile

import pytest

from tracknet_golf.data.cvat_parser import parse_cvat_xml, _parse_filename


# ------------------------------------------------------------------ helpers

SAMPLE_XML = textwrap.dedent("""\
    <?xml version="1.0" encoding="utf-8"?>
    <annotations>
      <version>1.1</version>
      <image id="0" name="00e53df4-cbd7-4d84-b0ca-e545fa7895a2/frames/trajectory/rel_+000__proposed_rel_-005__frame_003046.jpg" width="1080" height="1920">
        <points label="ball" source="manual" occluded="0" points="534.28,1624.03" z_order="0">
          <attribute name="visibility">sharp</attribute>
          <attribute name="usable_for_training">yes</attribute>
        </points>
      </image>
      <image id="1" name="00e53df4-cbd7-4d84-b0ca-e545fa7895a2/frames/trajectory/rel_+001__proposed_rel_-004__frame_003047.jpg" width="1080" height="1920">
        <points label="ball" source="manual" occluded="0" points="560.10,1600.50" z_order="0">
          <attribute name="visibility">blurred</attribute>
          <attribute name="usable_for_training">yes</attribute>
        </points>
      </image>
      <image id="2" name="another_shot/frames/trajectory/rel_+000__proposed_rel_+000__frame_000001.jpg" width="1080" height="1920">
      </image>
    </annotations>
""")


def _xml_to_df(xml_str: str):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", delete=False, encoding="utf-8") as f:
        f.write(xml_str)
        tmp = f.name
    return parse_cvat_xml(tmp)


# ------------------------------------------------------------------ tests

def test_parse_filename_basic():
    name = "00e53df4-cbd7-4d84-b0ca-e545fa7895a2/frames/trajectory/rel_+000__proposed_rel_-005__frame_003046.jpg"
    result = _parse_filename(name)
    assert result["shot_id"] == "00e53df4-cbd7-4d84-b0ca-e545fa7895a2"
    assert result["frame_index"] == 3046
    assert result["rel_frame"] == 0
    assert result["proposed_rel_frame"] == -5


def test_parse_filename_positive_proposed():
    name = "shot_abc/frames/trajectory/rel_+001__proposed_rel_+003__frame_000010.jpg"
    result = _parse_filename(name)
    assert result["shot_id"] == "shot_abc"
    assert result["frame_index"] == 10
    assert result["rel_frame"] == 1
    assert result["proposed_rel_frame"] == 3


def test_parse_cvat_xml_row_count():
    df = _xml_to_df(SAMPLE_XML)
    assert len(df) == 3


def test_parse_cvat_xml_shot_id():
    df = _xml_to_df(SAMPLE_XML)
    assert df.iloc[0]["shot_id"] == "00e53df4-cbd7-4d84-b0ca-e545fa7895a2"
    assert df.iloc[2]["shot_id"] == "another_shot"


def test_parse_cvat_xml_frame_index():
    df = _xml_to_df(SAMPLE_XML)
    assert df.iloc[0]["frame_index"] == 3046
    assert df.iloc[1]["frame_index"] == 3047
    assert df.iloc[2]["frame_index"] == 1


def test_parse_cvat_xml_rel_frames():
    df = _xml_to_df(SAMPLE_XML)
    assert df.iloc[0]["rel_frame"] == 0
    assert df.iloc[0]["proposed_rel_frame"] == -5
    assert df.iloc[1]["rel_frame"] == 1
    assert df.iloc[1]["proposed_rel_frame"] == -4


def test_parse_cvat_xml_coordinates():
    df = _xml_to_df(SAMPLE_XML)
    assert abs(df.iloc[0]["x_px"] - 534.28) < 1e-3
    assert abs(df.iloc[0]["y_px"] - 1624.03) < 1e-3


def test_parse_cvat_xml_visibility():
    df = _xml_to_df(SAMPLE_XML)
    assert df.iloc[0]["visibility_label"] == "sharp"
    assert df.iloc[1]["visibility_label"] == "blurred"


def test_parse_cvat_xml_usable():
    df = _xml_to_df(SAMPLE_XML)
    assert df.iloc[0]["usable_for_training"] == "yes"
    assert df.iloc[1]["usable_for_training"] == "yes"


def test_parse_cvat_xml_visible_flag():
    df = _xml_to_df(SAMPLE_XML)
    assert df.iloc[0]["visible"] == 1
    assert df.iloc[1]["visible"] == 1
    # image with no points element
    assert df.iloc[2]["visible"] == 0
    assert df.iloc[2]["x_px"] == 0.0
    assert df.iloc[2]["y_px"] == 0.0


def test_parse_cvat_xml_orig_size():
    df = _xml_to_df(SAMPLE_XML)
    assert df.iloc[0]["orig_width"] == 1080
    assert df.iloc[0]["orig_height"] == 1920


def test_parse_cvat_xml_sample_id():
    df = _xml_to_df(SAMPLE_XML)
    assert df.iloc[0]["sample_id"] == "00e53df4-cbd7-4d84-b0ca-e545fa7895a2:3046"


def test_parse_cvat_xml_image_id():
    df = _xml_to_df(SAMPLE_XML)
    assert df.iloc[0]["image_id"] == 0
    assert df.iloc[1]["image_id"] == 1
