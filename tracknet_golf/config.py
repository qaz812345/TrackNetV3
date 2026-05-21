"""Config loader for the golf TrackNet pipeline."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

from tracknet_golf.constants import (
    BG_MODES,
    DEFAULT_INPUT_HEIGHT,
    DEFAULT_INPUT_WIDTH,
    DEFAULT_SEQ_LEN,
    DEFAULT_SIGMA,
    DEFAULT_SLIDING_STEP,
    DEFAULT_TARGET_MODE,
)

_DEFAULTS: dict[str, Any] = {
    "seed": 13,
    "input_height": DEFAULT_INPUT_HEIGHT,
    "input_width": DEFAULT_INPUT_WIDTH,
    "seq_len": DEFAULT_SEQ_LEN,
    "sliding_step": DEFAULT_SLIDING_STEP,
    "batch_size": 4,
    "epochs": 30,
    "learning_rate": 0.001,
    "optimizer": "Adam",
    "bg_mode": "",
    "target_mode": DEFAULT_TARGET_MODE,
    "sigma": DEFAULT_SIGMA,
    "threshold": 0.5,
    "eval_tolerance_px_input": 4,
    "num_workers": 4,
    "mixed_precision": False,
}


class GolfConfig:
    """Flat config object for a golf TrackNet run.

    All keys are accessible as attributes. Unknown keys are accepted so that
    YAML files can carry path / experiment metadata alongside training params.
    """

    def __init__(self, data: dict[str, Any]) -> None:
        cfg = copy.deepcopy(_DEFAULTS)
        cfg.update(data)
        _validate(cfg)
        self.__dict__.update(cfg)

    # ------------------------------------------------------------------ helpers

    @property
    def tracknet_in_dim(self) -> int:
        """Input channel count for TrackNet given bg_mode and seq_len."""
        return _tracknet_in_dim(self.bg_mode, self.seq_len)

    @property
    def tracknet_out_dim(self) -> int:
        return self.seq_len

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if not k.startswith("_")}


# ------------------------------------------------------------------ public API

def load_config(path: str | Path) -> GolfConfig:
    path = Path(path)
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return GolfConfig(data)


def default_config() -> GolfConfig:
    return GolfConfig({})


# ------------------------------------------------------------------ internals

def _tracknet_in_dim(bg_mode: str, seq_len: int) -> int:
    if bg_mode == "":
        return seq_len * 3
    if bg_mode == "subtract":
        return seq_len
    if bg_mode == "subtract_concat":
        return seq_len * 4
    if bg_mode == "concat":
        return (seq_len + 1) * 3
    raise ValueError(f"Unknown bg_mode: {bg_mode!r}")


def _validate(cfg: dict[str, Any]) -> None:
    if cfg["bg_mode"] not in BG_MODES:
        raise ValueError(f"bg_mode must be one of {BG_MODES}, got {cfg['bg_mode']!r}")
    if cfg["input_height"] % 8 != 0 or cfg["input_width"] % 8 != 0:
        raise ValueError(
            f"input_height and input_width must be divisible by 8, "
            f"got {cfg['input_height']}x{cfg['input_width']}"
        )
    if cfg["seq_len"] < 1:
        raise ValueError("seq_len must be >= 1")
