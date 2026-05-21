"""Golf-specific constants — all configurable via config.py; defaults here are the baseline."""

# Portrait aspect ratio: 9:16, same pixel count as original 288x512 landscape
DEFAULT_INPUT_HEIGHT = 512
DEFAULT_INPUT_WIDTH = 288

# Heatmap
DEFAULT_SIGMA = 2.5
DEFAULT_TARGET_MODE = "binary_disk"

# Sequence
DEFAULT_SEQ_LEN = 8
DEFAULT_SLIDING_STEP = 1

# Background modes supported
BG_MODES = ("", "subtract", "subtract_concat", "concat")

# Visibility labels from CVAT
VISIBILITY_LABELS = ("sharp", "blurred", "streak", "unclear")

# Image format used by the golf dataset (JPEGs from iPhone)
IMG_FORMAT = "jpg"
