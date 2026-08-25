import csv
import os
import shlex
import time
from pathlib import Path

import cv2


EXPERIMENT_LOG_FILE = Path("experiments/experiment_log.csv")
EXPERIMENT_LOG_FIELDS = [
    "date",
    "id",
    "video_info",
    "angle",
    "background",
    "preprocessing",
    "resolution",
    "fps",
    "duration_s",
    "bitrate",
    "runtime_s",
    "model_load_s",
    "median_s",
    "inference_s",
    "write_csv_s",
    "video_merge_s",
    "result_notes",
    "input_video",
    "memo",
    "accuracy",
    "command",
]


def format_seconds(seconds):
    return f"{seconds:.3f}"


def read_experiment_rows(log_file):
    if not log_file.exists():
        return []

    with log_file.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def next_experiment_id(rows):
    ids = []
    for row in rows:
        try:
            ids.append(int(row.get("id", "")))
        except ValueError:
            continue
    return max(ids, default=0) + 1


def get_video_metadata(video_file):
    cap = cv2.VideoCapture(video_file)
    if not cap.isOpened():
        return {
            "resolution": "",
            "fps": "",
            "duration_s": "",
            "bitrate": "",
        }

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    duration_s = frame_count / fps if fps and fps > 0 else 0
    file_size_bits = os.path.getsize(video_file) * 8 if os.path.exists(video_file) else 0
    bitrate = int(file_size_bits / duration_s) if duration_s > 0 else ""

    return {
        "resolution": f"{width}x{height}" if width and height else "",
        "fps": f"{fps:.3f}".rstrip("0").rstrip(".") if fps else "",
        "duration_s": f"{duration_s:.3f}".rstrip("0").rstrip(".") if duration_s else "",
        "bitrate": bitrate,
    }


def append_prediction_log(args, runtime_s, stage_times, argv):
    log_file = EXPERIMENT_LOG_FILE
    metadata = get_video_metadata(args.video_file)
    rows = read_experiment_rows(log_file)
    row = {
        "date": time.time(),
        "id": next_experiment_id(rows),
        "video_info": "",
        "angle": "",
        "background": "",
        "preprocessing": "",
        "resolution": metadata["resolution"],
        "fps": metadata["fps"],
        "duration_s": metadata["duration_s"],
        "bitrate": metadata["bitrate"],
        "runtime_s": format_seconds(runtime_s),
        "model_load_s": format_seconds(stage_times.get('model_load_s', 0)),
        "median_s": format_seconds(stage_times.get('median_s', 0)),
        "inference_s": format_seconds(stage_times.get('inference_s', 0)),
        "write_csv_s": format_seconds(stage_times.get('write_csv_s', 0)),
        "video_merge_s": format_seconds(stage_times.get('video_merge_s', 0)),
        "result_notes": "",
        "input_video": os.path.basename(args.video_file),
        "memo": "",
        "accuracy": "",
        "command": " ".join(shlex.quote(arg) for arg in argv),
    }

    log_file.parent.mkdir(parents=True, exist_ok=True)
    exists = log_file.exists()
    with log_file.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=EXPERIMENT_LOG_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)

    print(f"Experiment log: appended #{row['id']} to {log_file}")
