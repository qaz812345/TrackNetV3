#!/usr/bin/env python
"""Convert a CVAT for images 1.1 XML file to the canonical golf manifest CSV.

Usage:
    python scripts/convert_cvat_to_manifest.py \\
        --cvat-xml data/golf/annotations/annotations.xml \\
        --frame-root data/golf/raw/frames \\
        --out data/golf/processed/manifest.csv \\
        --seed 13 \\
        --train-ratio 0.8 \\
        --val-ratio 0.1 \\
        --test-ratio 0.1
"""

import argparse
import json
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="CVAT XML → golf manifest CSV")
    parser.add_argument("--cvat-xml", required=True, help="Path to CVAT XML annotation file")
    parser.add_argument("--frame-root", required=True, help="Root directory containing extracted frames")
    parser.add_argument("--out", required=True, help="Output manifest CSV path")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print("ERROR: train-ratio + val-ratio + test-ratio must sum to 1.0", file=sys.stderr)
        sys.exit(1)

    from tracknet_golf.data.manifest import build_manifest, save_manifest

    print(f"Parsing {args.cvat_xml} ...")
    df, summary = build_manifest(
        cvat_xml=args.cvat_xml,
        frame_root=args.frame_root,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    save_manifest(df, args.out)
    print(f"Saved manifest to {args.out}")
    print("\n--- Data quality summary ---")
    print(json.dumps(summary, indent=2))

    if summary["missing_file_count"] > 0:
        print(f"\nWARNING: {summary['missing_file_count']} frame files not found at frame_root={args.frame_root!r}")


if __name__ == "__main__":
    main()
