#!/usr/bin/env python
"""Re-assign shot-level train/val/test splits on an existing manifest CSV.

Use this when you want to experiment with different split ratios or seeds
without re-parsing the CVAT XML from scratch.

Usage:
    python scripts/build_golf_splits.py \\
        --manifest data/golf/processed/manifest.csv \\
        --out data/golf/processed/manifest_resplit.csv \\
        --seed 42 \\
        --train-ratio 0.7 \\
        --val-ratio 0.15 \\
        --test-ratio 0.15
"""

import argparse
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Re-split an existing golf manifest by shot id")
    parser.add_argument("--manifest", required=True, help="Input manifest CSV")
    parser.add_argument("--out", required=True, help="Output manifest CSV (can be same path to overwrite)")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    args = parser.parse_args()

    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 1e-6:
        print("ERROR: train-ratio + val-ratio + test-ratio must sum to 1.0", file=sys.stderr)
        sys.exit(1)

    from tracknet_golf.data.manifest import load_manifest, save_manifest
    from tracknet_golf.data.splits import assign_shot_splits

    df = load_manifest(args.manifest)
    split_map = assign_shot_splits(
        df["shot_id"].tolist(),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )
    df["split"] = df["shot_id"].map(split_map)
    save_manifest(df, args.out)

    counts = df["split"].value_counts().to_dict()
    shot_counts = df.groupby("split")["shot_id"].nunique().to_dict()
    print(f"Saved re-split manifest to {args.out}")
    print(f"  frames — {counts}")
    print(f"  shots  — {shot_counts}")


if __name__ == "__main__":
    main()
