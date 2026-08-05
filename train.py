#!/usr/bin/env python3
"""Train Agri-PIDNet-S with the settings reported in the paper.

The full train/val/test dataset is required. The representative minimal
dataset is intended for inspection only and cannot reproduce paper results.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--config", default="my_config/pidnet_all.py")
    parser.add_argument("--work-dir", default="work_dirs/agri_pidnet_s")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20000)
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    data = args.data_root.resolve()
    config = (repo / args.config).resolve()
    launcher = repo / "tools" / "train.py"
    required = [
        data / split / kind
        for split in ("train", "val", "test")
        for kind in ("images", "labels")
    ]
    missing = [str(path) for path in [config, launcher, *required] if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing required path(s):\n" + "\n".join(missing))

    root = data.as_posix()
    options = [
        f"data.samples_per_gpu={args.batch_size}",
        f"runner.max_iters={args.iterations}",
        f"checkpoint_config.interval={args.iterations}",
        "optimizer.lr=0.0005",
        "optimizer.weight_decay=0.01",
        "optimizer_config.grad_clip.max_norm=1.0",
        "lr_config.power=0.9",
        "lr_config.warmup_iters=1500",
    ]
    for split in ("train", "val", "test"):
        options += [
            f"data.{split}.data_root={root}",
            f"data.{split}.img_dir={split}/images",
            f"data.{split}.ann_dir={split}/labels",
        ]

    command = [
        sys.executable,
        str(launcher),
        str(config),
        "--work-dir",
        str((repo / args.work_dir).resolve()),
        "--seed",
        "42",
        "--deterministic",
        "--cfg-options",
        *options,
    ]
    print("Running:", " ".join(command))
    subprocess.run(command, cwd=repo, check=True)


if __name__ == "__main__":
    main()
