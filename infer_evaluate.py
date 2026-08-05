#!/usr/bin/env python3
"""Run Agri-PIDNet-S inference and report the paper's main metrics."""

import argparse
import json
import sys
from pathlib import Path


CLASSES = ("background", "crop_row", "furrow", "obstacle")
COLORS = ((205, 200, 185), (0, 128, 0), (139, 69, 19), (255, 0, 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--config", default="my_config/pidnet_all.py")
    parser.add_argument("--checkpoint", default="work_dirs/pidnet_all/iter_20000.pth")
    parser.add_argument("--output-dir", type=Path, default=Path("predictions"))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    repo = args.repo_root.resolve()
    sys.path.insert(0, str(repo))
    config = (repo / args.config).resolve()
    checkpoint = (repo / args.checkpoint).resolve()
    data = args.data_root.resolve()
    split_root = data / args.split
    image_dir = (split_root if (split_root / "images").is_dir() else data) / "images"
    label_dir = image_dir.parent / "labels"
    output_dir = args.output_dir.resolve()
    for path in (config, checkpoint, image_dir):
        if not path.exists():
            raise FileNotFoundError(path)

    import numpy as np
    from PIL import Image
    from mmseg.apis import inference_segmentor, init_segmentor

    model = init_segmentor(str(config), str(checkpoint), device=args.device)
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in {
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
    })
    if not images:
        raise FileNotFoundError(f"No images found in {image_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)
    confusion = np.zeros((4, 4), dtype=np.int64)
    evaluated = 0
    palette = [value for color in COLORS for value in color] + [0] * (768 - 12)

    for image_path in images:
        pred = np.asarray(inference_segmentor(model, str(image_path))[0], dtype=np.uint8)
        label_path = label_dir / f"{image_path.stem}.png"
        if label_path.exists():
            target = np.asarray(Image.open(label_path), dtype=np.uint8)
            if pred.shape != target.shape:
                pred = np.asarray(
                    Image.fromarray(pred).resize(target.shape[::-1], Image.Resampling.NEAREST)
                )
            valid = target < 4
            confusion += np.bincount(
                4 * target[valid].astype(np.int64) + pred[valid], minlength=16
            ).reshape(4, 4)
            evaluated += 1

        mask = Image.fromarray(pred, mode="P")
        mask.putpalette(palette)
        mask.save(output_dir / f"{image_path.stem}.png")

    report = {"images": len(images), "evaluated": evaluated, "classes": CLASSES}
    if evaluated:
        tp = np.diag(confusion).astype(float)
        actual = confusion.sum(1).astype(float)
        predicted = confusion.sum(0).astype(float)
        iou = np.divide(tp, actual + predicted - tp, out=np.full(4, np.nan), where=(actual + predicted - tp) > 0)
        dice = np.divide(2 * tp, actual + predicted, out=np.full(4, np.nan), where=(actual + predicted) > 0)
        precision = np.divide(tp, predicted, out=np.full(4, np.nan), where=predicted > 0)
        recall = np.divide(tp, actual, out=np.full(4, np.nan), where=actual > 0)
        obstacle_f1 = 2 * precision[3] * recall[3] / (precision[3] + recall[3])
        report.update({
            "IoU": dict(zip(CLASSES, (iou * 100).round(2).tolist())),
            "mIoU": round(float(np.nanmean(iou) * 100), 2),
            "mDice": round(float(np.nanmean(dice) * 100), 2),
            "obstacle_precision": round(float(precision[3] * 100), 2),
            "obstacle_recall": round(float(recall[3] * 100), 2),
            "obstacle_F1": round(float(obstacle_f1 * 100), 2),
            "confusion_matrix": confusion.tolist(),
        })

    report_path = output_dir / "metrics.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
