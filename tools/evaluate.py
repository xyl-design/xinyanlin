#!/usr/bin/env python
"""Evaluate a checkpoint and export image-level confusion matrices."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from mmengine import Config
from mmengine.utils import import_modules_from_strings
from mmseg.apis import inference_model, init_model
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agri_pidnet.checkpoint import load_compatible_checkpoint  # noqa: E402

CLASS_NAMES = ('background', 'crop_row', 'furrow', 'obstacle')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--images', type=Path,
                        default=ROOT / 'data/cotton_709/test/images')
    parser.add_argument('--masks', type=Path,
                        default=ROOT / 'data/cotton_709/test/masks')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--crop', choices=('tomato', 'watermelon', 'maize'),
                        default=None)
    return parser.parse_args()


def confusion_matrix(gt, pred, num_classes=4, ignore_index=255):
    valid = (gt != ignore_index) & (gt >= 0) & (gt < num_classes)
    encoded = gt[valid].astype(np.int64) * num_classes + \
        pred[valid].astype(np.int64)
    return np.bincount(encoded, minlength=num_classes ** 2).reshape(
        num_classes, num_classes)


def safe_divide(numerator, denominator):
    return np.divide(
        numerator, denominator,
        out=np.full_like(numerator, np.nan, dtype=float),
        where=denominator != 0,
    )


def metrics_from_confusion(matrix):
    matrix = matrix.astype(float)
    tp = np.diag(matrix)
    gt = matrix.sum(axis=1)
    pred = matrix.sum(axis=0)
    fp = pred - tp
    fn = gt - tp
    iou = safe_divide(tp, tp + fp + fn)
    recall = safe_divide(tp, tp + fn)
    precision = safe_divide(tp, tp + fp)
    f1 = safe_divide(2 * tp, 2 * tp + fp + fn)
    result = {
        'aAcc': matrix.trace() / matrix.sum(),
        'mIoU': np.nanmean(iou),
        'mAcc': np.nanmean(recall),
        'mFscore': np.nanmean(f1),
        'mPrecision': np.nanmean(precision),
        'mRecall': np.nanmean(recall),
        'mDice': np.nanmean(f1),
    }
    for index, name in enumerate(CLASS_NAMES):
        result.update({
            f'{name}_IoU': iou[index],
            f'{name}_Acc': recall[index],
            f'{name}_Fscore': f1[index],
            f'{name}_Precision': precision[index],
            f'{name}_Recall': recall[index],
            f'{name}_Dice': f1[index],
        })
    return result


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = Config.fromfile(str(args.config))
    import_modules_from_strings(**cfg.custom_imports)
    model = init_model(cfg, checkpoint=None, device=args.device)
    load_report = load_compatible_checkpoint(model, args.checkpoint)
    model.dataset_meta = dict(
        classes=CLASS_NAMES,
        palette=[[0, 0, 0], [65, 177, 102],
                 [229, 172, 68], [229, 68, 68]],
    )
    print(
        f"Loaded {load_report['loaded_tensors']} checkpoint tensors; "
        f"skipped {len(load_report['skipped_tensors'])} incompatible tensors.")

    rows = []
    total = np.zeros((4, 4), dtype=np.int64)
    image_paths = sorted(
        path for path in args.images.iterdir()
        if path.suffix.lower() in {'.png', '.jpg', '.jpeg'})
    for image_path in image_paths:
        mask_path = args.masks / f'{image_path.stem}.png'
        if not mask_path.is_file():
            raise FileNotFoundError(mask_path)
        prediction = inference_model(
            model, str(image_path)).pred_sem_seg.data.squeeze(0)
        prediction = prediction.detach().cpu().numpy()
        ground_truth = np.asarray(Image.open(mask_path))
        if prediction.shape != ground_truth.shape:
            raise ValueError(f'Shape mismatch for {image_path.name}')
        matrix = confusion_matrix(ground_truth, prediction)
        total += matrix
        row = {'image_id': image_path.name}
        if args.crop:
            row['crop'] = args.crop
        row.update({f'c{i}{j}': int(matrix[i, j])
                    for i in range(4) for j in range(4)})
        rows.append(row)

    fields = ['image_id'] + (['crop'] if args.crop else []) + [
        f'c{i}{j}' for i in range(4) for j in range(4)]
    with (args.output_dir / 'per_image_confusion.csv').open(
            'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    metrics = metrics_from_confusion(total)
    with (args.output_dir / 'summary_metrics.csv').open(
            'w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['metric', 'value_percent'])
        for name, value in metrics.items():
            writer.writerow([name, f'{value * 100:.4f}'])
    print(f'Evaluated {len(rows)} images.')
    print(f"mIoU: {metrics['mIoU'] * 100:.2f}%")


if __name__ == '__main__':
    main()
