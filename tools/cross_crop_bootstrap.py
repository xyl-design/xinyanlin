#!/usr/bin/env python
"""Image-level bootstrap confidence intervals for cross-crop evaluation."""

import argparse
import csv
from pathlib import Path

import numpy as np
import pandas as pd


CLASS_NAMES = ('background', 'crop_row', 'furrow', 'obstacle')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--per-image-csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--resamples', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=20260917)
    return parser.parse_args()


def divide(numerator, denominator):
    if denominator == 0:
        return np.nan
    return numerator / denominator


def metrics(matrix):
    matrix = matrix.astype(float)
    values = {}
    ious = []
    for index, name in enumerate(CLASS_NAMES):
        tp = matrix[index, index]
        fp = matrix[:, index].sum() - tp
        fn = matrix[index, :].sum() - tp
        iou = divide(tp, tp + fp + fn)
        precision = divide(tp, tp + fp)
        recall = divide(tp, tp + fn)
        f1 = divide(2 * tp, 2 * tp + fp + fn)
        values.update({
            f'{name}_iou': iou,
            f'{name}_precision': precision,
            f'{name}_recall': recall,
            f'{name}_f1': f1,
        })
        ious.append(iou)
    values['miou'] = np.nanmean(ious)
    values['structural_miou'] = np.nanmean(ious[:3])
    return values


def matrices_from_frame(frame):
    columns = [f'c{i}{j}' for i in range(4) for j in range(4)]
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f'Missing confusion columns: {sorted(missing)}')
    return frame[columns].to_numpy(dtype=np.int64).reshape(-1, 4, 4)


def summarize_subset(name, frame, resamples, rng):
    matrices = matrices_from_frame(frame)
    point = metrics(matrices.sum(axis=0))
    draws = {metric: [] for metric in point}
    for _ in range(resamples):
        selected = rng.integers(0, len(matrices), size=len(matrices))
        sample = metrics(matrices[selected].sum(axis=0))
        for metric, value in sample.items():
            draws[metric].append(value)

    rows = []
    for metric, point_value in point.items():
        values = np.asarray(draws[metric], dtype=float)
        finite = values[np.isfinite(values)]
        lower, upper = (np.nan, np.nan) if not len(finite) else \
            np.percentile(finite, [2.5, 97.5])
        rows.append({
            'subset': name,
            'n_images': len(frame),
            'metric': metric,
            'point_percent': point_value * 100,
            'ci_lower_percent': lower * 100,
            'ci_upper_percent': upper * 100,
            'valid_resamples': len(finite),
            'undefined_resamples': resamples - len(finite),
        })
    return rows


def main():
    args = parse_args()
    frame = pd.read_csv(args.per_image_csv)
    if 'crop' not in frame.columns:
        raise ValueError('The input CSV must include a crop column.')
    expected = {'tomato': 24, 'watermelon': 23, 'maize': 22}
    actual = frame['crop'].value_counts().to_dict()
    if actual != expected or len(frame) != 69:
        raise ValueError(f'Expected tomato/watermelon/maize 24/23/22; got {actual}')

    rng = np.random.default_rng(args.seed)
    rows = []
    for crop in ('tomato', 'watermelon', 'maize'):
        rows.extend(summarize_subset(
            crop, frame[frame['crop'] == crop], args.resamples, rng))
    # Overall is sampled directly from all 69 images and is not crop-stratified.
    rows.extend(summarize_subset('overall', frame, args.resamples, rng))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'Wrote {len(rows)} metric rows to {args.output}')


if __name__ == '__main__':
    main()

