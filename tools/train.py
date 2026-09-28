#!/usr/bin/env python
"""Train one reported Agri-PIDNet-S configuration."""

import argparse
import sys
from pathlib import Path

import yaml
from mmengine import Config
from mmengine.runner import Runner
from mmengine.utils import import_modules_from_strings


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path,
                        default=ROOT / 'configs' / 'agri_pidnet_s.py')
    parser.add_argument('--matrix', type=Path,
                        default=ROOT / 'configs' / 'experiment_matrix.yaml')
    parser.add_argument('--experiment', default='full')
    parser.add_argument('--seed', type=int, choices=(0, 1, 3407), required=True)
    parser.add_argument('--data-root', type=Path,
                        default=ROOT / 'data' / 'cotton_two_stage')
    parser.add_argument('--work-dir', type=Path, default=None)
    parser.add_argument('--load-from', type=Path, default=None)
    return parser.parse_args()


def resolve_variant(matrix, name):
    variants = dict(matrix.get('experiments', {}))
    variants.update(matrix.get('internal_ablations', {}))
    if name not in variants:
        raise KeyError(f'Unknown experiment: {name}')
    variant = dict(variants[name])
    parent = variant.pop('parent', None)
    if parent:
        base = resolve_variant(matrix, parent)
        base.update(variant)
        return base
    return variant


def main():
    args = parse_args()
    cfg = Config.fromfile(str(args.config))
    import_modules_from_strings(**cfg.custom_imports)
    with args.matrix.open('r', encoding='utf-8') as stream:
        matrix = yaml.safe_load(stream)
    variant = resolve_variant(matrix, args.experiment)

    head = cfg.model.decode_head
    for key in ('use_ca_mkir', 'use_sae_d2t', 'use_bwr_bag',
                'use_ca', 'use_mbcr', 'use_window_attention'):
        if key in variant:
            head[key] = variant[key]

    for loader_name in (
            'train_dataloader', 'val_dataloader', 'test_dataloader'):
        cfg[loader_name].dataset.data_root = str(args.data_root)
    cfg.randomness = dict(seed=args.seed, deterministic=True)
    cfg.work_dir = str(args.work_dir or
                       (ROOT / 'work_dirs' /
                        f'{args.experiment}_seed{args.seed}'))
    if args.load_from:
        cfg.load_from = str(args.load_from)

    runner = Runner.from_cfg(cfg)
    runner.train()


if __name__ == '__main__':
    main()
