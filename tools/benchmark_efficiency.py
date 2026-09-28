#!/usr/bin/env python
"""Measure forward-only FLOPs and FPS under the reported protocol."""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from mmengine import Config
from mmengine.analysis import get_model_complexity_info
from mmengine.registry import init_default_scope
from mmengine.utils import import_modules_from_strings
from mmseg.registry import MODELS


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agri_pidnet.checkpoint import load_compatible_checkpoint  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, default=None)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--total-forwards', type=int, default=200)
    parser.add_argument('--output', type=Path,
                        default=ROOT / 'results/efficiency_measurement.json')
    return parser.parse_args()


class ForwardOnly(nn.Module):
    """Expose the raw model forward used by the FLOPs calculation."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, inputs):
        return self.model(inputs, mode='tensor')


def format_count(value):
    for divisor, suffix in ((1e12, 'T'), (1e9, 'G'), (1e6, 'M'), (1e3, 'K')):
        if value >= divisor:
            return f'{value / divisor:.3f}{suffix}'
    return str(int(value))


def main():
    args = parse_args()
    if args.total_forwards <= args.warmup:
        raise ValueError('total-forwards must exceed warmup.')
    if not args.device.startswith('cuda') or not torch.cuda.is_available():
        raise RuntimeError('The reported FPS protocol requires a CUDA GPU.')

    cfg = Config.fromfile(str(args.config))
    import_modules_from_strings(**cfg.custom_imports)
    init_default_scope(cfg.get('default_scope', 'mmseg'))
    model = MODELS.build(cfg.model)
    if args.checkpoint:
        load_compatible_checkpoint(model, args.checkpoint)
    model.eval()
    complexity = get_model_complexity_info(
        ForwardOnly(model),
        input_shape=(3, 1024, 1024),
        show_table=False,
        show_arch=False)
    flops = int(complexity['flops'])
    params = int(complexity['params'])

    model.to(args.device).eval()
    tensor = torch.randn(1, 3, 1024, 1024, device=args.device,
                         dtype=torch.float32)

    elapsed = []
    with torch.no_grad():
        for index in range(args.total_forwards):
            torch.cuda.synchronize()
            started = time.perf_counter()
            model(tensor, mode='tensor')
            torch.cuda.synchronize()
            duration = time.perf_counter() - started
            if index >= args.warmup:
                elapsed.append(duration)

    result = {
        'input': '1x3x1024x1024',
        'batch_size': 1,
        'precision': 'FP32',
        'warmup_forwards': args.warmup,
        'total_forwards': args.total_forwards,
        'timed_forwards': len(elapsed),
        'mean_forward_seconds': sum(elapsed) / len(elapsed),
        'fps': len(elapsed) / sum(elapsed),
        'flops': format_count(flops),
        'parameters': format_count(params),
        'flops_raw': flops,
        'parameters_raw': params,
        'flops_convention': 'MMEngine complexity counter; 1 MAC reported as 1 FLOP',
        'timing_scope': ('model forward only; excludes image reading, '
                         'preprocessing, CPU-to-GPU transfer, and postprocessing'),
        'gpu': torch.cuda.get_device_name(torch.device(args.device)),
        'python': platform.python_version(),
        'pytorch': torch.__version__,
        'cuda': torch.version.cuda,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
