"""Compatibility loading for the released seed-0 checkpoint."""

from pathlib import Path

import torch


def load_compatible_checkpoint(model, checkpoint_path, map_location='cpu'):
    """Load matching tensors and report intentionally unused training keys.

    The published checkpoint contains the full inference path. Two inherited
    ``BaseDecodeHead.conv_seg`` tensors are not used because Agri-PIDNet-S
    produces predictions with ``final_layer``. The released training config
    also contains a boundary-supervision head, which does not participate in
    inference and is absent from this checkpoint. Filtering is limited to
    missing, unexpected, or shape-incompatible tensors; every compatible
    tensor is loaded unchanged.
    """
    checkpoint_path = Path(checkpoint_path)
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=map_location, weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=map_location)

    state_dict = checkpoint.get('state_dict', checkpoint)
    state_dict = {
        (key[7:] if key.startswith('module.') else key): value
        for key, value in state_dict.items()
    }
    model_state = model.state_dict()
    compatible = {}
    skipped = {}
    for key, value in state_dict.items():
        if key not in model_state:
            skipped[key] = 'unexpected key'
        elif tuple(value.shape) != tuple(model_state[key].shape):
            skipped[key] = (
                f'shape {tuple(value.shape)} != '
                f'{tuple(model_state[key].shape)}')
        else:
            compatible[key] = value

    incompatible = model.load_state_dict(compatible, strict=False)
    allowed_skipped = {
        'decode_head.conv_seg.weight',
        'decode_head.conv_seg.bias',
    }
    allowed_missing_prefixes = (
        'decode_head.conv_seg.',
        'decode_head.seghead_d.',
    )
    unapproved_skipped = set(skipped) - allowed_skipped
    unapproved_missing = [
        key for key in incompatible.missing_keys
        if not key.startswith(allowed_missing_prefixes)
    ]
    if unapproved_skipped or unapproved_missing:
        raise RuntimeError(
            'Checkpoint is not compatible with this release: '
            f'unapproved skipped keys={sorted(unapproved_skipped)}, '
            f'unapproved missing keys={unapproved_missing}')
    report = {
        'checkpoint': str(checkpoint_path),
        'loaded_tensors': len(compatible),
        'skipped_tensors': skipped,
        'missing_keys': list(incompatible.missing_keys),
        'unexpected_keys': list(incompatible.unexpected_keys),
    }
    return report
