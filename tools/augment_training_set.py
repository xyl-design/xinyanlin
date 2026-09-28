#!/usr/bin/env python
"""Generate the paper's fixed offline-augmented training set.

The 495 original training pairs are retained and three augmented pairs are
created from every original, producing 1,980 training pairs. Validation and
test pairs are copied unchanged. Online random augmentation remains in the
training configuration.
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np


EXPECTED_SPLIT_COUNTS = {'train': 495, 'val': 107, 'test': 107}
COPIES_PER_IMAGE = 3

# Probabilities used to form the predefined augmentation combinations.
PROBABILITIES = {
    'brightness': 0.50,
    'contrast': 0.50,
    'saturation': 0.50,
    'gaussian_blur': 0.30,
    'rotation': 0.50,
    'vertical_flip': 0.10,
    'gaussian_noise': 0.30,
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixed-split-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    return parser.parse_args()


def image_files(directory):
    suffixes = {'.png', '.jpg', '.jpeg'}
    return sorted(path for path in directory.iterdir()
                  if path.suffix.lower() in suffixes)


def mask_for_image(mask_dir, image_path):
    same_name = mask_dir / image_path.name
    png_name = mask_dir / f'{image_path.stem}.png'
    if same_name.is_file():
        return same_name
    if png_name.is_file():
        return png_name
    raise FileNotFoundError(f'No mask found for {image_path.name}')


def copy_pair(image_path, mask_path, image_out, mask_out, output_name):
    image_out.mkdir(parents=True, exist_ok=True)
    mask_out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(image_path, image_out / output_name)
    shutil.copy2(mask_path, mask_out / output_name)


def adjust_brightness(image, factor):
    return np.clip(image.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def adjust_contrast(image, factor):
    mean = image.astype(np.float32).mean(axis=(0, 1), keepdims=True)
    return np.clip((image.astype(np.float32) - mean) * factor + mean,
                   0, 255).astype(np.uint8)


def adjust_saturation(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def rotate_pair(image, mask, angle):
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), angle, 1.0)
    image = cv2.warpAffine(
        image, matrix, (width, height), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    mask = cv2.warpAffine(
        mask, matrix, (width, height), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255)
    return image, mask


def ensure_nonempty_recipe(flags, rng):
    if any(flags.values()):
        return flags
    candidates = tuple(flags)
    flags[candidates[int(rng.integers(0, len(candidates)))]] = True
    return flags


def augment_pair(image, mask, rng):
    flags = ensure_nonempty_recipe(
        {name: bool(rng.random() < probability)
         for name, probability in PROBABILITIES.items()}, rng)
    if flags['brightness']:
        factor = float(rng.uniform(0.80, 1.20))
        image = adjust_brightness(image, factor)
    if flags['contrast']:
        factor = float(rng.uniform(0.80, 1.20))
        image = adjust_contrast(image, factor)
    if flags['saturation']:
        factor = float(rng.uniform(0.80, 1.20))
        image = adjust_saturation(image, factor)
    if flags['gaussian_blur']:
        kernel = int(rng.choice([3, 5]))
        sigma = float(rng.uniform(0.10, 1.50))
        image = cv2.GaussianBlur(image, (kernel, kernel), sigmaX=sigma)
    if flags['rotation']:
        angle = float(rng.uniform(-15.0, 15.0))
        image, mask = rotate_pair(image, mask, angle)
    if flags['vertical_flip']:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    if flags['gaussian_noise']:
        standard_deviation = float(rng.uniform(5.0, 20.0))
        noise = rng.normal(0.0, standard_deviation, image.shape)
        image = np.clip(image.astype(np.float32) + noise,
                        0, 255).astype(np.uint8)

    return image, mask


def verify_input(root):
    for split, expected in EXPECTED_SPLIT_COUNTS.items():
        images = image_files(root / split / 'images')
        if len(images) != expected:
            raise ValueError(
                f'Expected {expected} {split} images, found {len(images)}.')
        for image_path in images:
            mask_for_image(root / split / 'masks', image_path)


def main():
    args = parse_args()
    verify_input(args.fixed_split_root)
    rng = np.random.default_rng(0)
    training_pair_count = 0

    # Validation and test data remain unchanged.
    for split in ('val', 'test'):
        for image_path in image_files(
                args.fixed_split_root / split / 'images'):
            mask_path = mask_for_image(
                args.fixed_split_root / split / 'masks', image_path)
            copy_pair(
                image_path, mask_path,
                args.output / split / 'images',
                args.output / split / 'masks',
                image_path.name)

    train_images = image_files(args.fixed_split_root / 'train' / 'images')
    train_image_out = args.output / 'train' / 'images'
    train_mask_out = args.output / 'train' / 'masks'

    for image_path in train_images:
        mask_path = mask_for_image(
            args.fixed_split_root / 'train' / 'masks', image_path)
        original_name = f'{image_path.stem}.png'
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        mask = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
        if image is None or mask is None:
            raise ValueError(f'Unable to decode {image_path.name}')
        if image.shape[:2] != mask.shape[:2]:
            raise ValueError(f'Image-mask shape mismatch: {image_path.name}')

        copy_pair(
            image_path, mask_path, train_image_out, train_mask_out,
            original_name)
        training_pair_count += 1

        for copy_index in range(1, COPIES_PER_IMAGE + 1):
            augmented_image, augmented_mask = augment_pair(
                image.copy(), mask.copy(), rng)
            output_name = f'{image_path.stem}_aug{copy_index:02d}.png'
            train_image_out.mkdir(parents=True, exist_ok=True)
            train_mask_out.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(train_image_out / output_name), augmented_image):
                raise IOError(f'Unable to write {output_name}')
            if not cv2.imwrite(str(train_mask_out / output_name), augmented_mask):
                raise IOError(f'Unable to write mask {output_name}')
            training_pair_count += 1

    expected_total = EXPECTED_SPLIT_COUNTS['train'] * (COPIES_PER_IMAGE + 1)
    if training_pair_count != expected_total:
        raise RuntimeError(
            f'Expected {expected_total} training pairs, '
            f'got {training_pair_count}.')
    print('Offline augmentation completed: 495 originals + 1,485 augmented')
    print('Training pairs: 1,980; validation: 107; test: 107')


if __name__ == '__main__':
    main()
