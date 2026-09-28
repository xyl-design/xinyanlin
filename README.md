# Agri-PIDNet-S

This repository contains the code, configurations, and evaluation scripts for
Agri-PIDNet-S, a semantic-segmentation model for crop rows, furrows, field
obstacles, and background regions.

## Environment

- Python 3.10
- PyTorch 2.1.0+cu121
- CUDA 12.1
- MMEngine 0.10.3
- MMCV 2.1.0
- MMSegmentation 1.2.2
- NumPy, pandas, Pillow, OpenCV, and PyYAML

## Data

The cotton-field dataset, cross-crop test set, fixed split manifest, and seed-0
checkpoint are provided separately on Zenodo. The cotton dataset uses the fixed
495/107/107 training, validation, and test split and should not be reshuffled.

Zenodo DOI: [10.5281/zenodo.23006176](https://doi.org/10.5281/zenodo.23006176)

After downloading and extracting the Zenodo files, use the following layout:

```text
data/
  cotton_709/
    train/images/  train/masks/
    val/images/    val/masks/
    test/images/   test/masks/
  cross_crop_69/
    tomato/images/      tomato/masks/
    watermelon/images/  watermelon/masks/
    maize/images/       maize/masks/
checkpoints/
  agri_pidnet_s.pth
```

## Usage

Run all commands from the repository root.

Generate the fixed offline-augmented training set:

```bash
python tools/augment_training_set.py --fixed-split-root data/cotton_709 --output data/cotton_two_stage
```

Train the complete model with the reported seed-0 setting:

```bash
python tools/train.py --experiment full --seed 0 --data-root data/cotton_two_stage
```

The full-factorial variants are listed in `configs/experiment_matrix.yaml`.
Use seeds `0`, `1`, and `3407` with `--experiment full` for the reported
training-seed stability runs.

Evaluate the released seed-0 checkpoint on the fixed cotton test set:

```bash
python tools/evaluate.py --config configs/agri_pidnet_s.py --checkpoint checkpoints/agri_pidnet_s.pth --images data/cotton_709/test/images --masks data/cotton_709/test/masks --output-dir outputs/seed0_test
```

For cross-crop evaluation, run `tools/evaluate.py` separately for tomato,
watermelon, and maize by specifying the corresponding `--images`, `--masks`,
`--crop`, and `--output-dir` arguments. Combine the three generated
`per_image_confusion.csv` files into one CSV with a single header, then run:

```bash
python tools/cross_crop_bootstrap.py --per-image-csv outputs/cross_crop/per_image_confusion_all.csv --output outputs/cross_crop/bootstrap_ci.csv --resamples 10000 --seed 20260917
```

The released checkpoint is the seed-0 model used for the main comparisons in
Tables 1 and 2. The summary files `results/random_seed_results.csv`,
`results/factorial_ablation_results.csv`, and `results/cross_crop_results.csv`
correspond to Table 3, Table 4, and Tables 7 and 8, respectively.

## License

This repository is released under the MIT License. The implementation is based
on [PIDNet](https://github.com/XuJiacong/PIDNet) and
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation).
