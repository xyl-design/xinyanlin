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

## Data

The cotton-field dataset, cross-crop test set, fixed split manifest, and seed-0
checkpoint are provided separately on Zenodo. The cotton dataset uses the fixed
495/107/107 training, validation, and test split.

Zenodo DOI: [10.5281/zenodo.23006176](https://doi.org/10.5281/zenodo.23006176)

## License

This repository is released under the MIT License. The implementation is based
on [PIDNet](https://github.com/XuJiacong/PIDNet) and
[MMSegmentation](https://github.com/open-mmlab/mmsegmentation).
