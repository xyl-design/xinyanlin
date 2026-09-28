from mmseg.datasets import BaseSegDataset
from mmseg.registry import DATASETS


@DATASETS.register_module()
class CottonFieldDataset(BaseSegDataset):
    """Four-class cotton-field semantic-segmentation dataset."""

    METAINFO = dict(
        classes=('background', 'crop_row', 'furrow', 'obstacle'),
        palette=[
            [0, 0, 0],
            [65, 177, 102],
            [229, 172, 68],
            [229, 68, 68],
        ],
    )

    def __init__(self,
                 img_suffix='.png',
                 seg_map_suffix='.png',
                 ignore_index=255,
                 **kwargs):
        super().__init__(
            img_suffix=img_suffix,
            seg_map_suffix=seg_map_suffix,
            ignore_index=ignore_index,
            **kwargs)
