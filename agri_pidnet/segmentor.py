"""Whole-image MMSegmentation 1.x wrapper for Agri-PIDNet-S."""

from mmseg.models.segmentors.base import BaseSegmentor
from mmseg.models.segmentors.encoder_decoder import EncoderDecoder
from mmseg.registry import MODELS


@MODELS.register_module()
class AgriEncoderDecoder(EncoderDecoder):
    """Encoder-decoder wrapper whose decode head consumes RGB tensors."""

    def __init__(self,
                 decode_head,
                 train_cfg=None,
                 test_cfg=None,
                 data_preprocessor=None,
                 pretrained=None,
                 init_cfg=None):
        BaseSegmentor.__init__(
            self,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg,
        )
        self.decode_head = MODELS.build(decode_head)
        self.align_corners = self.decode_head.align_corners
        self.num_classes = self.decode_head.num_classes
        self.out_channels = self.decode_head.out_channels
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def extract_feat(self, inputs):
        """The released head contains the complete PIDNet-S network."""
        return inputs

    def forward_dummy(self, inputs):
        """Forward-only entry used by the FLOPs/FPS utility."""
        return self._forward(inputs)
