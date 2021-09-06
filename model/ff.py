import torch
import torch.nn as nn
from block import fusions

from model.base import ImageEncoderTextEncoderBase


class FeatureFusionMethod(ImageEncoderTextEncoderBase):
    """FeatureFusionMethod"""

    def __init__(self, cfg, texts):
        super(FeatureFusionMethod, self).__init__(cfg, texts)
        ff_type = cfg.TRAIN.MODEL.composer_model.name
        assert ff_type in fusions.__dict__, f"Invalid fusion method is given: {ff_type}"
        _in_c = [self.out_feature_image, self.out_feature_text]
        _out_c = self.out_feature_image
        self.model["fusion"] = fusions.__dict__[ff_type](_in_c, _out_c)

        # define model
        self.model = nn.ModuleDict(self.model)

    def compose_img_text(self, f_img, f_text):
        return self.model["fusion"]((f_img, f_text))
