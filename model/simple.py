"""simple.py
"""

import torch.nn as nn

from misc.common import ConcatModule
from model.base import ImageEncoderTextEncoderBase


class SimpleTextOnlyModel(ImageEncoderTextEncoderBase):
    def __init__(self, cfg, texts):
        super(SimpleTextOnlyModel, self).__init__(cfg, texts)
        self.model = nn.ModuleDict(self.model)

    def compose_img_text(self, f_img, f_text):
        return f_text


class SimpleImageOnlyModel(ImageEncoderTextEncoderBase):
    def __init__(self, cfg, texts):
        super(SimpleImageOnlyModel, self).__init__(cfg, texts)
        self.model = nn.ModuleDict(self.model)

    def compose_img_text(self, f_img, f_text):
        return f_img


class SimpleConcatModel(ImageEncoderTextEncoderBase):
    def __init__(self, cfg, texts):
        super(SimpleConcatModel, self).__init__(cfg, texts)
        embed_dim = cfg.TRAIN.MODEL.out_feature_image
        self.model["compose"] = nn.Sequential(
            ConcatModule(),
            nn.BatchNorm1d(2 * embed_dim),
            nn.ReLU(),
            nn.Linear(2 * embed_dim, 2 * embed_dim),
            nn.BatchNorm1d(2 * embed_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(2 * embed_dim, embed_dim),
        )

        self.model = nn.ModuleDict(self.model)

    def compose_img_text(self, f_img, f_text):
        return self.model["compose"]((f_img, f_text))
