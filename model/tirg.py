import math
import random
import string

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as M
from torch.autograd import Variable

import model.resnet as resnet
from misc.common import ConcatModule
from model.base import ImageEncoderTextEncoderBase


class TirgCompositionModule(nn.Module):
    def __init__(self, in_c, out_c):
        super(TirgCompositionModule, self).__init__()
        self.w = nn.Parameter(torch.FloatTensor([1.0, 10.0, 1.0, 1.0]))
        self.gated_feature_composer = torch.nn.Sequential(
            ConcatModule(),
            nn.BatchNorm1d(in_c),
            nn.ReLU(),
            nn.Linear(in_c, out_c),
        )
        self.res_info_composer = torch.nn.Sequential(
            ConcatModule(),
            nn.BatchNorm1d(in_c),
            nn.ReLU(),
            nn.Linear(in_c, in_c),
            nn.ReLU(),
            nn.Linear(in_c, out_c),
        )

    def forward(self, x):
        f1 = self.gated_feature_composer(x)
        f2 = self.res_info_composer(x)
        f = torch.sigmoid(f1) * x[0] * self.w[0] + f2 * self.w[1]
        return f


class TIRG(ImageEncoderTextEncoderBase):
    """The TIRG model.
    The method is described in
    Nam Vo, Lu Jiang, Chen Sun, Kevin Murphy, Li-Jia Li, Li Fei-Fei, James Hays.
    "Composing Text and Image for Image Retrieval - An Empirical Odyssey"
    CVPR 2019. arXiv:1812.07119
    """

    def __init__(self, cfg, texts):
        super(TIRG, self).__init__(cfg, texts)

        self.model["tirg_compose_it"] = TirgCompositionModule(
            in_c=self.out_feature_image + self.out_feature_text,
            out_c=self.out_feature_image,
        )

        # define model.
        self.model = nn.ModuleDict(self.model)

    def compose_img_text(self, f_img, f_text):
        return self.model["tirg_compose_it"]((f_img, f_text))
