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

import misc.act as aa
from model.base import ImageEncoderTextEncoderBase


class RticFusionModule(nn.Module):
    def __init__(self, in_c_img, in_c_text, act_fn):
        super(RticFusionModule, self).__init__()
        assert act_fn in aa.__dict__
        layers = [
            nn.BatchNorm1d(in_c_img + in_c_text),
            nn.__dict__[act_fn](),
            nn.Linear(in_c_img + in_c_text, in_c_img),
        ]
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        x = torch.cat(x, dim=1)
        return self.module(x)


class RticGatingModule(nn.Module):
    def __init__(self, in_c_img, act_fn):
        super(RticGatingModule, self).__init__()
        assert act_fn in aa.__dict__
        layers = [
            nn.Linear(in_c_img, in_c_img),
            nn.BatchNorm1d(in_c_img),
            aa.__dict__[act_fn](),
            nn.Linear(in_c_img, in_c_img),
            nn.Sigmoid(),
        ]
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        return self.module(x)


class RticErrorEncodingModule(nn.Module):
    def __init__(self, in_c_img, act_fn):
        super(RticErrorEncodingModule, self).__init__()
        assert act_fn in aa.__dict__
        layers = [
            nn.Linear(in_c_img, in_c_img // 2),
            nn.BatchNorm1d(in_c_img // 2),
            aa.__dict__[act_fn](),
            nn.Linear(in_c_img // 2, in_c_img // 2),
            nn.BatchNorm1d(in_c_img // 2),
            aa.__dict__[act_fn](),
            nn.Linear(in_c_img // 2, in_c_img),
        ]
        self.module = nn.Sequential(*layers)

    def forward(self, x):
        """x = fusion_feature (f_f)"""
        return self.module(x) + x


class RticCompositionModule(nn.Module):
    def __init__(self, in_c_img, in_c_text, n_blocks, act_fn="LeakyReLU"):
        super(RticCompositionModule, self).__init__()
        self.in_c_img = in_c_img
        self.in_c_text = in_c_text
        self.n_blocks = n_blocks

        # fusion block
        self.fs = RticFusionModule(
            in_c_img=in_c_img,
            in_c_text=in_c_text,
            act_fn=act_fn,
        )

        # gaiting block
        self.gating = RticGatingModule(
            in_c_img=in_c_img,
            act_fn=act_fn,
        )

        # error encoding block
        self.ee = nn.ModuleList()
        for i in range(self.n_blocks):
            ee = RticErrorEncodingModule(
                in_c_img=in_c_img,
                act_fn=act_fn,
            )
            self.ee.append(ee)

    def forward(self, x, return_ee=False):
        f = self.fs(x)
        g = self.gating(f)
        for ee in self.ee:
            f = ee(f)
        if return_ee:
            ee = f.clone()
            out = (x[0] * g) + (f * (1 - g))
            return out, ee
        else:
            out = (x[0] * g) + (f * (1 - g))
            return out


class RTIC(ImageEncoderTextEncoderBase):
    """Redisual Text Image Composer"""

    def __init__(self, cfg, texts):
        super(RTIC, self).__init__(cfg, texts)

        params = cfg.TRAIN.MODEL.composer_model.params.rtic
        self.act_fn = params.act_fn
        self.n_blocks = params.n_blocks

        self.model["rtic_compose_it"] = RticCompositionModule(
            in_c_img=self.out_feature_image,
            in_c_text=self.out_feature_image,
            n_blocks=self.n_blocks,
            act_fn=self.act_fn,
        )

        # define model
        self.model = nn.ModuleDict(self.model)

    def compose_img_text(self, f_img, f_text):
        return self.model["rtic_compose_it"]((f_img, f_text))
