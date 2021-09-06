"""
Implementation of MRN
ref: https://github.com/jnhwkim/nips-mrn-vqa/blob/master/netdef/MRN.lua
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import kaiming_uniform, normal

from model.base import ImageEncoderTextEncoderBase


class MrnCompositionModule(nn.Module):
    def __init__(self, in_c, out_c):
        super(MrnCompositionModule, self).__init__()
        self.v_block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_c, in_c),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(in_c, in_c),
            nn.Tanh(),
        )
        self.t_block = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_c, in_c),
            nn.Tanh(),
        )
        self.t_linear = nn.Sequential(nn.Dropout(0.5), nn.Linear(in_c, in_c))

    def forward(self, x):
        f_v = self.v_block(x[0])
        f_t = self.t_block(x[1])
        f = f_v * f_t
        f += self.t_linear(x[1])
        return f


class MRN(ImageEncoderTextEncoderBase):
    """MRN, structure (b) in Figure 3 of MRN paper (num_layers==1 is used for simplicity)"""

    def __init__(self, cfg, texts):
        super(MRN, self).__init__(cfg, texts)

        self.model["mrn_compose_it"] = MrnCompositionModule(
            in_c=self.out_feature_image,
            out_c=self.out_feature_image,
        )
        self.model = nn.ModuleDict(self.model)

    def compose_img_text(self, f_img, f_text):
        return self.model["mrn_compose_it"]((f_img, f_text))
