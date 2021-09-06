"""
Implementation of FiLM
ref: https://github.com/rosinality/film-pytorch/edit/master/model.py
"""

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.init import kaiming_uniform, normal

from model.base import ImageEncoderTextEncoderBase


class ResBlock(nn.Module):
    def __init__(self, filter_size):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(filter_size, filter_size, [1, 1], 1, 1)
        self.conv2 = nn.Conv2d(filter_size, filter_size, [3, 3], 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(filter_size, affine=False)
        self.reset()

    def forward(self, input, gamma, beta):
        out = self.conv1(input)
        resid = F.relu(out)
        out = self.conv2(resid)
        out = self.bn(out)

        gamma = gamma.unsqueeze(2).unsqueeze(3)
        beta = beta.unsqueeze(2).unsqueeze(3)
        out = gamma * out + beta

        out = F.relu(out)
        out = out + resid
        return out

    def reset(self):
        kaiming_uniform(self.conv1.weight)
        self.conv1.bias.data.zero_()
        kaiming_uniform(self.conv2.weight)


class FilmCompositionModule(nn.Module):
    def __init__(self, out_feature_text, out_feature_image, n_resblock):
        super(FilmCompositionModule, self).__init__()
        self.n_resblock = n_resblock
        self.film = nn.Linear(out_feature_text, out_feature_image * 2 * n_resblock)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.resblocks = nn.ModuleList()
        for i in range(n_resblock):
            self.resblocks.append(ResBlock(out_feature_image))

    def forward(self, x):
        f_img, f_text = x
        b = f_img.size(0)
        film = self.film(f_text).chunk(self.n_resblock * 2, 1)
        for i, resblock in enumerate(self.resblocks):
            f_img = resblock(f_img, film[i * 2], film[i * 2 + 1])
        f_img = self.avgpool(f_img).view(b, -1)
        return f_img


class FiLM(ImageEncoderTextEncoderBase):
    """FiLM"""

    def __init__(self, cfg, texts):
        super(FiLM, self).__init__(cfg, texts)
        params = cfg.TRAIN.MODEL.composer_model.params.film
        self.model["film_compose_it"] = FilmCompositionModule(
            self.out_feature_text, self.out_feature_image, params.n_resblock
        )
        self.model = nn.ModuleDict(self.model)

    def compose_img_text(self, f_img, f_text):
        return self.model["film_compose_it"]((f_img, f_text))

    def get_manipulated_image_feature(self, x):
        """
        x = {
            'c_img': c_img,
            'c_iid': data['c_iid'],
            't_iid': data['t_iid'],
            'mod_key': data['mod_key'],
            'mod_str': mod_str,
        }
        """
        f_img = self.extract_image_feature(x["c_img"], pool=False)
        f_text = self.extract_text_feature(x["mod_str"])
        x = self.compose_img_text(f_img, f_text)
        return self.model["norm"](x)

    def forward(self, x):
        """
        data = {
            'c_img': c_img,
            'c_cap': c_cap,
            't_img': t_img,
            't_cap': t_cap,
            'mod_key': mod_key,
            'mod_str': mod_str,
        }
        """
        f_img_c = self.extract_image_feature(x["c_img"], pool=False)
        f_img_t = self.extract_image_feature(x["t_img"])
        f_text = self.extract_text_feature(x["mod_str"])
        f_cit_t = self.compose_img_text(f_img_c, f_text)
        return dict(f_img_c=f_img_c, f_img_t=f_img_t, f_cit_t=f_cit_t, f_text=f_text)
