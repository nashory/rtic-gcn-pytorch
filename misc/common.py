#!/usr/bin/env python
# encoding: utf-8

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return torch.flatten(x, 1)


class ConcatModule(torch.nn.Module):
    def __init__(self):
        super(ConcatModule, self).__init__()

    def forward(self, x):
        x = list(x)
        x = torch.cat(x, dim=1)
        return x


class NormalizationLayer(torch.nn.Module):
    def __init__(self, normalize_scale=1.0, learn_scale=True):
        super(NormalizationLayer, self).__init__()
        self.norm_s = float(normalize_scale)
        if learn_scale:
            self.norm_s = torch.nn.Parameter(torch.FloatTensor((self.norm_s,)))

    def forward(self, x):
        features = self.norm_s * x / torch.norm(x, dim=1, keepdim=True).expand_as(x)
        return features


class GlobalAvgPool2d(torch.nn.Module):
    def forward(self, x):
        return F.adaptive_avg_pool2d(x, (1, 1))
