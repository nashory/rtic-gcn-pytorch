import math
import random
import string

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as M
from einops import rearrange
from torch.autograd import Variable

import model.resnet as resnet
from model.base import ImageEncoderTextEncoderBase


class ComposeAECompositionModule(nn.Module):
    def __init__(self, in_c_text, in_c_img):
        super(ComposeAECompositionModule, self).__init__()
        self.CONJUGATE = Variable(
            torch.cuda.FloatTensor(10000, 1).fill_(-1.0), requires_grad=False
        )  # large enough values.
        self.a = torch.nn.Parameter(torch.tensor([1.0, 10.0, 1.0, 1.0]))

        self.encoderLinear = torch.nn.Sequential(
            ComplexProjectionModule(in_c_img, in_c_text),
            LinearMapping(in_c_img),
        )

        self.encoderWithConv = torch.nn.Sequential(
            ComplexProjectionModule(in_c_img, in_c_text),
            ConvMapping(in_c_img),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_c_img),
            torch.nn.ReLU(),
            torch.nn.Linear(in_c_img, in_c_img),
            torch.nn.ReLU(),
            torch.nn.Linear(in_c_img, in_c_img),
        )

        self.txtdecoder = torch.nn.Sequential(
            torch.nn.BatchNorm1d(in_c_img),
            torch.nn.ReLU(),
            torch.nn.Linear(in_c_img, in_c_text),
            torch.nn.ReLU(),
            torch.nn.Linear(in_c_text, in_c_text),
        )

    def forward(self, x):
        f_img, f_text = x
        theta_linear = self.encoderLinear((f_img, f_text, self.CONJUGATE))
        theta_conv = self.encoderWithConv((f_img, f_text, self.CONJUGATE))
        theta = theta_linear * self.a[1] + theta_conv * self.a[0]

        dct_with_representations = {
            "repres": theta,
            "repr_to_compare_with_source": self.decoder(theta),
            "repr_to_compare_with_mods": self.txtdecoder(theta),
        }

        return dct_with_representations


class ComposeAE(ImageEncoderTextEncoderBase):
    """The ComposeAE model.

    The method is described in
    Muhammad Umer Anwaar, Egor Labintcev and Martin Kleinsteuber.
    ``Compositional Learning of Image-Text Query for Image Retrieval"
    arXiv:2006.11149
    """

    def __init__(self, **kwargs):
        super(ComposeAE, self).__init__(**kwargs)

        self.l2_loss = nn.MSELoss().to("cuda")

        self.model["composeae_compose_it"] = ComposeAECompositionModule(
            in_c_text=self.out_feature_text,
            in_c_img=self.out_feature_image,
        )

        # define model
        self.model = nn.ModuleDict(self.model)

    def update(self, output, opt):
        """
        output = {
            "f_img_c",
            "f_img_t",
            "f_text",
            "f_cit_t": {
                "repres",
                "repr_to_compare_with_source",
                "repr_to_compare_with_mods"
            }
        }
        """

        # assign input
        f_img_t_without_norm = output["f_img_t"]
        f_img_t = self.model["norm"](output["f_img_t"])  # target
        f_cit_t = self.model["norm"](output["f_cit_t"]["repres"])  # manipulated

        # loss
        loss = self.model["criterion"](f_img_t, f_cit_t)

        # rotational symmetry Loss
        conjugate_f_cit_t = self.compose_img_text(
            f_img_t_without_norm, output["f_text"]
        )
        composed_target_image_features = self.model["norm"](conjugate_f_cit_t["repres"])
        source_image_features = self.model["norm"](output["f_img_c"])

        loss += self.model["criterion"](
            composed_target_image_features, source_image_features
        )
        loss += self.l2_loss(
            output["f_cit_t"]["repr_to_compare_with_source"], output["f_img_c"]
        )
        loss += self.l2_loss(
            output["f_cit_t"]["repr_to_compare_with_mods"], output["f_text"]
        )

        # backward
        opt.zero_grad()
        loss.backward()
        opt.step()

        # return log
        log_data = dict()
        log_data["loss"] = float(loss.data)
        return log_data

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

        f_img_c = self.extract_image_feature(x["c_img"])
        f_text = self.extract_text_feature(x["mod_str"])
        f_cit_t = self.compose_img_text(f_img_c, f_text)
        return self.model["norm"](f_cit_t["repres"])

    def compose_img_text(self, f_img, f_text):
        return self.model["composeae_compose_it"]((f_img, f_text))

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
        f_img_c = self.extract_image_feature(x["c_img"])
        f_img_t = self.extract_image_feature(x["t_img"])
        f_text = self.extract_text_feature(x["mod_str"])
        f_cit_t = self.compose_img_text(f_img_c, f_text)

        return dict(f_img_c=f_img_c, f_img_t=f_img_t, f_text=f_text, f_cit_t=f_cit_t)


class ComplexProjectionModule(torch.nn.Module):
    def __init__(self, image_embed_dim=1024, text_embed_dim=1024):
        super().__init__()
        self.text_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(text_embed_dim),
            torch.nn.Linear(text_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
        )
        self.image_features = torch.nn.Sequential(
            torch.nn.BatchNorm1d(image_embed_dim),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
            torch.nn.Dropout(p=0.5),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
        )

    def forward(self, x):
        x1 = self.image_features(x[0])
        x2 = self.text_features(x[1])
        # default value of CONJUGATE is 1. Only for rotationally symmetric loss value is -1.
        # which results in the CONJUGATE of text features in the complex space
        CONJUGATE = x[2]
        num_samples = x[0].shape[0]
        CONJUGATE = CONJUGATE[:num_samples]
        delta = x2  # text as rotation
        re_delta = torch.cos(delta)
        im_delta = CONJUGATE * torch.sin(delta)

        re_score = x1 * re_delta
        im_score = x1 * im_delta

        concat_x = torch.cat([re_score, im_score], 1)
        x0copy = x[0].unsqueeze(1)
        x1 = x1.unsqueeze(1)
        x2 = x2.unsqueeze(1)
        re_score = re_score.unsqueeze(1)
        im_score = im_score.unsqueeze(1)

        return concat_x, x1, x2, x0copy, re_score, im_score


class LinearMapping(torch.nn.Module):
    """
    This is linear mapping to image space. rho(.)
    """

    def __init__(self, image_embed_dim=1024):
        super().__init__()
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
        )

    def forward(self, x):
        theta_linear = self.mapping(x[0])
        return theta_linear


class ConvMapping(torch.nn.Module):
    """
    This is convoultional mapping to image space. rho_conv(.)
    """

    def __init__(self, image_embed_dim=1024):
        super().__init__()
        self.image_embed_dim = image_embed_dim
        self.mapping = torch.nn.Sequential(
            torch.nn.BatchNorm1d(2 * image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(2 * image_embed_dim, image_embed_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(image_embed_dim, image_embed_dim),
        )
        # in_channels, output channels
        self.conv = torch.nn.Conv1d(5, 64, kernel_size=3, padding=1)
        self.adaptivepooling = torch.nn.AdaptiveMaxPool1d(2 * image_embed_dim // 64)

    def forward(self, x):
        concat_features = torch.cat(x[1:], 1)
        concat_x = self.conv(concat_features)
        concat_x = self.adaptivepooling(concat_x)
        final_vec = rearrange(concat_x, "b ... -> b (...)")
        theta_conv = self.mapping(final_vec)
        return theta_conv
