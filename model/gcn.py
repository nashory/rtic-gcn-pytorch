import logging
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from misc.common import NormalizationLayer
from misc.loss import BatchBasedXentLoss, FocalLoss


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters_xavier()

    def reset_parameters_uniform(self):
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def reset_parameters_xavier(self):
        nn.init.xavier_normal_(self.weight.data, gain=0.02)
        if self.bias is not None:
            nn.init.constant_(self.bias.data, 0.0)

    def forward(self, x, adj):
        support = torch.mm(x, self.weight)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            output += self.bias
        return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )


class GCNStreamModule(nn.Module):
    def __init__(self, cfg, out_c, graph_info):
        """
        GCNStreamModule
        """
        super(GCNStreamModule, self).__init__()
        self.cfg = cfg
        self.X = graph_info["X"]
        self.A = graph_info["A"]
        self.act = nn.ReLU()

        self.model = dict()
        in_c_img = self.X.size(1) // 2
        in_c_text = self.X.size(1) // 2

        model_type = cfg.TRAIN.MODEL.composer_model.name
        params = cfg.TRAIN.MODEL.composer_model.params[model_type]

        if model_type == "rtic":
            from model.rtic import RticCompositionModule

            self.model["composer"] = RticCompositionModule(
                in_c_img=in_c_img,
                in_c_text=in_c_text,
                n_blocks=params.n_blocks,
            )
        elif model_type == "tirg":
            from model.tirg import TirgCompositionModule

            self.model["composer"] = TirgCompositionModule(
                in_c=self.X.shape[1],
                out_c=out_c,
            )
        elif model_type == "mrn":
            from model.mrn import MrnCompositionModule

            self.model["composer"] = MrnCompositionModule(
                in_c=self.X.shape[1] // 2,
                out_c=out_c,
            )
        elif model_type == "param_hash":
            from model.parameter_hashing import ParameterHashingCompositionModule

            self.model["composer"] = ParameterHashingCompositionModule(
                in_c_text=self.X.shape[1] // 2,
                in_c_img=self.X.shape[1] // 2,
            )
        elif model_type == "compose_ae":
            from model.compose_ae import ComposeAECompositionModule

            self.model["composer"] = ComposeAECompositionModule(
                in_c_text=self.X.shape[1] // 2,
                in_c_img=self.X.shape[1] // 2,
            )
        else:
            raise ValueError(f"Not supported model type: {model_type}")

        self.model["gcn_1"] = GraphConvolution(
            out_c, self.cfg.TRAIN.MODEL.gcn_model.gcn_hidden_dim
        )
        self.model["gcn_2"] = GraphConvolution(
            self.cfg.TRAIN.MODEL.gcn_model.gcn_hidden_dim, out_c
        )

        self.model = nn.ModuleDict(self.model)

    def get_config_optim(self, lr):
        params = []
        for k, v in self.model.items():
            params.append(
                {
                    "params": v.parameters(),
                    "lr": lr,
                    "lrp": self.cfg.TRAIN.MODEL.gcn_model.lrp,
                }
            )
        return params

    def forward(self, x):

        input_v = self.X[:, : self.X.size(1) // 2]
        input_t = self.X[:, self.X.size(1) // 2 :]

        # composer
        try:
            f = self.model["composer"]((input_v, input_t))["repres"]  # composeae
        except:
            f = self.model["composer"]((input_v, input_t))

        # gcn
        f = self.model["gcn_1"](f, self.A)
        f = self.act(f)
        f = self.model["gcn_2"](f, self.A)

        x = torch.matmul(x, f.t())
        return torch.sigmoid(x)


class JointTrainingWithGCNStream(nn.Module):
    def __init__(self, cfg, composer, gcn_module, graph_info):
        super(JointTrainingWithGCNStream, self).__init__()
        self.cfg = cfg
        self.graph_info = graph_info

        self.dml_loss = BatchBasedXentLoss()

        if self.cfg.TRAIN.MODEL.gcn_model.weight_balance:
            self.gcn_loss = nn.BCELoss(reduction="none")
        else:
            self.gcn_loss = nn.BCELoss()

        # define model
        self.model = dict()
        self.model["composer"] = composer
        self.model[f"gcn_module"] = gcn_module
        self.model = nn.ModuleDict(self.model)

    def save(self, path, state={}):
        self.model["composer"].save(path, state)

    def load(self, path):
        pass

    def get_original_image_feature(self, x):
        """
        data = {
            'img': img,
            'iid': iid,
        }
        """
        x = self.model["composer"].extract_image_feature(x["img"])
        return self.model["composer"].model["norm"](x)

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
        f_img = self.model["composer"].extract_image_feature(x["c_img"])
        f_text = self.model["composer"].extract_text_feature(x["mod_str"])
        x = self.compose_img_text(f_img, f_text)
        return self.model["composer"].model["norm"](x)

    def get_config_optim(self, lr):
        params = []
        for k, v in self.model.items():
            if self.cfg.TRAIN.MODEL.gcn_model.load_pretrained_composer:
                if k == "composer":
                    # smaller lr for pretrained composer.
                    params += self.model[k].get_config_optim(
                        lr, fixed_lrp=self.cfg.TRAIN.MODEL.gcn_model.fixed_lrp
                    )
                else:
                    params += self.model[k].get_config_optim(lr)
            else:
                params += self.model[k].get_config_optim(lr)
        return params

    def sample_pseudo_label(self, mod_keys, Z):
        y = torch.zeros((len(mod_keys), Z.size(1))).to("cuda")
        for i, mk in enumerate(mod_keys):
            try:
                idx = self.graph_info["mk2idx"][mk]
                y[i] = Z[idx]
            except Exception as err:
                raise ValueError(f"Error Found. Is it intended? : {err}")
        return y

    def update(self, x, opt):
        """
        input = (f_img_c, f_img_t, f_cit_t, f_gcn_logit, f_gcn_label)
        """
        # assign input
        f_img_t = self.model["composer"].model["norm"](x[1])  # target
        f_cit_t = self.model["composer"].model["norm"](x[2])  # manipulated
        f_gcn_logit = x[3]  # logit for gcn node classification
        f_gcn_label = x[4]  # label for gcn node classification

        # dml loss
        dml_loss = self.dml_loss(f_img_t, f_cit_t)

        # gcn loss
        if self.cfg.TRAIN.MODEL.gcn_model.weight_balance:
            lpos = f_gcn_label == 1
            lneg = f_gcn_label == 0
            wp = 0.5 * lpos / (lpos.sum() + 1e-8)
            wn = 0.5 * lneg / (lneg.sum() + 1e-8)
            w = wp + wn
            gcn_loss = w * self.gcn_loss(f_gcn_logit.squeeze(), f_gcn_label.squeeze())
            gcn_loss = gcn_loss.sum()
        else:
            gcn_loss = self.gcn_loss(f_gcn_logit, f_gcn_label)
        loss = dml_loss + gcn_loss

        # terminate process if gradient overflow occurs.
        if (
            self.cfg.TRAIN.MODEL.gcn_model.terminate_when_gradient_overflow
            and opt.epoch > 0
            and loss > 30.0
        ):
            logging.info("Terminate process due to gradient overflow ...")
            logging.info("Kill Process")
            sys.exit(1)

        # backward
        opt.zero_grad()
        loss.backward()

        if self.cfg.TRAIN.MODEL.gcn_model.gradient_clipping:
            # check if gradients are not too large
            nn.utils.clip_grad_norm(
                self.model.parameters(), max_norm=10.0
            )  # this prevents gradient exploding when applied for some methods (e.g. TIRG)
        opt.step()

        # return log
        log_data = dict()
        log_data["dml_loss"] = float(dml_loss.data)
        log_data["gcn_loss"] = float(gcn_loss.data)
        log_data["loss"] = float(loss.data)
        return log_data

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
        out = self.model["composer"](x)
        f_img_c, f_img_t, f_cit_t, f_text = (
            out["f_img_c"],
            out["f_img_t"],
            out["f_cit_t"],
            out["f_text"],
        )

        f_gcn_label = self.sample_pseudo_label(x["mod_key"], self.graph_info["Z"])
        f_gcn_logit = self.model[f"gcn_module"](f_cit_t)
        return (f_img_c, f_img_t, f_cit_t, f_gcn_logit, f_gcn_label)

    def compose_img_text(self, f_img, f_text):
        try:
            return self.model["composer"].compose_img_text(f_img, f_text)[
                "repres"
            ]  # composeae
        except:
            return self.model["composer"].compose_img_text(f_img, f_text)
