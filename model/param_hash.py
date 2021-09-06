"""paramter_hashing.py
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import xxhash

from model.base import ImageEncoderTextEncoderBase


class ParameterHashingCompositionModule(nn.Module):
    def __init__(self, in_c_text, in_c_img):
        super(ParameterHashingCompositionModule, self).__init__()
        self.hash_linear = HashLinear(in_c_text, in_c_img)
        self.multimodal_composer = nn.Sequential(
            nn.BatchNorm1d(in_c_img),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(in_c_img, in_c_img),
        )

    def forward(self, x):
        f_img, f_text = x

        # Parameter hashing
        hashed_out = self.hash_linear(f_text)

        # Composing
        gate_out = f_img + hashed_out  # nn.CAddTable at original code
        out = self.multimodal_composer(gate_out)
        return out


class ParameterHashingModel(ImageEncoderTextEncoderBase):
    def __init__(self, cfg, texts):
        super(ParameterHashingModel, self).__init__(cfg, texts)

        self.model["paramhash_compose_it"] = ParameterHashingCompositionModule(
            in_c_text=self.out_feature_text,
            in_c_img=self.out_feature_image,
        )
        self.model = nn.ModuleDict(self.model)

    def compose_img_text(self, f_img, f_text):
        return self.model["paramhash_compose_it"]((f_img, f_text))


class HashLinear(nn.Module):
    """https://github.com/jfainberg/hashed_nets

    This layer implements a linear hashed network as in
     - Chen, W., Wilson, J., Tyree, S., Weinberger, K. and Chen, Y., 2015,
       Compressing neural networks with the hashing trick.
       In International Conference on Machine Learning (pp. 2285-2294).

    It is largely based on the above authors (Lua)Torch implementation:
    https://www.cse.wustl.edu/~ychen/HashedNets

    Note that some static hashing parameters are wrapped with
    `Parameter(..., requires_grad=False)` so that they get sent
    to device along with the layer. I.e. check for requires_grad
    when computing total number of parameters.
    """

    # compression=0.015625
    def __init__(
        self,
        in_features,
        out_features,
        compression=1,
        xi=True,
        hash_bias=True,
        bias=True,
        hash_seed=2,
    ):
        super(HashLinear, self).__init__()

        self.hash_seed = hash_seed
        self.use_bias = bias
        self.hash_bias = hash_bias
        self.xi = xi

        self.in_features = in_features
        self.out_features = out_features

        #  Virtual sizes
        self.size_w = in_features * out_features
        self.size_b = out_features

        #  Compressed sizes
        self.hsize_w = math.ceil(self.size_w * compression)
        if self.hash_bias:
            self.hsize_b = math.ceil(self.size_b * compression)
        else:
            self.hsize_b = self.size_b

        self.h_weight = nn.Parameter(torch.Tensor(self.hsize_w))
        if bias:
            self.h_bias = nn.Parameter(torch.Tensor(self.hsize_b))

        self.xxhash = xxhash

        self.hash_config("W")
        if bias and self.hash_bias:
            self.hash_config("B")

        self.reset_parameters()

    def hash_config(self, WorB):
        """
        Returns virtual matrices with indices into the compressed
        size given by the hashing function.
        """
        assert WorB == "W" or WorB == "B"

        if WorB == "W":
            h_size = self.hsize_w
            dim1 = self.out_features
            dim2 = self.in_features
            self.idxW = self.hash_func(h_size, dim1, dim2, "idxW")
        elif WorB == "B":
            h_size = self.hsize_b
            dim1 = self.out_features
            dim2 = 1
            self.idxB = self.hash_func(h_size, dim1, dim2, "idxB").squeeze()

        if self.xi:
            # Returns 1 and -1
            if WorB == "W":
                self.xiW = nn.Parameter(
                    self.hash_func(2, dim1, dim2, "xiW").add(1).mul(2).add(-3).float(),
                    requires_grad=False,
                )
            elif WorB == "B":
                self.xiB = nn.Parameter(
                    self.hash_func(2, dim1, dim2, "xiB")
                    .add(1)
                    .mul(2)
                    .add(-3)
                    .float()
                    .squeeze(),
                    requires_grad=False,
                )

    def hash_func(self, hN, size_out, size_in, extra_str=""):
        """
        Hash matrix indices to an index in the compressed vector
        representation.

        Returns a matrix of indices with size size_out x size_in,
        where the indices are in the range [0,hN).
        """
        idx = torch.LongTensor(size_out, size_in)
        for i in range(size_out):
            for j in range(size_in):
                key = "{}_{}{}".format(i, j, extra_str)

                # Wrap hashed values to the compressed range
                idx[i, j] = self.xxhash.xxh32(key, self.hash_seed).intdigest() % hN

        return idx

    def reset_parameters(self):
        stdv = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.h_weight, -stdv, stdv)
        if self.use_bias:
            nn.init.uniform_(self.h_bias, -stdv, stdv)

    def forward(self, x):
        # self.idxW is a matrix of the full size of type LongTensor,
        # which contains indices that selects from the elements in h_weight

        if self.use_bias:
            if self.hash_bias:
                if self.xi:
                    return F.linear(
                        x,
                        self.h_weight[self.idxW] * self.xiW,
                        self.h_bias[self.idxB] * self.xiB,
                    )
                else:
                    return F.linear(x, self.h_weight[self.idxW], self.h_bias[self.idxB])
            else:
                if self.xi:
                    return F.linear(x, self.h_weight[self.idxW] * self.xiW, self.h_bias)
                else:
                    return F.linear(x, self.h_weight[self.idxW], self.h_bias)
        else:
            if self.xi:
                return F.linear(x, self.h_weight[self.idxW] * self.xiW, None)
            else:
                return F.linear(x, self.h_weight[self.idxW], None)
