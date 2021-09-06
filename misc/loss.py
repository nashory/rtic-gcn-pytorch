#!/usr/bin/env python
# encoding: utf-8

import math
import time
import warnings

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

warnings.filterwarnings("ignore")

from torch import nn
from torch.nn import Parameter


class FocalLoss(nn.Module):
    def __init__(self, gamma=1.0, logits=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.logits = logits

    def forward(self, x, y):
        if not self.logits:
            x = torch.sigmoid(x)
        if self.logits:
            bce_loss = F.binary_cross_entropy_with_logits(x, y, reduction="none")
        else:
            bce_loss = F.binary_cross_entropy(x, y, reduction="none")
        pt = torch.exp(-bce_loss)
        loss = torch.pow((1 - pt), self.gamma) * bce_loss
        return loss.mean()


class BatchHardTripletLoss(nn.Module):
    def __init__(self, margin=1, p=2, squared=True, soft=False):
        super(BatchHardTripletLoss, self).__init__()
        self.margin = margin
        self.p = p
        self.squared = squared
        self.soft = soft

    def __pairwise_distances__(self, embeddings, squared=False, p=2):
        assert p == 1 or p == 2

        if p == 2:
            dot_product = torch.mm(embeddings, embeddings.transpose(0, 1))
            square_norm = torch.diag(dot_product)
            distances = (
                square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
            )
            distances = torch.clamp(distances, min=0.0)

            if not squared:
                mask = torch.eq(distances, 0.0).float()
                distances = distances + mask * 1e-6
                distances = torch.sqrt(distances)
                distances = distances * (1.0 - mask)

        elif p == 1:
            abs = torch.abs(embeddings.unsqueeze(0) - embeddings.unsqueeze(1))
            distances = torch.sum(abs, dim=2)

        return distances

    def forward(self, x1, x2):
        pairwise_dist = self.__pairwise_distances__(
            torch.cat([x1, x2]), p=self.p, squared=self.squared
        )

        labels = torch.Tensor(list(range(x1.shape[0])) + list(range(x2.shape[0])))
        mask_anchor_positive = self.__get_anchor_positive_triplet_mask__(labels).float()
        anchor_positive_dist = torch.mul(mask_anchor_positive, pairwise_dist)
        hardest_positive_dist = torch.max(anchor_positive_dist, dim=1, keepdim=True)[0]

        mask_anchor_negative = self.__get_anchor_negative_triplet_mask__(labels).float()
        max_anchor_negative_dist = torch.max(pairwise_dist, dim=1, keepdim=True)[0]
        anchor_negative_dist = pairwise_dist + max_anchor_negative_dist * (
            1.0 - mask_anchor_negative
        )
        hardest_negative_dist = torch.min(anchor_negative_dist, dim=1, keepdim=True)[0]

        if self.soft:
            triplet_loss = torch.log(
                hardest_positive_dist - hardest_negative_dist + self.margin
            )
        else:
            triplet_loss = hardest_positive_dist - hardest_negative_dist + self.margin
        triplet_loss = torch.max(triplet_loss, torch.zeros_like(triplet_loss))
        triplet_loss = torch.mean(triplet_loss)

        return triplet_loss

    def __get_anchor_positive_triplet_mask__(self, labels):
        indices_equal = torch.eye(labels.shape[0]).bool()
        indices_not_equal = ~indices_equal
        labels_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1))
        mask = indices_not_equal.__and__(labels_equal)
        return mask.to("cuda")

    def __get_anchor_negative_triplet_mask__(self, labels):
        labels_equal = torch.eq(labels.unsqueeze(0), labels.unsqueeze(1)).bool()
        mask = ~labels_equal
        return mask.to("cuda")


class BatchBasedXentLoss(nn.Module):
    def __init__(self):
        super(BatchBasedXentLoss, self).__init__()

    def forward(self, x1, x2):
        x = torch.mm(x1, x2.transpose(0, 1))
        labels = torch.tensor(list(range(x.size(0)))).long()
        labels = torch.autograd.Variable(labels).to("cuda")
        return F.cross_entropy(x, labels)


class TripletLoss(torch.nn.Module):
    """Class for the triplet loss."""

    def __init__(self, pre_layer=None):
        super(TripletLoss, self).__init__()
        self.pre_layer = pre_layer

    # def forward(self, x, triplets):
    def forward(self, mod_img1, img2):
        x = torch.cat([mod_img1, img2])
        if self.pre_layer is not None:
            x = self.pre_layer(x)

        triplets = self.__get_triplet(mod_img1, img2)
        loss = MyTripletLossFunc(triplets)(x)

        return loss

    def __get_triplet(self, mod_img1, img2):
        triplets = []
        labels = list(range(mod_img1.shape[0])) + list(range(img2.shape[0]))
        for i in range(len(labels)):
            triplets_i = []
            for j in range(len(labels)):
                if labels[i] == labels[j] and i != j:
                    for k in range(len(labels)):
                        if labels[i] != labels[k]:
                            triplets_i.append([i, j, k])
            np.random.shuffle(triplets_i)
            triplets += triplets_i[:3]
        assert triplets and len(triplets) < 2000

        return triplets


class MyTripletLossFunc(torch.autograd.Function):
    def __init__(self, triplets):
        super(MyTripletLossFunc, self).__init__()
        self.triplets = triplets
        self.triplet_count = len(triplets)

    def forward(self, features):
        self.save_for_backward(features)
        self.distances = pairwise_distances(features).cpu().numpy()

        loss = 0.0
        triplet_count = 0.0
        correct_count = 0.0
        for i, j, k in self.triplets:
            w = 1.0
            triplet_count += w
            loss += w * np.log(1 + np.exp(self.distances[i, j] - self.distances[i, k]))
            if self.distances[i, j] < self.distances[i, k]:
                correct_count += 1

        loss /= triplet_count
        return torch.FloatTensor((loss,)).to()

    def backward(self, grad_output):
        (features,) = self.saved_tensors
        features_np = features.cpu().numpy()
        grad_features = features.clone() * 0.0
        grad_features_np = grad_features.cpu().numpy()

        for i, j, k in self.triplets:
            w = 1.0
            f = 1.0 - 1.0 / (1.0 + np.exp(self.distances[i, j] - self.distances[i, k]))
            grad_features_np[i, :] += (
                w * f * (features_np[i, :] - features_np[j, :]) / self.triplet_count
            )
            grad_features_np[j, :] += (
                w * f * (features_np[j, :] - features_np[i, :]) / self.triplet_count
            )
            grad_features_np[i, :] += (
                -w * f * (features_np[i, :] - features_np[k, :]) / self.triplet_count
            )
            grad_features_np[k, :] += (
                -w * f * (features_np[k, :] - features_np[i, :]) / self.triplet_count
            )

        for i in range(features_np.shape[0]):
            grad_features[i, :] = torch.from_numpy(grad_features_np[i, :])
        grad_features *= float(grad_output.data[0])
        return grad_features


def pairwise_distances(x, y=None):
    """
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between
    x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    source:
    https://discuss.pytorch.org/t/efficient-distance-matrix-computation/9065/2
    """
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_t = torch.transpose(y, 0, 1)
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y_t = torch.transpose(x, 0, 1)
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, y_t)
    # Ensure diagonal is zero if x=y
    # if y is None:
    #     dist = dist - torch.diag(dist.diag)
    return torch.clamp(dist, 0.0, np.inf)
