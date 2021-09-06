from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import os
import sys

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.utils.data as data
import torchvision.datasets as D
import torchvision.models as M
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image

from utils.transform_utils import PaddedResize


class BaseDataset(object):
    def __init__(self, cfg, mode, is_train, is_graph_infer):
        self.cfg = cfg
        self.mode = mode
        self.is_train = is_train
        self.is_graph_infer = is_graph_infer
        self.data_root = cfg.DATA_ROOT
        self.image_size = cfg.IMAGE_SIZE
        self.crop_size = cfg.CROP_SIZE
        self.transform_style = cfg.TRAIN.TRANSFORM_STYLE
        self.transform = None
        self.all_texts = []

    def __len__(self):
        raise NotImplementedError()

    def __getitem__(self, index):
        if self.is_train:
            return self.__sample_train__(index)
        else:
            if self.mode == "query":
                return self.__sample_query__(index)
            elif self.mode == "index":
                return self.__sample_index__(index)

    def __load_train_data__(self):
        raise NotImplementedError()

    def __load_test_data__(self):
        raise NotImplementedError()

    def __sample_train__(self, index):
        raise NotImplementedError()

    def __sample_query__(self, index):
        raise NotImplementedError()

    def __sample_index__(self, index):
        raise NotImplementedError()

    def __load_pil_image__(self, path):
        try:
            with open(path, "rb") as f:
                img = Image.open(f)
                return img.convert("RGB")
        except Exception as err:
            logging.info(err)
            img = Image.new("RGB", (224, 224))
            return img

    def __set_transform__(self):
        IMAGE_SIZE = self.image_size
        if self.is_train:
            if self.transform_style == "optimize":
                if self.is_graph_infer:
                    IMAGE_SIZE = int(
                        self.image_size * 1.08
                    )  # use slightly larger image for inference.
        else:
            if self.transform_style == "optimize":
                IMAGE_SIZE = int(
                    self.image_size * 1.08
                )  # use slightly larger image for inference.

        if self.is_train and not self.is_graph_infer:
            if self.transform_style == "standard":
                self.transform = T.Compose(
                    [
                        PaddedResize(IMAGE_SIZE),
                        T.RandomCrop(self.crop_size),
                        T.RandomHorizontalFlip(p=0.5),
                        T.RandomAffine(
                            degrees=45, translate=(0.15, 0.15), scale=(0.9, 1.1)
                        ),
                        T.ToTensor(),
                        T.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
            elif self.transform_style == "optimize":
                self.transform = T.Compose(
                    [
                        PaddedResize(IMAGE_SIZE),
                        T.RandomHorizontalFlip(p=0.5),
                        T.RandomAffine(
                            degrees=45, translate=(0.15, 0.15), scale=(0.9, 1.1)
                        ),
                        T.ToTensor(),
                        T.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
        else:
            if self.transform_style == "standard":
                self.transform = T.Compose(
                    [
                        PaddedResize(IMAGE_SIZE),
                        T.CenterCrop(self.crop_size),
                        T.ToTensor(),
                        T.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
            elif self.transform_style == "optimize":
                self.transform = T.Compose(
                    [
                        PaddedResize(IMAGE_SIZE),
                        T.ToTensor(),
                        T.Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

    def refresh(self):
        self.__init_data__()

    def set_mode(self, mode):
        assert mode in ["query", "index"]
        self.mode = mode

    def get_all_texts(self):
        return sorted(
            self.all_texts
        )  # make sure the texts are returned in sorted order.
