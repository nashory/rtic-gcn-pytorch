import json
import logging
import math
import os
from pprint import pprint

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm


class Trainer:
    def __init__(self, cfg, data_loader, model, opt, logger=None):
        self.cfg = cfg
        self.data_loader = data_loader
        self.model = model
        self.opt = opt
        self.logger = logger

        self.processed_images = 0
        self.global_step = 0

    def __adjust_lr__(self, epoch, **kwargs):
        lr = self.cfg.TRAIN.OPTIMIZER.lr.scaled_lr
        if self.cfg.TRAIN.OPTIMIZER.policy == "step_lr":
            for e in self.cfg.TRAIN.OPTIMIZER.lr_decay_steps:
                if epoch >= e:
                    lr *= self.cfg.TRAIN.OPTIMIZER.lr_decay_factor
        elif self.cfg.TRAIN.OPTIMIZER.policy == "linear_warmup_cosine":
            total_step = float(self.cfg.TRAIN.OPTIMIZER.total_step)
            warmup_step = self.cfg.TRAIN.OPTIMIZER.warmup_ratio * total_step
            if self.global_step < warmup_step:
                lr = lr * self.global_step / warmup_step
            else:
                step = (self.global_step - warmup_step) / (total_step - warmup_step)
                lr = lr * 0.5 * (math.cos(math.pi * step) + 1)
        else:
            raise NotImplementedError(
                f"Unsupported LR Scheduling Policy: {self.cfg.TRAIN.OPTIMIZER.policy}"
            )
        for param_group in self.opt.param_groups:
            param_group["lr"] = lr * param_group["lrp"]
        self.cur_lr = lr

    def __logging__(self, log_data):
        msg = f"[Train][{self.cfg.EXPR_NAME}/{self.cfg.VERSION}]"
        msg += f"[Epoch: {self.epoch}]"
        msg += f"[Lr:{self.cur_lr:.8f}]"
        log_data["lr"] = self.cur_lr
        for k, v in log_data.items():
            if self.logger is not None:
                self.logger.add_scalar(k, v, self.global_step)
            if isinstance(v, float):
                msg += f" {k}:{v:.4f}"
            else:
                msg += f" {k}:{v}"
        return msg

    def train(self, epoch):
        self.epoch = epoch
        self.opt.epoch = epoch
        self.model.train()
        for bidx, data in enumerate(tqdm(self.data_loader, desc="Train")):
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
            bs = data["c_img"][0].size(0)
            self.global_step += 1
            self.processed_images += bs
            self.__adjust_lr__(epoch)

            # data
            for k, v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = data[k].to(f"cuda:{self.cfg.GPU_ID}")

            # forward and update
            output = self.model(data)
            log_data = self.model.update(output, self.opt)
            msg = self.__logging__(log_data)
            if (bidx % self.cfg.LOGGING.PRINT_FREQ) == 0:
                logging.info(msg)
