import datetime
import logging
import os
import sys

import torch
from tensorboardX import SummaryWriter
from torch import distributed as dist


class Logger(object):
    def __init__(self, log_path=None):
        self.log_fn = os.path.join(log_path, "train.log")
        self.summary_writer = SummaryWriter(log_path)
        self.prefix = ""

    def add_scalar(self, key, val, step):
        """write on tensorboard"""
        self.summary_writer.add_scalar(key, val, step)

    def print(self, content):
        """print log on cmd"""
        msg = self.prefix + content
        logging.info(msg)
        sys.stdout.flush()
        with open(self.log_fn, "a") as fp:
            fp.write(msg + "\n")
            fp.flush()
