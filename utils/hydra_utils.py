import json
import logging
import os
import pprint
import sys
import uuid
from datetime import datetime
from typing import Any, List

import torch
from omegaconf import DictConfig, OmegaConf

from misc.attr_dict import AttrDict


def is_hydra_available():
    """
    Check if Hydra is available. Simply python import to test.
    """
    try:
        import hydra  # NOQA

        hydra_available = True
    except ImportError:
        hydra_available = False
    return hydra_available


def print_cfg(cfg):
    """
    Supports printing both Hydra DictConfig and also the AttrDict config
    """
    logging.info("Training with config:")
    logging.getLogger().setLevel(logging.DEBUG)
    if isinstance(cfg, DictConfig):
        logging.info(cfg.pretty())
    else:
        logging.info(pprint.pformat(cfg))


def initialize_config(
    cfg: DictConfig, cmdline_args: List[Any] = None, infer_and_assert=True
):
    if cmdline_args:
        # convert the command line args to DictConfig
        sys.argv = cmdline_args
        cli_conf = OmegaConf.from_cli(cmdline_args)

        # merge the command line args with config
        cfg = OmegaConf.merge(cfg, cli_conf)

    # convert the config to AttrDict
    cfg = OmegaConf.to_container(cfg)
    cfg = AttrDict(cfg)

    # assert the config and infer
    cfg = cfg.config
    if infer_and_assert:
        cfg = infer_and_assert_hydra_config(cfg)

    return cfg


def infer_and_assert_hydra_config(cfg):
    # auto scale learning rate
    cfg.TRAIN.OPTIMIZER.lr.scaled_lr = cfg.TRAIN.OPTIMIZER.lr.base_lr
    if cfg.TRAIN.OPTIMIZER.lr.auto_scale:
        cfg.TRAIN.OPTIMIZER.lr.scaled_lr = (
            cfg.TRAIN.OPTIMIZER.lr.base_lr
            * cfg.TRAIN.BATCH_SIZE
            / float(cfg.TRAIN.OPTIMIZER.lr.base_lr_batch_size)
        )

    # model
    if cfg.TRAIN.MODEL.word_embedding_init == "bert":
        cfg.TRAIN.MODEL.in_feature_text = 1024
    elif cfg.TRAIN.MODEL.word_embedding_init == "gpt2-xl":
        cfg.TRAIN.MODEL.in_feature_text = 1600
    elif cfg.TRAIN.MODEL.word_embedding_init == "gpt-neo":
        cfg.TRAIN.MODEL.in_feature_text = 2048
    elif cfg.TRAIN.MODEL.word_embedding_init == "glove":
        cfg.TRAIN.MODEL.in_feature_text = 1100

    assert len(cfg.PROJ_ROOT) > 0

    # dataset-agnostic paths
    cfg.CKPT_PATH = os.path.join(cfg.PROJ_ROOT, "ckpt")
    cfg.LOGS_PATH = os.path.join(cfg.PROJ_ROOT, "logs")

    # logs path, ckpt path
    cfg.VERSION = str(uuid.uuid4()).strip().replace("-", "")[:6]
    return cfg
