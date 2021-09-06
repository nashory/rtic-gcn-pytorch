#!/usr/bin/python

import argparse
import json
import logging
import os
import random
import sys
import time
from pprint import pprint

import easydict
import torch

from utils import main_utils
from utils.gcn_utils import build_graph
from utils.logger_utils import Logger
from utils.main_utils import (
    init_data_loader,
    init_env,
    init_gcn_model,
    init_hydra_config,
    init_model,
    init_optimizer,
    init_runner,
)


# main
def main():
    # prepare environment
    cfg = init_hydra_config()
    cfg = init_env(cfg)

    # build train/val data loader
    train_loader, dataset_info, cfg = init_data_loader(cfg, is_train=True)
    test_loader = {}
    for target_name in dataset_info["target_names"]:
        test_loader[target_name], _, _ = init_data_loader(
            cfg,
            is_train=False,
            target_name=target_name,
        )

    # build model
    if cfg.GCN_MODE:
        # extract graph information
        assert len(cfg.LOAD_FROM) > 0
        graph_loader, _, _ = init_data_loader(cfg, is_train=True, is_graph_infer=True)
        with open(os.path.join(cfg.CKPT_PATH, cfg.LOAD_FROM, "config.json"), "r") as f:
            pretrained_cfg = easydict.EasyDict(json.load(f))
        texts = torch.load(
            os.path.join(cfg.CKPT_PATH, cfg.LOAD_FROM, "best_model.pth")
        )[
            "texts"
        ]  # in order to keep the same order in vocab.
        assert len(texts) == len(dataset_info["texts"])
        pretrained_model = main_utils.init_model(
            pretrained_cfg, texts, load_from=cfg.LOAD_FROM
        )
        graph_info = build_graph(
            cfg,
            model=pretrained_model,
            train_loader=graph_loader,
        )

        del pretrained_model
        torch.cuda.empty_cache()

        # build model
        model = init_gcn_model(
            cfg,
            pretrained_cfg,
            texts=texts,
            graph_info=graph_info,
        )
    else:
        if len(cfg.LOAD_FROM) > 0:
            load_from = cfg.LOAD_FROM
            texts = torch.load(
                os.path.join(cfg.CKPT_PATH, cfg.LOAD_FROM, "best_model.pth")
            )[
                "texts"
            ]  # in order to keep the same order in vocab.
        else:
            load_from = None
            texts = dataset_info["texts"]

        assert len(texts) == len(dataset_info["texts"])
        model = init_model(cfg, texts=texts, load_from=load_from)

    # logger
    log_path = os.path.join(cfg.LOGS_PATH, cfg.EXPR_NAME, cfg.VERSION)
    logger = Logger(log_path)

    # optimizer
    opt = init_optimizer(cfg, model)

    # train/test for N-epochs.
    trainer, evaluator = init_runner(
        cfg=cfg,
        train_loader=train_loader,
        test_loader=test_loader,
        model=model,
        opt=opt,
        logger=logger,
    )

    for epoch in range(cfg.TRAIN.MAX_EPOCHS):
        evaluator.test(epoch)
        trainer.train(epoch)
    evaluator.test(cfg.TRAIN.MAX_EPOCHS)
    logging.info("Congrats! You just finished traininig.")


if __name__ == "__main__":
    main()
