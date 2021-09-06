import logging
import os
import random
import sys
from pprint import pprint

import hydra
import numpy as np
import torch
import torch.multiprocessing as mp

import dataset as Dataset
import model as M
import runner
from utils.hydra_utils import initialize_config, print_cfg


def init_hydra_config():
    # set logging level
    logging.getLogger().setLevel(logging.DEBUG)

    overrides = sys.argv[1:]
    logging.info(f"####### overrides: {overrides}")
    with hydra.initialize_config_module(config_module="cfg"):
        cfg = hydra.compose("default", overrides=overrides)

    cfg = initialize_config(cfg)
    print_cfg(cfg)
    return cfg


def init_env(cfg):
    if cfg.SEED is None:
        cfg.SEED = random.randint(0, 12345)
    random.seed(cfg.SEED)
    np.random.seed(cfg.SEED)
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.set_device(f"cuda:{cfg.GPU_ID}")
        torch.backends.cudnn.benchmark = True  # speed up training.
    return cfg


def init_data_loader(cfg, is_train, is_graph_infer=False, target_name=None):
    shuffle = True
    if is_train:
        if is_graph_infer:
            shuffle = False
    else:
        shuffle = False
    num_workers = min(10, mp.cpu_count())
    drop_last = True if is_train else False
    dataset = Dataset.FashionIQDataset(
        cfg,
        mode=None,
        is_train=is_train,
        is_graph_infer=is_graph_infer,
        target_name=target_name,
    )
    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=False,
    )
    dataset_info = {
        "target_names": data_loader.dataset.target_names,
        "texts": data_loader.dataset.get_all_texts(),
    }
    if (
        is_train
        and not is_graph_infer
        and cfg.TRAIN.OPTIMIZER.policy == "linear_warmup_cosine"
    ):
        cfg.TRAIN.OPTIMIZER.total_step = cfg.TRAIN.MAX_EPOCHS * len(data_loader)

    return data_loader, dataset_info, cfg


def init_model(cfg, texts, load_from=None):
    name = cfg.TRAIN.MODEL.composer_model.name

    model_args = {
        "cfg": cfg,
        "texts": texts,
    }
    if name == "image_only":
        m = M.SimpleImageOnlyModel
    elif name == "text_only":
        m = M.SimpleTextOnlyModel
    elif name == "concat":
        m = M.SimpleConcatModel
    elif name == "mrn":
        m = M.MRN
    elif name == "film":
        m = M.FiLM
    elif name == "tirg":
        m = M.TIRG
    elif name == "compose_ae":
        m = M.ComposeAE
    elif name == "rtic":
        m = M.RTIC
    elif name == "param_hash":
        m = M.ParameterHashingModel
    elif name in {"block", "mutan", "mlb", "mfb", "mfh", "mcb"}:
        m = M.FeatureFusionMethod
    else:
        raise NotImplementedError()

    model = m(**model_args)
    if torch.cuda.is_available():
        model = model.to(f"cuda:{cfg.GPU_ID}")

    if load_from is not None:
        logging.info(f"Load pretrained model from checkpoint: {load_from}")
        model.load(os.path.join(cfg.CKPT_PATH, load_from, "best_model.pth"))
    return model


def init_gcn_model(cfg, pretraiend_cfg, texts, graph_info):
    from model.gcn import JointTrainingWithGCNStream

    # composer
    if cfg.TRAIN.MODEL.gcn_model.load_pretrained_composer:
        # use weights of pretrained composer for finetuning.
        # in this case, the composer architecture is forced to follow the one in the previous stage.
        composer = init_model(pretraiend_cfg, texts, load_from=cfg.LOAD_FROM)
    else:
        # train composer from scratch.
        composer = init_model(cfg, texts, load_from=None)

    # gcn_module
    from model.gcn import GCNStreamModule

    gcn_module = GCNStreamModule(
        cfg=cfg,
        out_c=composer.out_feature_image,
        graph_info=graph_info,
    )

    model = JointTrainingWithGCNStream(
        cfg=cfg,
        composer=composer,
        gcn_module=gcn_module,
        graph_info=graph_info,
    )

    if torch.cuda.is_available():
        model = model.to(f"cuda:{cfg.GPU_ID}")

    return model


def init_optimizer(cfg, model):

    lr = cfg.TRAIN.OPTIMIZER.lr.scaled_lr
    params = model.get_config_optim(lr)

    if cfg.TRAIN.OPTIMIZER.name == "sgd":
        opt = torch.optim.SGD(
            params,
            lr=lr,
            momentum=cfg.TRAIN.OPTIMIZER.momentum,
            weight_decay=1e-6,
        )
    elif cfg.TRAIN.OPTIMIZER.name == "adam":
        opt = torch.optim.Adam(
            params,
            lr=lr,
            betas=[cfg.TRAIN.OPTIMIZER.beta1, cfg.TRAIN.OPTIMIZER.beta2],
        )
    elif cfg.TRAIN.OPTIMIZER.name == "adamw":
        opt = torch.optim.AdamW(
            params,
            lr=lr,
            betas=[cfg.TRAIN.OPTIMIZER.beta1, cfg.TRAIN.OPTIMIZER.beta2],
        )
    else:
        raise NotImplementedError
    return opt


def init_runner(cfg, train_loader, test_loader, model, opt, logger):
    trainer = runner.Trainer(cfg, train_loader, model, opt, logger)
    evaluator = runner.Evaluator(cfg, test_loader, model, opt, logger)
    return trainer, evaluator
