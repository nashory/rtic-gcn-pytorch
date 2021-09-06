import logging
import os
import pickle

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm


def autotune_threshold(ratio, graph):
    threshold = 1.0
    r = 0.0
    for th in range(100, 0, -1):
        th = th / 100.0
        logging.info(
            f"Autotuning threshold for graph binarization ... (threshold={th}, target ratio={ratio})"
        )
        g = torch.where(
            graph >= th, torch.ones(graph.size()), torch.zeros(graph.size())
        )
        g = g.sum(1) / float(len(graph))
        r = g.mean()
        if r >= ratio:
            threshold = th
            break
    logging.info(f"> Found: threshold={threshold}, ratio={r}")
    return threshold, r


def build_graph(
    cfg,
    model,
    train_loader,
):
    # extract graph
    model.eval()
    mk2idx = {}
    X_img = []
    X_text = []
    F = []
    for bidx, data in enumerate(tqdm(train_loader, desc="Build Graph")):
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = data[k].to(f"cuda:{cfg.GPU_ID}")

        with torch.no_grad():
            f_img_c = model.extract_image_feature(data["c_img"])
            f_img_t = model.extract_image_feature(data["t_img"])
            f_text = model.extract_text_feature(data["mod_str"])
            f_cit_t = model.compose_img_text(f_img_c, f_text)

        for i in range(data["t_img"].size(0)):
            # target image feature
            _f_target = f_img_t[i].cpu().numpy()
            _f_target /= np.linalg.norm(_f_target)
            _f_composed = f_cit_t[i].squeeze().cpu().numpy()
            _f_composed /= np.linalg.norm(_f_composed)
            _f = np.concatenate([_f_target, _f_composed])
            _f /= np.linalg.norm(_f)

            F.append(_f)
            X_img.append(f_img_c[i].cpu())
            X_text.append(f_text[i].cpu())

            assert data["mod_key"][i] not in mk2idx
            mk2idx[data["mod_key"][i]] = len(F) - 1

    # node feature matrix
    X = torch.cat([torch.stack(X_img, dim=0), torch.stack(X_text, dim=0)], dim=1)

    F = torch.from_numpy(np.asarray(F))  # composed image/text feature
    A = torch.from_numpy(np.dot(F, F.T))
    A = (A + 1.0) / 2.0
    A = torch.clamp(A, 0.0, 1.0)

    ratio = cfg.TRAIN.MODEL.gcn_model.ratio

    # binarization (A) & pseudo-label (Z)
    threshold, ratio = autotune_threshold(ratio=ratio, graph=A)
    Z = torch.where(A >= threshold, torch.ones(A.size()), torch.zeros(A.size()))
    A = torch.where(A >= threshold, torch.ones(A.size()), torch.zeros(A.size()))

    # re-weighting
    A -= torch.eye(A.size(0))
    A /= torch.norm(A, p=1, dim=1, keepdim=True) + 1e-8
    A *= cfg.TRAIN.MODEL.gcn_model.tau
    A += torch.eye(A.size(0))

    # normalize adjacency matrix (A' = D^(1/2) * A * D^(1/2)
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    A = torch.matmul(torch.matmul(A, D).t(), D)

    graph_info = dict(
        X=X,
        A=A,
        Z=Z,
        threshold=threshold,
        mk2idx=mk2idx,
    )

    for k, _ in graph_info.items():
        if isinstance(graph_info[k], torch.Tensor):
            graph_info[k] = graph_info[k].to(f"cuda:{cfg.GPU_ID}")
    return graph_info
