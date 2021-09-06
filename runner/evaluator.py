import json
import os

import numpy as np
import torch
from torch.autograd import Variable
from tqdm import tqdm


class Evaluator(object):
    def __init__(self, cfg, data_loader, model, opt, logger):
        self.cfg = cfg
        self.data_loader = data_loader
        self.model = model
        self.opt = opt
        self.logger = logger
        self.test_freq = cfg.LOGGING.TEST_FREQ
        self.best_score = 0.0
        self.ckpt_path = os.path.join(cfg.CKPT_PATH, cfg.EXPR_NAME, cfg.VERSION)
        os.makedirs(self.ckpt_path, exist_ok=True)
        self.targets = list(self.data_loader.keys())
        self.score_log = {}

    def test(self, epoch):
        if not epoch % self.test_freq == 0:
            return
        # test.
        self.epoch = epoch
        self.model.eval()
        r10 = 0.0
        r50 = 0.0
        r10r50 = 0.0
        for target, data_loader in self.data_loader.items():
            q_feats = []
            q_ids = []
            i_feats = []
            i_caps = []
            i_ids = []
            gt = []
            q_feats_dict = {}
            i_feats_dict = {}

            # compute query features
            data_loader.dataset.set_mode("query")
            for bidx, data in enumerate(tqdm(data_loader, desc="Query")):
                """
                data = {
                    'c_img': c_img,
                    'c_cap': data['c_cap'],
                    'c_iid': data['c_iid'],
                    't_iid': data['t_iid'],
                    't_cap': data['t_cap'],
                    'mod_key': data['mod_key'],
                    'mod_str': mod_str,
                }
                """
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = data[k].to(f"cuda:{self.cfg.GPU_ID}")

                with torch.no_grad():
                    _feat = self.model.get_manipulated_image_feature(data)

                for i in range(_feat.size(0)):
                    _gt = data["t_cap"][i].split(",")
                    for __gt in _gt:
                        gt.append(__gt)
                        _q_feat = _feat[i].squeeze()
                        q_feats.append(_q_feat.cpu().numpy())
                        q_ids.append(data["c_iid"][i])

            # compute index features
            data_loader.dataset.set_mode("index")
            for bidx, data in enumerate(tqdm(data_loader, desc="Index")):
                """
                data = {
                    'img': img,
                    'cap': cap,
                    'iid': iid,
                }
                """
                for k, v in data.items():
                    if isinstance(v, torch.Tensor):
                        data[k] = data[k].to(f"cuda:{self.cfg.GPU_ID}")

                with torch.no_grad():
                    _feat = self.model.get_original_image_feature(data)

                for i in range(_feat.size(0)):
                    _i_feat = _feat[i].squeeze()
                    i_feats.append(_i_feat.cpu().numpy())
                    i_caps.append(data["cap"][i])
                    i_ids.append(data["iid"][i])

            # compute cosine similarity
            q_feats = np.stack(q_feats, axis=0)
            i_feats = np.stack(i_feats, axis=0)
            sims = np.dot(q_feats, i_feats.T)

            self.logger.print(f"Calculate similarity score: {sims.shape}")

            # remove query image from nn result
            for i in range(sims.shape[0]):
                i_idx = i_ids.index(q_ids[i])
                sims[i][i_idx] = -10e10
            nn_result = [np.argsort(-sims[i, :])[:110] for i in range(sims.shape[0])]
            nn_result = [[i_caps[nn] for nn in nns] for nns in nn_result]

            # compute recalls
            score = {}
            kappa = [1, 5, 10, 50]
            for k in kappa:
                r = 0.0
                for i, nns in enumerate(nn_result):
                    if gt[i] in nns[:k]:
                        r += 1
                r /= float(len(nn_result))
                score[str(k)] = r

            _r1 = score[str(1)]
            _r5 = score[str(5)]
            _r10 = score[str(10)]
            _r50 = score[str(50)]
            _r10r50 = 0.5 * (_r10 + _r50)
            r10 += _r10
            r50 += _r50
            r10r50 += _r10r50
            if self.logger is not None:
                self.logger.add_scalar(f"test/{target}/R1", _r1, epoch)
                self.logger.add_scalar(f"test/{target}/R5", _r5, epoch)
                self.logger.add_scalar(f"test/{target}/R10", _r10, epoch)
                self.logger.add_scalar(f"test/{target}/R50", _r50, epoch)
                self.logger.add_scalar(f"test/{target}/R10R50", _r10r50, epoch)
                self.logger.print(
                    f"{target}>> R10:{_r10:.4f}\tR50:{_r50:.4f}\tR10R50:{_r10r50:.4f}"
                )

            self.score_log[f"{target}_r10"] = _r10
            self.score_log[f"{target}_r50"] = _r50
            self.score_log[f"{target}_r10R50"] = _r10r50

        # mean score.
        r10r50 /= len(self.data_loader)
        r10 /= len(self.data_loader)
        r50 /= len(self.data_loader)
        self.logger.print(
            f"Overall>> R10:{r10:.4f}\tR50:{r50:.4f}\tR10R50:{r10r50:.4f}"
        )

        # logging
        if self.logger is not None:
            self.logger.add_scalar("test/overall/R10", r10, epoch)
            self.logger.add_scalar("test/overall/R50", r50, epoch)
            self.logger.add_scalar("test/overall/R10R50", r10r50, epoch)

        self.score_log["overall_r10"] = r10
        self.score_log["overall_r50"] = r50
        self.score_log["overall_r10r50"] = r10r50
        cur_score = r10r50

        # save checkpoint
        if cur_score > self.best_score:
            self.best_score = cur_score
            self.cfg.BEST_SCORE = self.best_score
            self.cfg.BEST_EPOCH = self.epoch
            self.cfg.SCORE_LOG = self.score_log
            with open(
                os.path.join(self.ckpt_path, "config.json"), "w", encoding="utf-8"
            ) as fopen:
                json.dump(self.cfg, fopen, indent=4, ensure_ascii=False)
                fopen.flush()
            state = {
                "score": self.best_score,
                "best_epoch": self.epoch,
                "best_epoch_score_log": self.score_log,
            }
            self.model.save(
                os.path.join(self.ckpt_path, "best_model.pth"),
                state,
            )
