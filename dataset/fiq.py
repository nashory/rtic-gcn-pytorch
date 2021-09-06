from __future__ import absolute_import, division, print_function, unicode_literals

import json
import logging
import os

import numpy as np
from tqdm import tqdm

from dataset.base import BaseDataset


class FashionIQDataset(BaseDataset):
    def __init__(self, cfg, mode, is_train, is_graph_infer, **dataset_specific_args):
        super(FashionIQDataset, self).__init__(cfg, mode, is_train, is_graph_infer)
        self.split = "train" if self.is_train else "val"
        self.target_names = ["dress", "toptee", "shirt"]

        self.target_name = dataset_specific_args.get("target_name", None)
        if self.target_name is not None:
            self.target_names = [self.target_name]
        self.__init_data__()
        self.__set_transform__()

    def __len__(self):
        if self.mode == "index":
            return len(self.index_dataset)
        else:
            return len(self.dataset)

    def __init_data__(self):
        if self.target_name is not None:
            # index
            split_file = f"image_splits/split.{self.target_name}.{self.split}.json"
            logging.info(f"[Dataset] load split file: {split_file}")
            with open(os.path.join(self.data_root, split_file), "r") as f:
                self.index_dataset = json.load(f)
            self.index_dataset = np.asarray(self.index_dataset)

        # train & query
        self.dataset = []
        self.all_texts = []
        for target_name in self.target_names:
            cap_file = f"captions/cap.{target_name}.{self.split}.json"
            logging.info(f"[Dataset] load annotation file: {cap_file}")
            full_cap_path = os.path.join(self.data_root, cap_file)
            assert os.path.exists(full_cap_path), f"{full_cap_path} does not exist"
            with open(full_cap_path, "r") as f:
                data = json.load(f)
                for i, d in enumerate(tqdm(data)):
                    c_iid = d["candidate"]
                    t_iid = d["target"]
                    self.all_texts.extend(d["captions"])
                    mod_str = [x.strip() for x in d["captions"]]
                    mod_key = f"{self.split}_{target_name}_{c_iid}_{i}"
                    _data = {
                        "c_img_path": os.path.join(
                            self.data_root, f"images/{c_iid}.jpg"
                        ),
                        "c_iid": c_iid,
                        "t_img_path": os.path.join(
                            self.data_root, f"images/{t_iid}.jpg"
                        ),
                        "t_iid": t_iid,
                        "mod_key": mod_key,
                        "mod_str": mod_str,
                    }
                    self.dataset.append(_data)

        self.all_texts = list(set(self.all_texts))
        self.dataset = np.asarray(self.dataset)

    def __sample_train__(self, index):
        data = self.dataset[index]
        c_img = self.__load_pil_image__(data["c_img_path"])
        t_img = self.__load_pil_image__(data["t_img_path"])

        if self.transform is not None:
            c_img = self.transform(c_img)
            t_img = self.transform(t_img)

        mod_str = data["mod_str"]
        np.random.shuffle(mod_str)
        mod_str = " [SEP] ".join(mod_str)

        ret = {
            "c_img": c_img,
            "c_cap": data["c_iid"],
            "c_iid": data["c_iid"],
            "t_img": t_img,
            "t_cap": data["t_iid"],
            "t_iid": data["t_iid"],
            "mod_key": data["mod_key"],
            "mod_str": mod_str,
        }
        return ret

    def __sample_query__(self, index):
        data = self.dataset[index]
        c_img = self.__load_pil_image__(data["c_img_path"])

        if self.transform is not None:
            c_img = self.transform(c_img)

        mod_str = data["mod_str"]
        mod_str = " [SEP] ".join(mod_str)

        ret = {
            "c_img": c_img,
            "c_cap": data["c_iid"],
            "c_iid": data["c_iid"],
            "t_cap": data["t_iid"],
            "t_iid": data["t_iid"],
            "mod_key": data["mod_key"],
            "mod_str": mod_str,
        }
        return ret

    def __sample_index__(self, index):
        iid = self.index_dataset[index]
        img = self.__load_pil_image__(os.path.join(self.data_root, f"images/{iid}.jpg"))

        if self.transform is not None:
            img = self.transform(img)

        ret = {
            "img": img,
            "cap": iid,
            "iid": iid,
        }
        return ret
