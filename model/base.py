"""Class for text data."""
import logging
import math
import pickle
import string
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.models as M
from torch.autograd import Variable
from tqdm import tqdm
from transformers import BertModel, BertTokenizer, GPT2Model, GPT2Tokenizer, GPTNeoModel

import model.resnet as resnet
from misc.common import Flatten, NormalizationLayer
from misc.loss import BatchBasedXentLoss, BatchHardTripletLoss
from misc.spellchecker import SpellChecker

__VERBOSE__ = False


# helper func.
def tokenize_text(text):
    # python3
    text = text.encode("ascii", "ignore").decode("ascii")
    table = str.maketrans(dict.fromkeys(string.punctuation))
    tokens = str(text).lower().translate(table).strip().split()
    return tokens


def apply_spell_correction(text):
    tokens = tokenize_text(text)
    for i, token in enumerate(tokens):
        tokens[i] = SpellChecker.correct_token(token)
    text = " ".join(tokens)
    return text


class SimpleVocab(object):
    def __init__(self):
        super(SimpleVocab, self).__init__()
        self.word2id = {}
        self.wordcount = {}
        self.add_special_token("[UNK]")
        self.add_special_token("[CLS]")
        self.add_special_token("[SEP]")

    def add_special_token(self, token):
        self.word2id[token] = len(self.word2id)
        self.wordcount[token] = 9e9

    def add_text_to_vocab(self, text):
        tokens = tokenize_text(text)
        if __VERBOSE__:
            logging.info(f"[Tokenizer] Text: {text} / Tokens: {tokens}")
        for token in tokens:
            if not token in self.word2id:
                self.word2id[token] = len(self.word2id)
                self.wordcount[token] = 0
            self.wordcount[token] += 1

    def get_size(self):
        return len(self.word2id)


class Word2Vec(nn.Module):
    def __init__(self, texts_to_build_vocab, word_embedding_init, **kwargs):
        super(Word2Vec, self).__init__()

        self.word_embedding_init = word_embedding_init

        if word_embedding_init == "bert":  # 1024 dim
            self.vocab = BertTokenizer.from_pretrained("bert-large-uncased")
            self.embedding = BertModel.from_pretrained("bert-large-uncased").embeddings
        elif word_embedding_init == "gpt2-xl":  # 1600 dim
            self.vocab = GPT2Tokenizer.from_pretrained("gpt2-xl")
            self.embedding = GPT2Model.from_pretrained("gpt2-xl").wte
        elif word_embedding_init == "gpt-neo":  # 2048 dim
            self.vocab = GPT2Tokenizer.from_pretrained("EleutherAI/gpt-neo-1.3B")
            self.embedding = GPTNeoModel.from_pretrained("EleutherAI/gpt-neo-1.3B").wte
        elif word_embedding_init == "glove":  # 1100 dim
            self.vocab = SimpleVocab()
            logging.info(
                f"Build vocabulary from scratch (w/ GloVe): {len(texts_to_build_vocab)}"
            )
            for text in tqdm(texts_to_build_vocab):
                self.vocab.add_text_to_vocab(text)
            vocab_size = self.vocab.get_size()
            embed_dim = 1100
            self.embedding = nn.Embedding(vocab_size, embed_dim)

            # GloVe Initiallization
            with open("glove/fiq.glove.1100d.pkl", "rb") as fopen:
                glove = pickle.load(fopen)
            word2id = self.vocab.word2id
            id2word = [0] * len(word2id)
            for k, v in word2id.items():
                id2word[v] = k
            weights = []
            for idx, word in enumerate(tqdm(id2word)):
                if word in glove:
                    w = torch.from_numpy(glove[word]).float()
                    assert w.shape[0] == 1100
                    weights.append(w)
                else:
                    if __VERBOSE__:
                        logging.info(f'[Warn] token "{word}" does not exist in GloVe.')
                    weights.append(self.embedding.weight[idx])
            weights = torch.stack(weights)
            self.embedding.weight = nn.Parameter(weights)
        else:
            self.vocab = SimpleVocab()
            logging.info(f"Build vocabulary from scratch: {len(texts_to_build_vocab)}")
            for text in tqdm(texts_to_build_vocab):
                self.vocab.add_text_to_vocab(text)
            vocab_size = self.vocab.get_size()
            embed_dim = kwargs["embed_dim"]
            self.embedding = nn.Embedding(vocab_size, embed_dim)

    def encode_text(self, text):
        if self.word_embedding_init in {"bert", "gpt2-xl", "gpt-neo"}:
            return self.vocab.encode(text)
        else:
            tokens = tokenize_text(text)
            x = [0]
            if len(tokens) > 0:
                x = [self.vocab.word2id.get(token, 0) for token in tokens]
            return x

    def forward(self, x):
        return self.embedding(x)


class TextLSTMModel(nn.Module):
    def __init__(
        self,
        texts_to_build_vocab,
        word_embed_dim,
        lstm_hidden_dim,
        num_layers,
        spell_correction=False,
        word_embedding_init="bert",
    ):
        super(TextLSTMModel, self).__init__()

        self.lstm_hidden_dim = lstm_hidden_dim

        if spell_correction:
            texts_to_build_vocab = [
                apply_spell_correction(text) for text in texts_to_build_vocab
            ]

        self.word2vec = Word2Vec(
            texts_to_build_vocab,
            word_embedding_init,
            embed_dim=word_embed_dim,
        )

        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(
            word_embed_dim,
            lstm_hidden_dim,
            num_layers=self.num_layers,
            dropout=0.1,
            bidirectional=False,
        )

        self.fc_output = nn.Linear(lstm_hidden_dim, lstm_hidden_dim)

    def forward(self, x):
        """input x: list of strings"""
        if isinstance(x, list) or isinstance(x, tuple):
            if isinstance(x[0], str) or isinstance(x[0], unicode):
                x = [self.word2vec.encode_text(text) for text in x]
        assert isinstance(x, list) or isinstance(x, tuple)
        assert isinstance(x[0], list) or isinstance(x[0], tuple)
        assert isinstance(x[0][0], int)
        return self.forward_encoded_texts(x)

    def forward_encoded_texts(self, texts):
        # to tensor
        lengths = [len(t) for t in texts]
        itexts = torch.zeros((np.max(lengths), len(texts))).long()
        for i in range(len(texts)):
            itexts[: lengths[i], i] = torch.LongTensor(texts[i])

        # embed words
        itexts = itexts.to("cuda")
        etexts = self.word2vec(itexts)

        # lstm
        lstm_output, _ = self.forward_lstm_(etexts)

        # use maxpool
        lstm_output, _ = torch.max(lstm_output, dim=0)

        # output
        return self.fc_output(lstm_output)

    def forward_lstm_(self, etexts):
        """https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM"""
        batch_size = etexts.shape[1]
        first_hidden = (
            torch.zeros(self.num_layers, batch_size, self.lstm_hidden_dim).to("cuda"),
            torch.zeros(self.num_layers, batch_size, self.lstm_hidden_dim).to("cuda"),
        )
        lstm_output, last_hidden = self.lstm(etexts, first_hidden)
        return lstm_output, last_hidden


class TextLSTMGRUModel(nn.Module):
    def __init__(
        self,
        texts_to_build_vocab,
        word_embed_dim,
        hidden_dim,
        num_layers,
        spell_correction=False,
        word_embedding_init="bert",
        with_fc=True,
    ):
        super(TextLSTMGRUModel, self).__init__()

        self.with_fc = with_fc
        self.word_embed_dim = word_embed_dim
        self.hidden_dim = hidden_dim

        if spell_correction:
            texts_to_build_vocab = [
                apply_spell_correction(text) for text in texts_to_build_vocab
            ]

        self.word2vec = Word2Vec(
            texts_to_build_vocab,
            word_embedding_init,
            embed_dim=word_embed_dim,
        )

        # 2-layer LSTM.
        self.num_layers = num_layers
        self.lstm = torch.nn.LSTM(
            word_embed_dim,
            hidden_dim,
            num_layers=self.num_layers,
            dropout=0.1,
            bidirectional=False,
        )

        # 2-layer GRU
        self.num_layers = num_layers
        self.gru = torch.nn.GRU(
            word_embed_dim,
            hidden_dim,
            num_layers=self.num_layers,
            dropout=0.1,
            bidirectional=False,
        )

        if with_fc:
            # squeeze [ lstm+gru(2) * hidden_dim ] -->[ hidden_dim ]
            self.fc_output = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, x):
        """input x: list of strings"""
        if isinstance(x, list) or isinstance(x, tuple):
            if isinstance(x[0], str) or isinstance(x[0], unicode):
                x = [self.word2vec.encode_text(text) for text in x]
        assert isinstance(x, list) or isinstance(x, tuple)
        assert isinstance(x[0], list) or isinstance(x[0], tuple)
        assert isinstance(x[0][0], int)
        x = self.forward_encoded_texts(x)
        if self.with_fc:
            x = self.fc_output(x)
        return x

    def forward_encoded_texts(self, texts):
        # to tensor
        lengths = [len(t) for t in texts]
        itexts = torch.zeros((np.max(lengths), len(texts))).long()
        for i in range(len(texts)):
            itexts[: lengths[i], i] = torch.LongTensor(texts[i])

        # embed words
        itexts = itexts.to("cuda")
        etexts = self.word2vec(itexts)

        # lstm
        lstm_output, _ = self.forward_lstm_(etexts)
        gru_output, _ = self.forward_gru_(etexts)

        # use maxpool
        lstm_output, _ = torch.max(lstm_output, dim=0)
        gru_output, _ = torch.max(gru_output, dim=0)
        text_features = torch.cat([lstm_output, gru_output], dim=1)
        return text_features

    def forward_lstm_(self, etexts):
        """https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM"""
        lstm_output, last_hidden = self.lstm(etexts)
        return lstm_output, last_hidden

    def forward_gru_(self, etexts):
        """https://pytorch.org/docs/stable/nn.html#torch.nn.GRU"""
        gru_output, last_hidden = self.gru(etexts)
        return gru_output, last_hidden


class ImageEncoderTextEncoderBase(nn.Module):
    """Base class for image and text encoder."""

    def __init__(self, cfg, texts):
        super(ImageEncoderTextEncoderBase, self).__init__()

        self.model = dict()
        self.cfg = cfg
        self.texts = texts

        # set number of feature dims.
        self.out_feature_image = cfg.TRAIN.MODEL.out_feature_image
        self.in_feature_text = cfg.TRAIN.MODEL.in_feature_text
        self.out_feature_text = cfg.TRAIN.MODEL.out_feature_text

        if cfg.TRAIN.LOSS == "batch_based_xent":
            self.model["criterion"] = BatchBasedXentLoss()
        elif cfg.TRAIN.LOSS == "batch_hard_triplet":
            self.model["criterion"] = BatchHardTripletLoss()
        else:
            raise NotImplementedError

        # load pretrained weights.
        pretrained = cfg.TRAIN.MODEL.image_model.pretrained
        backbone = cfg.TRAIN.MODEL.image_model.name
        logging.info(f"Backbone: {backbone} is loaded with pretrained={pretrained}")
        if backbone in {
            "resnet50",
            "resnet101",
            "resnet152",
            "resnext50_32x4d",
            "resnext101_32x8d",
            "wide_resnet50_2",
            "wide_resnet101_2",
        }:
            self.model["v_enc"] = resnet.__dict__[backbone](pretrained=pretrained)
            in_c_last = 2048
            in_c_intm = 1024
        else:
            raise NotImplementedError

        self.model["v_emb"] = nn.Conv2d(in_c_last, self.out_feature_image, 1)

        if cfg.TRAIN.MODEL.text_model.name == "lstm":
            self.model["t_enc"] = TextLSTMModel(
                texts_to_build_vocab=texts,
                word_embed_dim=self.in_feature_text,
                lstm_hidden_dim=self.out_feature_text,
                num_layers=cfg.TRAIN.MODEL.text_model.params.lstm.num_layers,
                spell_correction=cfg.TRAIN.MODEL.spell_correction,
                word_embedding_init=cfg.TRAIN.MODEL.word_embedding_init,
            )
        elif cfg.TRAIN.MODEL.text_model.name == "lstm_gru":
            self.model["t_enc"] = TextLSTMGRUModel(
                texts_to_build_vocab=texts,
                word_embed_dim=self.in_feature_text,
                hidden_dim=self.out_feature_text,
                num_layers=cfg.TRAIN.MODEL.text_model.params.lstm_gru.num_layers,
                spell_correction=cfg.TRAIN.MODEL.spell_correction,
                word_embedding_init=cfg.TRAIN.MODEL.word_embedding_init,
            )

        self.model["norm"] = NormalizationLayer(
            learn_scale=True, normalize_scale=cfg.TRAIN.MODEL.normalize_scale
        )

    def get_config_optim(self, lr, fixed_lrp=None):
        params = []
        for k, v in self.model.items():
            if fixed_lrp is not None:  # used for joint graining w/ GCN.
                assert isinstance(fixed_lrp, float)
                if k == "v_enc":
                    params.append(
                        {
                            "params": v.parameters(),
                            "lr": lr,
                            "lrp": float(self.cfg.TRAIN.OPTIMIZER.lrp) * fixed_lrp,
                        }
                    )
                else:
                    params.append(
                        {"params": v.parameters(), "lr": lr, "lrp": fixed_lrp}
                    )
            else:
                if k == "v_enc":
                    params.append(
                        {
                            "params": v.parameters(),
                            "lr": lr,
                            "lrp": float(self.cfg.TRAIN.OPTIMIZER.lrp),
                        }
                    )
                else:
                    params.append({"params": v.parameters(), "lr": lr, "lrp": 1.0})
        return params

    def save(self, path, state={}):
        state["state_dict"] = dict()
        for k, v in self.model.items():
            state["state_dict"][k] = v.state_dict()
        state["texts"] = self.texts
        torch.save(state, path)

    def load(self, path):
        state_dict = torch.load(path, map_location=torch.device("cpu"))["state_dict"]
        for k, v in state_dict.items():
            if k in self.model:
                self.model[k].load_state_dict(v)
            else:
                logging.info(
                    f"[!!! WARN !!!] {k} module is not copied. Is it intended?"
                )

    def extract_image_feature(self, x, pool=True):
        x = self.model["v_enc"](x)
        x = self.model["v_emb"](x)
        if pool:
            x = F.adaptive_avg_pool2d(x, (1, 1))
            x = torch.flatten(x, 1)
        return x

    def extract_text_feature(self, texts):
        x = self.model["t_enc"](texts)
        return x

    def compose_img_text(self, f_img, f_text):
        raise NotImplementedError

    def get_original_image_feature(self, x):
        """
        data = {
            'img': img,
            'iid': iid,
        }
        """

        x = self.extract_image_feature(x["img"])
        return self.model["norm"](x)

    def get_manipulated_image_feature(self, x):
        """
        x = {
            'c_img': c_img,
            'c_iid': data['c_iid'],
            't_iid': data['t_iid'],
            'mod_key': data['mod_key'],
            'mod_str': mod_str,
        }
        """
        f_img = self.extract_image_feature(x["c_img"])
        f_text = self.extract_text_feature(x["mod_str"])
        x = self.compose_img_text(f_img, f_text)
        return self.model["norm"](x)

    def update(self, x, opt):
        """
        input = (f_img_c, f_img_t, f_cit_t, f_text)
        """
        # assign input
        f_img_t = self.model["norm"](x["f_img_t"])  # target
        f_cit_t = self.model["norm"](x["f_cit_t"])  # manipulated

        # loss
        loss = self.model["criterion"](f_img_t, f_cit_t)

        # backward
        opt.zero_grad()
        loss.backward()
        opt.step()

        # return log
        log_data = dict()
        log_data["loss"] = float(loss.data)
        return log_data

    def forward(self, x):
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
        f_img_c = self.extract_image_feature(x["c_img"])
        f_img_t = self.extract_image_feature(x["t_img"])
        f_text = self.extract_text_feature(x["mod_str"])
        f_cit_t = self.compose_img_text(f_img_c, f_text)

        output = dict(
            f_img_c=f_img_c,
            f_img_t=f_img_t,
            f_text=f_text,
            f_cit_t=f_cit_t,
        )

        return output
