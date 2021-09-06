import os
import sys

sys.path.append("../")
import json
import pickle
import string

import numpy as np
from tqdm import tqdm

from misc.spellchecker import SpellChecker


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


targets = ["toptee", "dress", "shirt"]
splits = ["train", "test", "val"]
texts = []

for target in targets:
    for split in splits:
        with open(
            f"/data/public/rw/datasets/fashion-iq/captions/cap.{target}.{split}.json",
            "r",
        ) as fopen:
            data = json.load(fopen)
            for d in tqdm(data):
                texts.extend(d["captions"])

for i in tqdm(range(len(texts))):
    texts[i] = apply_spell_correction(texts[i])


tokens = []
for i in tqdm(range(len(texts))):
    tokens.extend(tokenize_text(texts[i]))
tokens = set(tokens)

glove = {}
glove_names = [
    "glove.42B.300d.txt",
    "glove.twitter.27B.200d.txt",
    "glove.6B.300d.txt",
    "glove.840B.300d.txt",
]

for fname in glove_names:
    glove[fname] = {}
    with open(fname, "r") as fopen:
        lines = fopen.readlines()
        for line in tqdm(lines):
            p = line.strip().split()
            word = p[0]
            if not word in tokens:
                continue
            try:
                embedding = np.asarray(
                    list(map(lambda x: float(x), p[1:])), dtype=np.float32
                )
                glove[fname][word] = embedding
            except:
                pass

final_glove = {}
for token in tqdm(tokens):
    if all(token in glove[name] for name in glove_names):
        x = []
        for name in glove_names:
            x.append(glove[name][token])
        final_glove[token] = np.concatenate(x)

with open("fiq.glove.1100d.pkl", "wb") as fopen:
    pickle.dump(final_glove, fopen)
