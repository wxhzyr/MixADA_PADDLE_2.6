# coding: utf-8
"""PaddlePaddle version of adversarial attack evaluation using OpenAttack.

- Supports MixText / RobertaMixText Paddle models (see `mixtext_paddle.py`).
- Uses PaddleNLP tokenizers; no PyTorch dependency.
- Keeps original CLI flags as much as possible.
"""
import argparse
import logging
import os
import random
from typing import List

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import (
    BertTokenizer, RobertaTokenizer,
    BertConfig, RobertaConfig,
)

import OpenAttack
from OpenAttack.utils.dataset import Dataset, DataInstance
from OpenAttack.attackers import *  # noqa: F401,F403

from mixtext_paddle import MixText, RobertaMixText  # local paddle models

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)

# ---------------------------------------------------------------------------
LABEL_LIST = ["0", "1"]
LABEL_MAP = {l: i for i, l in enumerate(LABEL_LIST)}
NUM_LABELS = len(LABEL_LIST)

ATTACKER_MAP = {
    "pwws": PWWSAttacker,
    "generic": GeneticAttacker,
    "hotflip": HotFlipAttacker,
    "pso": PSOAttacker,
    "textfooler": TextFoolerAttacker,
    "uat": UATAttacker,
    "viper": VIPERAttacker,
}

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def tokenize_texts(tokenizer, texts: List[str], max_len: int):
    """Batch‑tokenize a list of sentences and return numpy arrays."""
    encoded = tokenizer(texts,
                        max_length=max_len,
                        truncation=True,
                        
                        pad_to_max_seq_len=True,
                        return_attention_mask=True,
                        return_token_type_ids=True)
    return np.array(encoded["input_ids"], dtype="int64"), np.array(encoded["token_type_ids"], dtype="int64"), np.array(encoded["attention_mask"], dtype="int64")


# ---------------------------------------------------------------------------
# Classifiers
# ---------------------------------------------------------------------------
class PaddleClassifier(OpenAttack.Classifier):
    """Generic Paddle model wrapper compatible with OpenAttack."""

    def __init__(self, args):
        self.args = args
        if args.model_type.lower() == "roberta":
            tokenizer_cls, config_cls, model_cls = RobertaTokenizer, RobertaConfig, RobertaMixText
        else:
            tokenizer_cls, config_cls, model_cls = BertTokenizer, BertConfig, MixText

        self.tokenizer = tokenizer_cls.from_pretrained(args.model_name_or_path)
        config = config_cls.from_pretrained(args.model_name_or_path)
        config.num_labels = NUM_LABELS
        self.model = model_cls.from_pretrained(args.model_name_or_path, config=config)
        device = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        paddle.device.set_device(device)
        self.model.eval()
        logger.info("Loaded Paddle model (%s) on %s", args.model_type, device)

    # ---- OpenAttack interface ---------------------------------------------
    def get_prob(self, input_: List[str]):
        ids, token_type_ids, attn = tokenize_texts(self.tokenizer, input_, self.args.max_seq_length)
        batch_size = 64  # small batch to avoid OOM
        probs = []
        for i in range(0, len(ids), batch_size):
            batch_ids = paddle.to_tensor(ids[i:i + batch_size])
            batch_attn = paddle.to_tensor(attn[i:i + batch_size])
            logits, _ = self.model(batch_ids, batch_attn)
            probs.append(F.softmax(logits, axis=1).numpy())
        return np.vstack(probs)

    def get_pred(self, input_: List[str]):
        return np.argmax(self.get_prob(input_), axis=1)


class ModelClassifier(OpenAttack.Classifier):
    """Wrap an externally passed Paddle model & tokenizer (optional)."""

    def __init__(self, tokenizer, model, args):
        self.tokenizer = tokenizer
        self.model = model
        self.args = args
        self.model.eval()

    def get_prob(self, input_):
        ids, token_type_ids, attn = tokenize_texts(self.tokenizer, input_, self.args.max_seq_length)
        logits, _ = self.model(paddle.to_tensor(ids), paddle.to_tensor(attn))
        return F.softmax(logits, axis=1).numpy()

    def get_pred(self, input_):
        return np.argmax(self.get_prob(input_), axis=1)


# ---------------------------------------------------------------------------
# Dataset loader (SST‑2 formatted TSV)
# ---------------------------------------------------------------------------

def load_custom_dataset(path: str, sample_frac: float = 0.2):
    import csv
    samples = []
    with open(path, "r", encoding="utf‑8‑sig") as f:
        reader = csv.reader(f, delimiter="\t")
        for line in reader:
            if len(line) < 2:
                continue
            text, label = line[0].lower(), LABEL_MAP[line[1]]
            samples.append(DataInstance(x=text, y=label, target=1 - label, meta={}))
    if len(samples) > 10000:
        random.shuffle(samples)
        samples = samples[: int(len(samples) * sample_frac)]
    logger.info("Loaded %d examples from %s", len(samples), path)
    return Dataset(samples)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_name_or_path", required=True)
    p.add_argument("--model_type", choices=["bert", "roberta"], default="bert")
    p.add_argument("--data_dir", required=True, help="Path to test.tsv (SST‑2 style)")
    p.add_argument("--attacker", choices=list(ATTACKER_MAP.keys()), default="pwws")
    p.add_argument("--max_seq_length", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=128)  # not used but kept
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    classifier = PaddleClassifier(args)
    dataset = load_custom_dataset(args.data_dir)

    attacker_cls = ATTACKER_MAP[args.attacker]
    attacker = attacker_cls()

    attack_eval = OpenAttack.attack_evals.DefaultAttackEval(attacker, classifier, progress_bar=True)
    res = attack_eval.eval(dataset, visualize=False, save_dir=args.save_dir)
    print(res)


if __name__ == "__main__":
    main()
