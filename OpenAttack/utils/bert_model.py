import os
import pickle
from typing import List, Tuple

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import BertTokenizer, BertForSequenceClassification

from ..classifier import Classifier  # Assuming your project keeps the same relative import structure


class BertModelPaddle:
    """Lightweight wrapper around PaddleNLP BERT for prob / grad extraction."""

    def __init__(self, model_path: str, num_labels: int, max_len: int = 100, device: str = "cpu"):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model: BertForSequenceClassification = BertForSequenceClassification.from_pretrained(
            model_path,
            num_classes=num_labels,
        )
        self.max_len = max_len
        self.device = device
        paddle.device.set_device("gpu" if device.startswith("cuda") and paddle.is_compiled_with_cuda() else "cpu")
        self.model.eval()

        # Register forward hook on word embeddings to capture activations + grads
        self.curr_embedding = None

        def _hook(layer, inp, out):
            # out: Tensor (bsz, seq_len, hidden)
            self.curr_embedding = out
            out.stop_gradient = False  # enable grad

        self.model.bert.embeddings.word_embeddings.register_forward_post_hook(_hook)

        # load static vocab / embedding for non‑tokenized branch
        self.word2id = pickle.load(open(os.path.join(model_path, "bert_word2id.pkl"), "rb"))
        self.embedding = np.load(os.path.join(model_path, "bert_wordvec.npy"))

    # ---------------------------------------------------------------------
    # Helper tokenization --------------------------------------------------
    # ---------------------------------------------------------------------
    def _tokenize_batch(self, corpus: List[str]):
        input_ids, attention_masks, sent_lens = [], [], []
        for sent in corpus:
            enc = self.tokenizer(text=sent,
                                 max_seq_len=self.max_len,
                                 pad_to_max_seq_len=True,
                                 return_attention_mask=True,
                                 truncation=True)
            sent_lens.append(int(sum(enc["attention_mask"]) - 2))
            input_ids.append(enc["input_ids"])
            attention_masks.append(enc["attention_mask"])
        return np.array(input_ids, dtype="int64"), np.array(attention_masks, dtype="int64"), sent_lens

    # ---------------------------------------------------------------------
    # Core predict ---------------------------------------------------------
    # ---------------------------------------------------------------------
    def predict(self,
                sentences: List,
                labels: List[int] = None,
                tokenize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """Return (probs, grads) with grads shape = (bsz, max_seq_len, hidden).
        When `tokenize=False`, `sentences` is expected to be a list of token‑id sequences.
        """
        if tokenize:
            ids, masks, sent_lens = self._tokenize_batch(sentences)
        else:
            # sentences is already list of tokens
            sentences = [s[: self.max_len - 2] for s in sentences]
            sent_lens = [len(s) for s in sentences]
            masks = np.array([[1] * (len(s) + 2) + [0] * (self.max_len - 2 - len(s)) for s in sentences], dtype="int64")
            ids = [[self.word2id.get("[CLS]")]]
            ids = [
                [self.word2id.get("[CLS]")] + [self.word2id.get(tok, self.word2id.get("[UNK]")) for tok in s]
                + [self.word2id.get("[SEP]")] + [self.word2id.get("[PAD]")] * (self.max_len - 2 - len(s))
                for s in sentences
            ]
            ids = np.array(ids, dtype="int64")
            masks = masks

        if labels is None:
            labels = [0] * len(sentences)
        labels_tensor = paddle.to_tensor(labels, dtype="int64")
        probs_list, grad_list = [], []

        for i in range(len(ids)):
            xs = paddle.to_tensor(ids[i:i + 1])
            attn = paddle.to_tensor(masks[i:i + 1])
            logits = self.model(xs, attention_mask=attn)
            loss_fn = paddle.nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels_tensor[i:i + 1])
            loss = -loss  # maximize prob of the true class as in original code
            loss.backward()

            probs = F.softmax(logits, axis=-1).numpy()[0]
            grad_list.append(self.curr_embedding.grad.numpy()[0])
            probs_list.append(probs)
            self.model.clear_gradients()

        max_valid_len = max(sent_lens)
        probs_arr = np.array(probs_list)
        # slice grad: skip [CLS], limit to max sentence length
        grad_arr = np.stack(grad_list)[:, 1:1 + max_valid_len, :]
        return probs_arr, grad_arr


class BertClassifierPaddle(Classifier):
    def __init__(self, model_path: str, num_labels: int, max_len: int = 100, device: str = "cpu"):
        self._model = BertModelPaddle(model_path, num_labels, max_len, device)
        self.word2id = self._model.word2id
        self.embedding = self._model.embedding

    def to(self, device: str):
        self._model.to(device)
        return self

    def get_prob(self, inputs: List[str]):
        return self._model.predict(inputs)[0]

    def get_grad(self, token_id_sequences: List[List[int]], labels: List[int]):
        # expects pre‑tokenized sequence of ids (without CLS / SEP) similar to original code
        return self._model.predict(token_id_sequences, labels, tokenize=False)[1]
