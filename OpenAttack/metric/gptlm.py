import math
import os
import logging
from typing import Union

import paddle
import paddle.nn.functional as F
from paddlenlp.transformers import GPTTokenizer, GPTForPretraining


class GPT2LM:
    """Compute perplexity of a sentence using PaddlePaddle GPT‑2 (English).

    Parameters
    ----------
    use_gpu : bool, optional
        If ``True`` and CUDA is available, the model is placed on GPU.
    model_name : str, optional
        Name of the pretrained GPT‑2 weight in PaddleNLP hub. Default ``"gpt2-en"``.
    """

    def __init__(self, use_gpu: bool = True, model_name: str = "gpt2-en"):
        logging.getLogger("paddlenlp").setLevel(logging.ERROR)
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        self.tokenizer = GPTTokenizer.from_pretrained(model_name)
        self.model: GPTForPretraining = GPTForPretraining.from_pretrained(model_name)
        device = "gpu" if use_gpu and paddle.is_compiled_with_cuda() else "cpu"
        paddle.device.set_device(device)
        self.model.eval()

    # ------------------------------------------------------------------
    def _calc_loss(self, logits: paddle.Tensor, labels: paddle.Tensor) -> paddle.Tensor:
        """Cross‑entropy loss ignoring padding tokens."""
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        loss = F.cross_entropy(
            shift_logits.reshape([-1, shift_logits.shape[-1]]),
            shift_labels.flatten(),
            reduction="none",
        )
        # mask padding tokens (id 0 for GPT2 tokenizer)
        mask = shift_labels != self.tokenizer.pad_token_id
        loss = loss * mask.flatten().astype(loss.dtype)
        return loss.reshape(shift_labels.shape).sum(axis=1) / mask.sum(axis=1)

    # ------------------------------------------------------------------
    def __call__(self, sentence: Union[str, list[str]]) -> float:
        """Return perplexity of a single sentence or list of sentences."""
        single_input = isinstance(sentence, str)
        sents = [sentence] if single_input else sentence

        enc = self.tokenizer(sents, return_attention_mask=False, pad_to_max_seq_len=False, return_length=False)
        input_ids = paddle.to_tensor(enc["input_ids"], dtype="int64")
        with paddle.no_grad():
            logits = self.model(input_ids)[0]  # (bsz, seq_len, vocab)
            loss = self._calc_loss(logits, input_ids)
        ppl = paddle.exp(loss).numpy()
        return float(ppl[0]) if single_input else ppl.tolist()