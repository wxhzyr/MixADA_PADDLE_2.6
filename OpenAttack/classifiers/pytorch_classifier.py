import numpy as np
import paddle
import paddle.nn.functional as F
from . import ClassifierBase  # keep original import path
from ..text_processors import DefaultTextProcessor
from ..utils import check_parameters
from ..exceptions import ClassifierNotSupportException


DEFAULT_CONFIG = {
    "device": None,
    "processor": DefaultTextProcessor(),
    "embedding": None,
    "word2id": None,
    "max_len": None,
    "tokenization": False,
    "padding": False,
    "token_unk": "<UNK>",
    "token_pad": "<PAD>",
    "require_length": False,
}


class PytorchClassifier(ClassifierBase):
    """Universal PaddlePaddle text classifier wrapper with grad support.

    Parameters mirror the original ``PytorchClassifier`` but use Paddle APIs.
    """

    def __init__(self, model: paddle.nn.Layer, **kwargs):
        # Prepare config ---------------------------------------------------------------------
        self.model = model
        self.config = DEFAULT_CONFIG.copy()
        # default device: gpu if available else cpu
        self.config["device"] = "gpu" if paddle.is_compiled_with_cuda() else "cpu"
        self.config.update(kwargs)
        check_parameters(DEFAULT_CONFIG.keys(), self.config)
        paddle.device.set_device(self.config["device"])
        self.model.eval()

        # Parent class handles tokenization / padding etc.
        super().__init__(**self.config)

    # ------------------------------------------------------------------
    def to(self, device: str):
        self.config["device"] = device
        paddle.device.set_device(device)
        return self

    # ------------------------------------------------------------------
    def _prepare_input(self, arr_or_tensor):
        """Convert numpy to Paddle tensor on correct device."""
        if isinstance(arr_or_tensor, np.ndarray):
            return paddle.to_tensor(arr_or_tensor, place=paddle.CUDAPlace(0) if self.config["device"] == "gpu" else None)
        return arr_or_tensor

    # ------------------------------------------------------------------
    def get_pred(self, input_):
        data, seq_len = self.preprocess(input_)
        x = self._prepare_input(data)
        if self.config["tokenization"] and self.config["require_length"]:
            logits = self.model(x, seq_len)
        else:
            logits = self.model(x)
        return logits.argmax(axis=1).cpu().numpy()

    # ------------------------------------------------------------------
    def get_prob(self, input_):
        data, seq_len = self.preprocess(input_)
        x = self._prepare_input(data)
        if self.config["tokenization"] and self.config["require_length"]:
            logits = self.model(x, seq_len)
        else:
            logits = self.model(x)
        return F.softmax(logits, axis=1).cpu().numpy()

    # ------------------------------------------------------------------
    def get_grad(self, input_, labels):
        if self.config["word2id"] is None or self.config["embedding"] is None:
            raise ClassifierNotSupportException("gradient")

        data, seq_len = self.preprocess_token(input_)
        x = paddle.to_tensor(data, stop_gradient=False)
        label_tensor = paddle.to_tensor(labels, dtype="int64")

        if self.config["require_length"]:
            logits = self.model(x, seq_len)
        else:
            logits = self.model(x)
        loss = logits[paddle.arange(len(labels)), label_tensor].sum()
        (-loss).backward()  # maximize probability of true class to align with original
        return logits.cpu().detach().numpy(), x.grad.cpu().numpy()
