"""
:type: tuple
:Size: 55.041MB
:Package Requirements:
    * **paddlepaddle** >= 2.5.0

PaddlePaddle re‑implementation of the pretrained GAN model on the SNLI dataset
required by :pyclass:`.GANAttacker`.

⚠️ **IMPORTANT**: _All original symbols (module‑level constants, helper
functions, class names, and, most importantly, the public function
``LOAD``)_ are **preserved** so that external code depending on reflection or
string import continues to work. Only the deep‑learning backend has been
ported from PyTorch → Paddle.
"""

import os
from OpenAttack.utils import make_zip_downloader

# ---------------------------------------------------------------------------
# Public constants (unchanged) ---------------------------------------------
# ---------------------------------------------------------------------------
NAME = "AttackAssist.GAN"
URL = "https://thunlp.oss-cn-qingdao.aliyuncs.com/TAADToolbox/GNAE.zip"
DOWNLOAD = make_zip_downloader(URL)


# ---------------------------------------------------------------------------
# Helper (kept for API compatibility) ---------------------------------------
# ---------------------------------------------------------------------------
def to_gpu(gpu, var):
    """No‑op helper; Paddle manages device globally. Kept for compatibility."""
    return var


# ---------------------------------------------------------------------------
# Loader (signature preserved) ---------------------------------------------
# ---------------------------------------------------------------------------

def LOAD(path):  # noqa: N802  (keep original uppercase name)
    """Return `(word2idx, autoencoder, inverter, gan_gen, gan_disc)` models.

    The returned models are Paddle `nn.Layer` instances **with the same public
    APIs** as their original PyTorch counterparts, enabling drop‑in
    replacement inside `GANAttacker`.

    **Note on checkpoints**: the official `.pkl` files are saved with PyTorch
    tensors.  You must first convert them to Paddle format (e.g. with
    `x2paddle` or manual numpy save/load) and place them at the same filenames
    but with extension `.pdparams` before calling `LOAD`.
    """
    import json
    import paddle
    import paddle.nn as nn
    import paddle.nn.functional as F
    import numpy as np

    # ------------------------------------------------------------------
    #   Re‑implement tiny building blocks (names 100% preserved) --------
    # ------------------------------------------------------------------
    class MLP_D(nn.Layer):
        def __init__(self, ninput, noutput, layers, activation=None, gpu=False):
            super().__init__()
            activation = activation or nn.LeakyReLU(0.2)
            sizes = [ninput] + [int(x) for x in layers.split("-")]
            seq = []
            for i in range(len(sizes) - 1):
                seq.append(nn.Linear(sizes[i], sizes[i + 1]))
                if i:
                    seq.append(nn.BatchNorm1D(sizes[i + 1]))
                seq.append(activation)
            seq.append(nn.Linear(sizes[-1], noutput))
            self.layers = nn.Sequential(*seq)
            self.init_weights()

        def forward(self, x):
            return paddle.mean(self.layers(x))

        def init_weights(self):
            for m in self.sublayers():
                if isinstance(m, nn.Linear):
                    nn.initializer.Normal(std=0.02)(m.weight)
                    if m.bias is not None:
                        nn.initializer.Constant(0.0)(m.bias)

    class MLP_G(nn.Layer):
        def __init__(self, ninput, noutput, layers, activation=None, gpu=False):
            super().__init__()
            activation = activation or nn.ReLU()
            sizes = [ninput] + [int(x) for x in layers.split("-")]
            seq = []
            for i in range(len(sizes) - 1):
                seq.extend([nn.Linear(sizes[i], sizes[i + 1]), nn.BatchNorm1D(sizes[i + 1]), activation])
            seq.append(nn.Linear(sizes[-1], noutput))
            self.layers = nn.Sequential(*seq)
            self.init_weights()

        def forward(self, x):
            if isinstance(x, np.ndarray):
                x = paddle.to_tensor(x, dtype="float32")
            return self.layers(x)

        def init_weights(self):
            for m in self.sublayers():
                if isinstance(m, nn.Linear):
                    nn.initializer.Normal(std=0.02)(m.weight)
                    if m.bias is not None:
                        nn.initializer.Constant(0.0)(m.bias)

    class MLP_I(nn.Layer):
        def __init__(self, ninput, noutput, layers, activation=None, gpu=False):
            super().__init__()
            activation = activation or nn.ReLU()
            sizes = [ninput] + [int(x) for x in layers.split("-")]
            seq = []
            for i in range(len(sizes) - 1):
                seq.extend([nn.Linear(sizes[i], sizes[i + 1]), nn.BatchNorm1D(sizes[i + 1]), activation])
            seq.append(nn.Linear(sizes[-1], noutput))
            self.layers = nn.Sequential(*seq)
            self.init_weights()

        def forward(self, x):
            return self.layers(x)

        def init_weights(self):
            for m in self.sublayers():
                if isinstance(m, nn.Linear):
                    nn.initializer.Normal(std=0.02)(m.weight)
                    if m.bias is not None:
                        nn.initializer.Constant(0.0)(m.bias)

    class MLP_I_AE(nn.Layer):
        def __init__(self, ninput, noutput, layers, activation=None, gpu=False):
            super().__init__()
            activation = activation or nn.ReLU()
            self.gpu = gpu
            sizes = [ninput] + [int(x) for x in layers.split("-")]
            seq = []
            for i in range(len(sizes) - 1):
                seq.extend([nn.Linear(sizes[i], sizes[i + 1]), nn.BatchNorm1D(sizes[i + 1]), activation])
            seq.append(nn.Linear(sizes[-1], noutput))
            self.feat = nn.Sequential(*seq)
            self.mu = nn.Linear(noutput, noutput)
            self.logvar = nn.Linear(noutput, noutput)
            self.init_weights()

        def forward(self, x):
            h = self.feat(x)
            mu, logvar = self.mu(h), self.logvar(h)
            std = paddle.exp(0.5 * logvar)
            eps = paddle.randn(std.shape)
            return mu + eps * std

        def init_weights(self):
            for m in self.sublayers():
                if isinstance(m, nn.Linear):
                    nn.initializer.Normal(std=0.02)(m.weight)
                    if m.bias is not None:
                        nn.initializer.Constant(0.0)(m.bias)

    # --- Seq2SeqCAE --------------------------------------------------------
    class Seq2SeqCAE(nn.Layer):
        def __init__(self, emsize, nhidden, ntokens, nlayers, conv_windows="5-5-3", conv_strides="2-2-2",
                     conv_layer="500-700-1000", activation=None, noise_radius=0.2, hidden_init=False, dropout=0.0,
                     gpu=True):
            super().__init__()
            activation = activation or nn.LeakyReLU(0.2)
            self.nhidden = nhidden
            self.noise_radius = noise_radius
            self.hidden_init = hidden_init

            # Embeddings
            self.embedding = nn.Embedding(ntokens, emsize, padding_idx=0)
            self.embedding_decoder = nn.Embedding(ntokens, emsize, padding_idx=0)

            # CNN encoder
            c_sizes = [emsize] + [int(x) for x in conv_layer.split("-")]
            k_sizes = list(map(int, conv_windows.split("-")))
            s_sizes = list(map(int, conv_strides.split("-")))
            enc_layers = []
            for i in range(len(c_sizes) - 1):
                enc_layers.extend([
                    nn.Conv1D(c_sizes[i], c_sizes[i + 1], k_sizes[i], stride=s_sizes[i]),
                    nn.BatchNorm1D(c_sizes[i + 1]),
                    activation,
                ])
            self.encoder = nn.Sequential(*enc_layers)
            self.linear = nn.Linear(c_sizes[-1], emsize)

            # LSTM decoder
            self.decoder = nn.LSTM(emsize + nhidden, nhidden, 1, dropout=dropout)
            self.linear_dec = nn.Linear(nhidden, ntokens)

        def encode(self, ids):
            emb = self.embedding(ids).transpose([0, 2, 1])
            h = self.encoder(emb).squeeze(-1)
            h = F.normalize(self.linear(h), p=2, axis=1)
            if self.noise_radius > 0:
                h += paddle.randn(h.shape) * self.noise_radius
            return h

        def decode(self, hidden, ids):
            B, T = ids.shape
            hidden_expand = hidden.unsqueeze(1).expand([B, T, hidden.shape[1]])
            emb_dec = self.embedding_decoder(ids)
            dec_in = paddle.concat([emb_dec, hidden_expand], axis=2).transpose([1, 0, 2])
            out, _ = self.decoder(dec_in)
            out = out.transpose([1, 0, 2])
            return self.linear_dec(out)

        def forward(self, ids, lengths=None, noise=True, encode_only=False, generator=None, inverter=None):
            hidden = self.encode(ids) if noise else self.encode(ids)
            if encode_only:
                return hidden
            if generator is not None and inverter is not None:
                z_hat = inverter(hidden)
                hidden = generator(z_hat)
            return self.decode(hidden, ids)

    # --- Seq2Seq plain LSTM (packed sequence logic simplified) ------------
    class Seq2Seq(nn.Layer):
        def __init__(self, emsize, nhidden, ntokens, nlayers, noise_radius=0.2, hidden_init=False, dropout=0.0,
                     gpu=True):
            super().__init__()
            self.noise_radius = noise_radius
            self.hidden_init = hidden_init
            self.embedding = nn.Embedding(ntokens, emsize, padding_idx=0)
            self.embedding_dec = nn.Embedding(ntokens, emsize, padding_idx=0)
            self.encoder = nn.LSTM(emsize, nhidden, nlayers, dropout=dropout)
            self.decoder = nn.LSTM(emsize + nhidden, nhidden, 1, dropout=dropout)
            self.linear = nn.Linear(nhidden, ntokens)

        def encode(self, ids):
            emb = self.embedding(ids).transpose([1, 0, 2])
            _, (h, _) = self.encoder(emb)
            h = F.normalize(h[-1], p=2, axis=1)
            if self.noise_radius > 0:
                h += paddle.randn(h.shape) * self.noise_radius
            return h

        def decode(self, hidden, ids):
            B, T = ids.shape
            hidden_expand = hidden.unsqueeze(1).expand([B, T, hidden.shape[1]])
            emb_dec = self.embedding_dec(ids)
            dec_in = paddle.concat([emb_dec, hidden_expand], axis=2).transpose([1, 0, 2])
            out, _ = self.decoder(dec_in)
            out = out.transpose([1, 0, 2])
            return self.linear(out)

        def forward(self, ids, lengths=None, noise=True, encode_only=False, generator=None, inverter=None):
            h = self.encode(ids) if noise else self.encode(ids)
            if encode_only:
                return h
            if generator is not None and inverter is not None:
                h = generator(inverter(h))
            return self.decode(h, ids)

    # ------------------------------------------------------------------
    # Load vocabulary --------------------------------------------------
    word2idx = json.load(open(os.path.join(path, "vocab.json"), "r", encoding="utf-8"))
    ntokens = len(word2idx)

    # Instantiate models ------------------------------------------------
    autoencoder = Seq2SeqCAE(emsize=300, nhidden=300, ntokens=ntokens, nlayers=1,
                             noise_radius=0.2, conv_layer="500-700-1000", conv_windows="3-3-3", conv_strides="1-2-2",
                             hidden_init=False, dropout=0.0, gpu=False)
    inverter = MLP_I_AE(ninput=300, noutput=100, layers="300-300")
    gan_gen = MLP_G(ninput=100, noutput=300, layers="300-300")
    gan_disc = MLP_D(ninput=300, noutput=1, layers="300-300")

    # ------------------------------------------------------------------
    # NOTE: Convert and load weights -----------------------------------
    # ------------------------------------------------------------------
    # Expect converted Paddle parameter files named a.pdparams, i.pdparams, etc.
    try:
        autoencoder.set_state_dict(paddle.load(os.path.join(path, "a.pdparams")))
        inverter.set_state_dict(paddle.load(os.path.join(path, "i.pdparams")))
        gan_gen.set_state_dict(paddle.load(os.path.join(path, "g.pdparams")))
        gan_disc.set_state_dict(paddle.load(os.path.join(path, "d.pdparams")))
    except FileNotFoundError:
        print("[WARNING] Paddle parameter files not found; returning un‑initialized models.\n"
              "Convert PyTorch checkpoints to Paddle and save them as a.pdparams / i.pdparams / g.pdparams / d.pdparams in the same directory.")

    return word2idx, autoencoder, inverter, gan_gen, gan_disc
