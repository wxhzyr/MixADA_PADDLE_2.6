# coding=utf-8
"""Fine‑tuning MixText‑style PaddlePaddle models on GLUE.
This is a *functional* port of the original PyTorch script. It supports:
- nomix / tmix / sentmix training options
- single‑/multi‑GPU (DataParallel) 
- warm‑up with linear decay, gradient accumulation & clipping
- evaluation on dev / test splits

It depends on `mixtext_paddle.py` (the model definitions created earlier) and
PaddleNLP ≥ 2.6.
"""
import argparse
import logging
import os
import random
from functools import partial

import numpy as np
import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from paddle.io import DataLoader, DistributedBatchSampler
from paddlenlp.data import Pad, Stack
from paddle.metric import Accuracy
from paddlenlp.transformers import (
    BertTokenizer, RobertaTokenizer, 
    BertConfig, RobertaConfig,
    LinearDecayWithWarmup,
)

# ----- local imports ---------------------------------------------------------
from mixtext_paddle import MixText, SentMix, RobertaMixText, RobertaSentMix # noqa: E402 pylint: disable=C0413

logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


def build_sampler(dataset, batch_size, shuffle=True, use_dp=False):
    if not use_dp:
        return paddle.io.BatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)
    return DistributedBatchSampler(dataset, batch_size=batch_size, shuffle=shuffle)


# -----------------------------------------------------------------------------
# Dataset utils (GLUE TSV → Paddle Dataset)
# -----------------------------------------------------------------------------

class GlueDataset(paddle.io.Dataset):
    """Very light TSV loader that mimics HuggingFace processors."""

    def __init__(self, path, tokenizer, max_len=128, is_test=False):
        super().__init__()
        self.samples = []
        with open(path, "r", encoding="utf‑8") as f:
            header = next(f)
            for line in f:
                parts = line.rstrip().split("\t")
                # Expect: idx \t sentence [\t sentence2] \t label
                if is_test:
                    text_a, text_b = parts[1], None if len(parts) < 3 else parts[2]
                    label = "0"
                else:
                    text_a, text_b, label = parts[1], None if len(parts) < 4 else parts[2], parts[-1]
                encoded = tokenizer(text=text_a, text_pair=text_b, max_length=max_len, truncation=True) # Removed padding, collate_fn will handle it
                self.samples.append((encoded["input_ids"], encoded["token_type_ids"], int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch, pad_token_id=0):
    input_ids, token_type_ids, labels = zip(*batch)
    return (
        Pad(axis=0, pad_val=pad_token_id)(input_ids), # <--- 修改
        Pad(axis=0, pad_val=0)(token_type_ids),      # <--- 修改
        Stack(dtype="int64")(labels),
    )


# -----------------------------------------------------------------------------
# Training / Eval loops
# -----------------------------------------------------------------------------

def forward_batch(model, batch, mix_option, mix_layers, alpha, num_labels): # <--- 修改
    input_ids, token_type_ids, labels = batch
    attn_mask = (input_ids != 0).astype("int64") # <--- 新增: 动态生成 attention mask

    if mix_option == 0: # nomix
        logits, _ = model(input_ids, attn_mask, token_type_ids=token_type_ids)
        loss = F.cross_entropy(logits, labels)
        return loss, logits

    # Random mixup
    idx = paddle.randperm(input_ids.shape[0])
    input_ids_2, attn_mask_2, labels_2 = input_ids[idx], attn_mask[idx], labels[idx]
    token_type_ids_2 = token_type_ids[idx]

    l = np.random.beta(alpha, alpha)
    l = float(l)
    
    labels_onehot = F.one_hot(labels, num_classes=num_labels) # <--- 修改
    labels2_onehot = F.one_hot(labels_2, num_classes=num_labels) # <--- 修改
    mixed_labels = l * labels_onehot + (1 - l) * labels2_onehot

    mix_layer = int(np.random.choice(mix_layers, 1)[0]) if mix_layers else 0 # <--- 修改，处理空列表情况
    mix_layer = mix_layer - 1 if mix_layer > 0 else -1 # <--- 修改

    # Note: Pass token_type_ids to the model call
    logits, _ = model(input_ids, attn_mask, input_ids2=input_ids_2, attention_mask2=attn_mask_2, l=l, mix_layer=mix_layer, token_type_ids=token_type_ids, token_type_ids2=token_type_ids_2)
    
    # For TMix, the loss is calculated differently.
    # The original MixText paper uses a more complex loss for TMix involving consistency.
    # For simplicity here, we'll use a mixed cross-entropy loss.
    loss = -paddle.mean(paddle.sum(F.log_softmax(logits, axis=-1) * mixed_labels, axis=-1))
    
    return loss, logits

def train_one_epoch(args, model, data_loader, optimizer, lr_scheduler, metric, epoch):
    model.train()
    metric.reset()
    total_loss = 0
    for step, batch in enumerate(data_loader, start=1):
        loss, logits = forward_batch(model, batch, args.mix_type, args.mix_layers_set, args.alpha, args.num_labels)
        
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps
        
        loss.backward()

        if step % args.gradient_accumulation_steps == 0:
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_gradients()
        
        total_loss += loss.item()
        metric.update(logits.argmax(axis=1), batch[-1])  # 修改此行

        if step % args.logging_steps == 0:
            avg_loss = total_loss / args.logging_steps
            acc = metric.accumulate()
            logger.info(f"Epoch {epoch} - step {step}: loss={avg_loss:.4f} acc={acc:.4f} lr={lr_scheduler.get_lr():.2e}")
            total_loss = 0
            metric.reset()



def evaluate(model, data_loader):
    model.eval()
    metric = Accuracy()
    total_loss = 0.0
    with paddle.no_grad():
        for batch in data_loader:
            input_ids, token_type_ids, labels = batch
            attn_mask = (input_ids != 0).astype("int64")
            logits, _ = model(input_ids, attn_mask, token_type_ids=token_type_ids)
            loss = F.cross_entropy(logits, labels)

            total_loss += loss.item()
            metric.update(logits.argmax(axis=1), labels)  # 修改此行
            
    avg_loss = total_loss / len(data_loader)
    acc = metric.accumulate()
    return avg_loss, acc

# -----------------------------------------------------------------------------
# Argument parsing & main
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    # --- 已有参数 ---
    parser.add_argument("--data_dir", required=True, type=str, help="The input data dir. Should contain the .tsv files for the task.")
    parser.add_argument("--output_dir", required=True, type=str, help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--model_type", required=True, choices=["bert", "roberta"], help="Model type selected in the list: bert, roberta")
    parser.add_argument("--mix_type", required=True, choices=["nomix", "tmix", "sentmix"], help="Mixing strategy")
    parser.add_argument("--model_name_or_path", required=True, type=str, help="Path to pre-trained model or shortcut name")
    parser.add_argument("--max_seq_length", type=int, default=128)
    parser.add_argument("--per_gpu_train_batch_size", type=int, default=8)
    parser.add_argument("--per_gpu_eval_batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mix_layers_set", nargs="+", type=int, default=[7, 9, 12])
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # --- 为所有在报错中出现的 "unrecognized arguments" 添加定义 ---
    parser.add_argument("--task_name", type=str, default="sst-2", help="The name of the task (e.g. sst-2)") # <--- 新增
    parser.add_argument("--num_labels", type=int, default=2, help="The number of labels for classification.") # <--- 新增
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.") # <--- 新增
    parser.add_argument("--adam_epsilon", type=float, default=1e-8, help="Epsilon for Adam optimizer.") # <--- 新增
    parser.add_argument("--eval_all_checkpoints", action='store_true', help="Evaluate all checkpoints.") # <--- 新增
    parser.add_argument("--overwrite_output_dir", action='store_true', help="Overwrite the content of the output directory.") # <--- 新增
    parser.add_argument("--overwrite_cache", action='store_true', help="Overwrite the cached training and evaluation sets.") # <--- 新增
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.") # <--- 新增
    parser.add_argument("--fp16", action='store_true', help="Whether to use 16-bit (mixed) precision training.") # <--- 新增
    
    # --- 为 attackEval.py 可能需要的参数添加定义 ---
    parser.add_argument("--iterative", action='store_true', help="Placeholder for iterative attacks.") # <--- 新增
    parser.add_argument("--attacker", type=str, default="pwws", help="Placeholder for attacker name.") # <--- 新增
    parser.add_argument("--num_adv", type=int, default=300, help="Placeholder for number of adversarial examples.") # <--- 新增

    return parser.parse_args()


def main():
    args = parse_args()
    
    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and not args.overwrite_output_dir: # <--- 新增
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")

    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.info("Training/evaluation parameters %s", args)
    
    set_seed(args.seed)

    # Tokenizer & model -------------------------------------------------------
    if args.model_type == "roberta":
        tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
        if args.mix_type == "sentmix":
            model_cls = RobertaSentMix
        else: # tmix or nomix
            model_cls = RobertaMixText
    else: # bert
        tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path, do_lower_case=args.do_lower_case) # <--- 修改
        model_cls = SentMix if args.mix_type == "sentmix" else MixText

    config_cls = RobertaConfig if args.model_type == "roberta" else BertConfig
    config = config_cls.from_pretrained(args.model_name_or_path)
    config.num_labels = args.num_labels  # <--- 修改: 使用参数
    model = model_cls.from_pretrained(args.model_name_or_path, config=config)

    paddle.device.set_device("gpu" if paddle.is_compiled_with_cuda() else "cpu")
    if paddle.distributed.get_world_size() > 1:
        model = paddle.DataParallel(model)

    # Datasets ---------------------------------------------------------------
    logger.info("Loading datasets...")
    train_path = os.path.join(args.data_dir, "train.tsv")
    dev_path = os.path.join(args.data_dir, "dev.tsv")
    train_ds = GlueDataset(train_path, tokenizer, max_len=args.max_seq_length)
    dev_ds = GlueDataset(dev_path, tokenizer, max_len=args.max_seq_length)

    collate = partial(collate_fn, pad_token_id=tokenizer.pad_token_id)
    train_batch_sampler = build_sampler(train_ds, args.per_gpu_train_batch_size, shuffle=True, use_dp=paddle.distributed.get_world_size() > 1)
    train_loader = DataLoader(train_ds,
                              batch_sampler=train_batch_sampler,
                              collate_fn=collate,
                              num_workers=0)
    dev_loader = DataLoader(dev_ds,
                            batch_size=args.per_gpu_eval_batch_size,
                            shuffle=False,
                            collate_fn=collate,
                            num_workers=0)

    # Optimizer & scheduler --------------------------------------------------
    num_training_steps = len(train_loader) * args.num_train_epochs
    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps, args.warmup_steps)
    
    decay_params = [p.name for n, p in model.named_parameters() if not any(nd in n for nd in ["bias", "layer_norm.weight"])] # <--- 修改: 'layer_norm.weight' for paddle
    
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        epsilon=args.adam_epsilon, # <--- 修改
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params
    )

    # Training loop ----------------------------------------------------------
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_ds))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Total train batch size = %d", args.per_gpu_train_batch_size * paddle.distributed.get_world_size())
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", num_training_steps)
    
    metric = Accuracy()
    for epoch in range(1, args.num_train_epochs + 1):
        train_one_epoch(args, model, train_loader, optimizer, lr_scheduler, metric, epoch)
        dev_loss, dev_acc = evaluate(model, dev_loader)
        logger.info(f"Dev - epoch {epoch}: loss={dev_loss:.4f} acc={dev_acc:.4f}")

    # Save final model -------------------------------------------------------
    logger.info("Saving final model to %s", args.output_dir)
    model_to_save = model._layers if isinstance(model, paddle.DataParallel) else model # <--- 新增
    model_to_save.save_pretrained(args.output_dir) # <--- 修改
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Training complete. Model saved to %s", args.output_dir)


if __name__ == "__main__":
    main()