"""预训练 ~20M 中文 LLM (next-token prediction on pretrain.bin)。

数据: data/prepare_pretrain_data.py 产出的 uint16 .bin (memmap 随机窗口采样)。
命令示例 (4090 单卡):
    python pretrain.py --batch_size 64 --max_steps 60000
CPU 冒烟测试:
    python pretrain.py --batch_size 8 --max_steps 20 --eval_interval 10

训练完成后 checkpoint 保存在 checkpoints/pretrain_*.pt, 供 sft.py / generate.py 使用。
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import asdict  # noqa: E402

from config import ModelConfig, PretrainConfig, TOKENIZER_PATH, count_params  # noqa: E402
from model import MiniChineseLLM  # noqa: E402
from train_utils import (get_lr, autocast_ctx, configure_optimizer,  # noqa: E402
                         save_checkpoint, load_checkpoint, save_model_cfg, estimate_device)


def get_batch(data: np.memmap, batch_size: int, seq_len: int, device: str):
    """从 .bin 中随机采样 [batch, seq_len+1] 窗口。"""
    max_start = len(data) - seq_len - 1
    if max_start <= 0:
        raise ValueError(
            f"数据集 ({len(data):,} tokens) 小于 seq_len+1={seq_len + 1}, "
            "请增大 --val_fraction 重新生成 .bin 或减小 --seq_len")
    ix = torch.randint(0, max_start, (batch_size,))
    x = torch.stack([
        torch.from_numpy(data[i:i + seq_len].astype(np.int64)) for i in ix
    ])
    y = torch.stack([
        torch.from_numpy(data[i + 1:i + seq_len + 1].astype(np.int64)) for i in ix
    ])
    if device != "cpu":
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def eval_val(model, val_data, cfg: PretrainConfig, device, batch_size: int):
    model.eval()
    losses = []
    for _ in range(cfg.eval_steps):
        x, y = get_batch(val_data, batch_size, cfg.model.max_seq_len, device)
        with autocast_ctx(device, cfg.dtype):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    val_loss = float(np.mean(losses))
    print(f"  [val] loss={val_loss:.4f}  ppl={np.exp(val_loss):.2f}")
    return val_loss


def main():
    ap = argparse.ArgumentParser(description="预训练 Mini-Chinese-LLM")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seq_len", type=int, default=None, help="覆盖 max_seq_len")
    ap.add_argument("--grad_accum", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--init_from", type=str, default="",
                    help="从已有 checkpoint 恢复训练 (需要 optimizer 状态)")
    ap.add_argument("--eval_interval", type=int, default=None)
    ap.add_argument("--eval_steps", type=int, default=None)
    ap.add_argument("--save_interval", type=int, default=None)
    ap.add_argument("--log_interval", type=int, default=None)
    args = ap.parse_args()

    cfg = PretrainConfig()
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.max_steps: cfg.max_steps = args.max_steps
    if args.lr: cfg.lr = args.lr
    if args.seq_len: cfg.model.max_seq_len = args.seq_len
    if args.grad_accum: cfg.grad_accum_steps = args.grad_accum
    if args.dtype: cfg.dtype = args.dtype
    if args.out_dir: cfg.out_dir = args.out_dir
    if args.eval_interval: cfg.eval_interval = args.eval_interval
    if args.eval_steps: cfg.eval_steps = args.eval_steps
    if args.save_interval: cfg.save_interval = args.save_interval
    if args.log_interval: cfg.log_interval = args.log_interval
    device = args.device or estimate_device()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # 词表大小以分词器为准
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    cfg.model.vocab_size = tok.get_vocab_size()

    print(f"设备: {device}")
    print(f"模型参数量: {count_params(cfg.model):,} (vocab={cfg.model.vocab_size})")

    model = MiniChineseLLM(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"实际参数量: {n_params:,}")

    train_data = np.memmap(cfg.bin_path, dtype=np.uint16, mode="r")
    val_data = np.memmap(cfg.val_bin_path, dtype=np.uint16, mode="r")
    print(f"数据: train {len(train_data):,} tokens, val {len(val_data):,} tokens")

    optimizer = configure_optimizer(model, cfg.lr, cfg.weight_decay,
                                    (cfg.adam_beta1, cfg.adam_beta2))
    start_step = 0
    if args.init_from:
        step0, extra = load_checkpoint(args.init_from, model, device)
        if "optimizer" in extra:
            optimizer.load_state_dict(extra["optimizer"])
            print("已恢复 optimizer 状态")
        start_step = step0
        print(f"从 {args.init_from} 恢复, 当前 step={start_step}")

    save_model_cfg(cfg.out_dir, asdict(cfg.model))
    model.train()
    t0 = time.time()
    best_val = float("inf")
    tokens_per_step = cfg.batch_size * cfg.model.max_seq_len

    for step in range(start_step, cfg.max_steps):
        lr = get_lr(step, cfg.max_steps, cfg.lr, cfg.warmup_steps, cfg.min_lr_ratio)
        for g in optimizer.param_groups:
            g["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for _ in range(cfg.grad_accum_steps):
            x, y = get_batch(train_data, cfg.batch_size, cfg.model.max_seq_len, device)
            with autocast_ctx(device, cfg.dtype):
                _, loss = model(x, y)
            (loss / cfg.grad_accum_steps).backward()
            accum_loss += loss.item() / cfg.grad_accum_steps
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % cfg.log_interval == 0:
            dt = time.time() - t0
            tps = cfg.log_interval * tokens_per_step * cfg.grad_accum_steps / max(dt, 1e-9)
            print(f"step {step:>6d}/{cfg.max_steps} | loss {accum_loss:.4f} | "
                  f"lr {lr:.2e} | {tps / 1e6:.2f}M tokens/s")
            t0 = time.time()

        if (step + 1) % cfg.eval_interval == 0 or step == cfg.max_steps - 1:
            val_loss = eval_val(model, val_data, cfg, device, cfg.batch_size)
            best_val = min(best_val, val_loss)
            save_checkpoint(
                os.path.join(cfg.out_dir, "pretrain_best.pt"),
                model, asdict(cfg.model), step + 1, None,
                {"val_loss": best_val, "kind": "pretrain"},
            )
        if (step + 1) % cfg.save_interval == 0 or step == cfg.max_steps - 1:
            save_checkpoint(
                os.path.join(cfg.out_dir, "pretrain_latest.pt"),
                model, asdict(cfg.model), step + 1, optimizer,
                {"kind": "pretrain"},
            )

    print("预训练完成")


if __name__ == "__main__":
    main()
