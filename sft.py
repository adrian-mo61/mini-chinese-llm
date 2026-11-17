"""SFT (监督微调) Mini-Chinese-LLM。

数据: data/prepare_sft_data.py 产出的 sft_tokens.bin / sft_mask.bin / sft_lengths.npy。
ChatML 模板, 仅对 assistant 回答部分 + <|im_end|> 计算损失 (prompt 部分置 -100)。
通常从预训练 checkpoint 初始化:
    python sft.py --init_from checkpoints/pretrain_latest.pt
CPU 冒烟测试:
    python sft.py --init_from checkpoints/pretrain_latest.pt --batch_size 4 --epochs 1
"""
import argparse
import os
import sys

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dataclasses import asdict  # noqa: E402

from config import ModelConfig, SFTConfig, TOKENIZER_PATH, count_params  # noqa: E402
from model import MiniChineseLLM  # noqa: E402
from train_utils import (get_lr, autocast_ctx, configure_optimizer,  # noqa: E402
                         save_checkpoint, load_checkpoint, save_model_cfg, estimate_device)

PAD_ID = 0  # 由分词器确定, 在 main 中重新赋值


class SFTDataset:
    """按条随机采样, 截断/padding 到 max_seq_len。mask=1 的位置作为训练目标。

    最后 val_fraction 比例的样本划为验证集 (按写入顺序取尾部)。
    """

    def __init__(self, tokens_bin: str, mask_bin: str, lengths_npy: str,
                 val_fraction: float = 0.02):
        self.tokens = np.memmap(tokens_bin, dtype=np.uint16, mode="r")
        self.mask = np.memmap(mask_bin, dtype=np.uint8, mode="r")
        self.lengths = np.load(lengths_npy)
        self.offsets = np.concatenate([[0], np.cumsum(self.lengths)])
        assert self.offsets[-1] == len(self.tokens) == len(self.mask)
        n_val = max(int(len(self.lengths) * val_fraction), 1)
        self.val_idx = np.arange(len(self.lengths) - n_val, len(self.lengths))
        self.train_idx = np.arange(0, len(self.lengths) - n_val)

    def __len__(self):
        return len(self.lengths)

    def get(self, idx: int):
        s, e = self.offsets[idx], self.offsets[idx + 1]
        return self.tokens[s:e], self.mask[s:e]

    def get_batch(self, batch_size: int, seq_len: int, pad_id: int, device: str,
                  indices: np.ndarray = None):
        pool = self.train_idx if indices is None else indices
        ix = torch.from_numpy(pool[torch.randint(0, len(pool), (batch_size,)).numpy()])
        x = torch.full((batch_size, seq_len), pad_id, dtype=torch.int64)
        y = torch.full((batch_size, seq_len), -100, dtype=torch.int64)
        for r, i in enumerate(ix):
            ids, mask = self.get(int(i))
            L = min(len(ids), seq_len)
            ids = ids[:L].astype(np.int64)
            # mask 对齐到「该位置的 token 是否作为预测目标」
            tgt_mask = torch.from_numpy(mask[:L].astype(np.int64))
            x[r, :L] = torch.from_numpy(ids)
            # labels 左移一位: 预测 ids[t] 用 ids[t-1] 及之前的内容
            y[r, :L - 1] = torch.from_numpy(ids[1:])
            tm = tgt_mask[1:]
            y[r, :L - 1] = torch.where(tm.bool(), y[r, :L - 1], torch.full_like(y[r, :L - 1], -100))
        if device == "cpu":
            return x.to(device), y.to(device)
        return x.pin_memory().to(device, non_blocking=True), \
            y.pin_memory().to(device, non_blocking=True)


@torch.no_grad()
def eval_val(model, dataset: SFTDataset, cfg: SFTConfig, pad_id: int, device,
             batch_size: int, indices: np.ndarray = None):
    model.eval()
    losses = []
    for _ in range(cfg.eval_steps):
        x, y = dataset.get_batch(batch_size, cfg.model.max_seq_len, pad_id, device,
                                 indices=indices)
        with autocast_ctx(device, cfg.dtype):
            _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    val_loss = float(np.mean(losses))
    print(f"  [val] loss={val_loss:.4f}  ppl={np.exp(val_loss):.2f}")
    return val_loss


def main():
    global PAD_ID
    ap = argparse.ArgumentParser(description="SFT Mini-Chinese-LLM")
    ap.add_argument("--init_from", type=str, required=True,
                    help="预训练 checkpoint 路径")
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None)
    ap.add_argument("--eval_interval", type=int, default=None)
    ap.add_argument("--eval_steps", type=int, default=None)
    ap.add_argument("--save_interval", type=int, default=None)
    ap.add_argument("--log_interval", type=int, default=None)
    args = ap.parse_args()

    cfg = SFTConfig()
    if args.batch_size: cfg.batch_size = args.batch_size
    if args.epochs is not None: cfg.epochs = args.epochs
    if args.max_steps >= 0: cfg.max_steps = args.max_steps
    if args.lr: cfg.lr = args.lr
    if args.dtype: cfg.dtype = args.dtype
    if args.out_dir: cfg.out_dir = args.out_dir
    if args.eval_interval: cfg.eval_interval = args.eval_interval
    if args.eval_steps: cfg.eval_steps = args.eval_steps
    if args.save_interval: cfg.save_interval = args.save_interval
    if args.log_interval: cfg.log_interval = args.log_interval
    device = args.device or estimate_device()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    cfg.model.vocab_size = tok.get_vocab_size()
    PAD_ID = tok.token_to_id("<pad>")

    print(f"设备: {device}")
    dataset = SFTDataset(cfg.tokens_bin, cfg.mask_bin, cfg.lengths_npy)
    print(f"SFT 数据: {len(dataset):,} 条样本 (train {len(dataset.train_idx):,} / "
          f"val {len(dataset.val_idx):,}), {int(dataset.lengths.sum()):,} tokens")

    model = MiniChineseLLM(cfg.model).to(device)
    print(f"实际参数量: {sum(p.numel() for p in model.parameters()):,}")
    step0, _ = load_checkpoint(args.init_from, model, device)
    print(f"从 {args.init_from} 加载预训练权重 (该 ckpt step={step0})")

    optimizer = configure_optimizer(model, cfg.lr, cfg.weight_decay)
    total_steps = cfg.max_steps if cfg.max_steps > 0 else \
        max((len(dataset.train_idx) // cfg.batch_size) * cfg.epochs, 1)
    print(f"总步数: {total_steps} (batch={cfg.batch_size})")

    save_model_cfg(cfg.out_dir, asdict(cfg.model))
    model.train()
    best_val = float("inf")
    step = 0
    done = False
    epoch = 0
    pbar = tqdm(total=total_steps, desc="sft", dynamic_ncols=True, unit="step")
    while not done:
        epoch += 1
        for _ in range(len(dataset.train_idx) // cfg.batch_size):
            lr = get_lr(step, total_steps, cfg.lr, cfg.warmup_steps, cfg.min_lr_ratio)
            for g in optimizer.param_groups:
                g["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            accum_loss = 0.0
            for _ in range(cfg.grad_accum_steps):
                x, y = dataset.get_batch(cfg.batch_size, cfg.model.max_seq_len, PAD_ID, device)
                with autocast_ctx(device, cfg.dtype):
                    _, loss = model(x, y)
                (loss / cfg.grad_accum_steps).backward()
                accum_loss += loss.item() / cfg.grad_accum_steps
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optimizer.step()

            if step % cfg.log_interval == 0:
                pbar.set_postfix(loss=f"{accum_loss:.4f}", lr=f"{lr:.2e}", epoch=epoch)
            pbar.update(1)

            if (step + 1) % cfg.eval_interval == 0 or step == total_steps - 1:
                val_loss = eval_val(model, dataset, cfg, PAD_ID, device, cfg.batch_size,
                                    indices=dataset.val_idx)
                best_val = min(best_val, val_loss)
                save_checkpoint(
                    os.path.join(cfg.out_dir, "sft_best.pt"),
                    model, asdict(cfg.model), step + 1, None,
                    {"val_loss": best_val, "kind": "sft"},
                )
            if (step + 1) % cfg.save_interval == 0 or step == total_steps - 1:
                save_checkpoint(
                    os.path.join(cfg.out_dir, "sft_latest.pt"),
                    model, asdict(cfg.model), step + 1, optimizer,
                    {"kind": "sft"},
                )
            step += 1
            if step >= total_steps:
                done = True
                break
    pbar.close()

    print("SFT 完成")


if __name__ == "__main__":
    main()
