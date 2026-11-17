"""将 SFT 数据 train.jsonl 转换为 .bin 文件。

输出 (均在 data_cache/sft/ 下):
  sft_tokens.bin  : uint16, 所有样本的 token id 顺序拼接
  sft_mask.bin    : uint8,  与 tokens 等长; 1 = 该 token 作为训练目标
                    (仅答案部分 + <|im_end|> 为 1, 问题与模板为 0)
  sft_lengths.npy : int32, 每条样本的 token 长度, 便于定位与采样

聊天模板 (ChatML 风格):
  <|im_start|>user\\n{question}<|im_end|>\\n<|im_start|>assistant\\n{answer}<|im_end|>
损失只计算 assistant 回答部分。

用法:
    python data/prepare_sft_data.py --input datasets/train.jsonl
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOKENIZER_PATH, BASE_DIR  # noqa: E402

from data.prepare_pretrain_data import load_tokenizer  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402


def build_prompt(question: str) -> str:
    return f"<|im_start|>user\n{question}<|im_end|>\n<|im_start|>assistant\n"


def encode_sample(tok: Tokenizer, question: str, answer: str):
    """返回 (token ids, loss mask)。前缀对齐处理 BPE 边界差异。"""
    prompt = build_prompt(question)
    full = prompt + answer + "<|im_end|>"

    prompt_enc = tok.encode(prompt)
    full_enc = tok.encode(full)

    # 用共同前缀长度对齐, 避免单独编码与整体编码在模板边界处不一致
    p_ids, f_ids = prompt_enc.ids, full_enc.ids
    prefix = 0
    for a, b in zip(p_ids, f_ids):
        if a != b:
            break
        prefix += 1
    if prefix < len(p_ids):
        # 保守处理: 模板最后若干 token (assistant 头) 一并计入 mask=0 区域
        prefix = min(prefix, len(p_ids))

    ids = f_ids
    mask = np.zeros(len(ids), dtype=np.uint8)
    mask[prefix:] = 1  # 答案 + <|im_end|> 计算损失
    return ids, mask


def main():
    ap = argparse.ArgumentParser(description="train.jsonl -> sft .bin 文件")
    ap.add_argument("--input", type=str, default=os.path.join(BASE_DIR, "datasets", "train.jsonl"))
    ap.add_argument("--out_dir", type=str, default=os.path.join(BASE_DIR, "data_cache", "sft"))
    ap.add_argument("--max_lines", type=int, default=0, help="只处理前 N 行 (调试用), 0=全量")
    ap.add_argument("--max_len", type=int, default=512, help="超过该长度的样本直接丢弃")
    args = ap.parse_args()

    tok = load_tokenizer()
    os.makedirs(args.out_dir, exist_ok=True)
    tokens_path = os.path.join(args.out_dir, "sft_tokens.bin")
    mask_path = os.path.join(args.out_dir, "sft_mask.bin")
    lengths_path = os.path.join(args.out_dir, "sft_lengths.npy")

    tokens_f, mask_f = open(tokens_path, "wb"), open(mask_path, "wb")
    lengths = []
    counters = {"kept": 0, "drop": 0}
    t0 = time.time()
    batch = []

    def flush(batch_rows):
        for q, a in batch_rows:
            ids, mask = encode_sample(tok, q, a)
            if len(ids) > args.max_len:
                counters["drop"] += 1
                continue
            tokens_f.write(np.asarray(ids, dtype=np.uint16).tobytes())
            mask_f.write(mask.tobytes())
            lengths.append(len(ids))
            counters["kept"] += 1

    # 按字节进度显示
    file_size = os.path.getsize(args.input)
    pbar = tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024,
                desc="SFT 数据编码", dynamic_ncols=True)
    with open(args.input, "rb") as f:
        for i, raw in enumerate(f):
            pbar.update(len(raw))
            try:
                line = raw.decode("utf-8").strip()
            except UnicodeDecodeError:
                counters["drop"] += 1
                continue
            if not line:
                continue
            try:
                row = json.loads(line)
                q, a = row["question"], row["answer"]
            except (json.JSONDecodeError, KeyError):
                counters["drop"] += 1
                continue
            if not q or not a:
                counters["drop"] += 1
                continue
            batch.append((q, a))
            if len(batch) >= 1024:
                flush(batch)
                batch = []
                pbar.set_postfix(kept=counters["kept"], dropped=counters["drop"])
            if args.max_lines and (i + 1) >= args.max_lines:
                tqdm.write(f"达到 --max_lines={args.max_lines}, 提前结束")
                break
    flush(batch)
    pbar.close()

    tokens_f.close()
    mask_f.close()
    lengths_arr = np.asarray(lengths, dtype=np.int32)
    np.save(lengths_path, lengths_arr)

    total_tokens = int(lengths_arr.sum())
    print(f"\n完成: 保留 {len(lengths_arr):,} 条样本 (丢弃 {counters['drop']} 条), "
          f"共 {total_tokens:,} tokens, 平均 {lengths_arr.mean():.1f} tokens/条, "
          f"耗时 {(time.time() - t0) / 60:.1f} 分钟")
    print(f"输出: {tokens_path}\n      {mask_path}\n      {lengths_path}")


if __name__ == "__main__":
    main()
