"""将预训练语料 train.txt 转换为 token id 二进制文件 (.bin)。

流程: 逐行读取文档 -> BBPE 编码 -> 文档末尾追加 <eos> -> 以 uint16
顺序写入 pretrain.bin (训练) 与 pretrain_val.bin (验证)。
12.7GB 级别文件采用流式写入, 内存占用仅一个批次。

用法:
    python data/prepare_pretrain_data.py --input datasets/train.txt
    # 冒烟测试: 只处理前 200MB
    python data/prepare_pretrain_data.py --input datasets/train.txt --max_bytes 200000000
"""
import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOKENIZER_PATH, BASE_DIR  # noqa: E402

from tokenizers import Tokenizer  # noqa: E402

BATCH_DOCS = 2048          # 每批编码的文档数
WRITE_CHUNK = 1 << 24      # 写缓冲 (16M token ≈ 32MB)
UINT16_MAX = 65535


def load_tokenizer() -> Tokenizer:
    assert os.path.exists(TOKENIZER_PATH), (
        f"未找到分词器 {TOKENIZER_PATH}, 请先运行 tokenizer/train_tokenizer.py")
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    if tok.get_vocab_size() > UINT16_MAX + 1:
        raise ValueError("词表超过 65536, 无法用 uint16 存储, 请改小 vocab_size")
    return tok


class BinWriter:
    """把 uint16 数组顺序写入 .bin, 结束时按实际长度截断。"""

    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.path = path
        self.f = open(path, "wb")
        self.n_tokens = 0
        self._buf = []

    def write(self, ids):
        arr = np.asarray(ids, dtype=np.uint16)
        self._buf.append(arr)
        self.n_tokens += arr.size
        if sum(a.nbytes for a in self._buf) >= WRITE_CHUNK:
            self._flush()

    def _flush(self):
        if self._buf:
            self.f.write(np.concatenate(self._buf).tobytes())
            self._buf = []

    def close(self):
        self._flush()
        self.f.close()

    def finalize(self):
        """写完后统计信息打印。"""
        self.close()
        size_mb = os.path.getsize(self.path) / 1e6
        print(f"  已写出 {self.path}: {self.n_tokens:,} tokens ({size_mb:.1f} MB, uint16)")


def main():
    ap = argparse.ArgumentParser(description="train.txt -> pretrain.bin")
    ap.add_argument("--input", type=str, default=os.path.join(BASE_DIR, "datasets", "train.txt"))
    ap.add_argument("--out_dir", type=str, default=os.path.join(BASE_DIR, "data_cache", "pretrain"))
    ap.add_argument("--max_bytes", type=int, default=0, help="只读取前 N 字节 (调试用), 0=全量")
    ap.add_argument("--val_fraction", type=float, default=0.005, help="验证集文档占比")
    ap.add_argument("--resume", action="store_true", help="跳过已存在的输出文件")
    args = ap.parse_args()

    train_path = os.path.join(args.out_dir, "pretrain.bin")
    val_path = os.path.join(args.out_dir, "pretrain_val.bin")
    if args.resume and os.path.exists(train_path) and os.path.exists(val_path):
        print("输出文件已存在, 跳过 (--resume)")
        return

    tok = load_tokenizer()
    eos_id = tok.token_to_id("<eos>")
    assert eos_id is not None, "分词器缺少 <eos> 特殊 token"

    train_writer = BinWriter(train_path)
    val_writer = BinWriter(val_path)

    n_docs = 0
    total_bytes = 0
    t0 = time.time()
    batch_texts = []

    def flush_batch():
        nonlocal batch_texts
        if not batch_texts:
            return
        encodings = tok.encode_batch(batch_texts)
        is_val = ((n_docs - len(batch_texts)) + np.arange(len(batch_texts))) % round(
            1 / max(args.val_fraction, 1e-9)) == 0
        for enc, is_v in zip(encodings, is_val):
            writer = val_writer if is_v else train_writer
            writer.write(enc.ids + [eos_id])
        batch_texts = []

    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            batch_texts.append(text)
            n_docs += 1
            total_bytes += len(line.encode("utf-8"))
            if len(batch_texts) >= BATCH_DOCS:
                flush_batch()
            if n_docs % 500000 == 0:
                speed = total_bytes / 1e6 / max(time.time() - t0, 1e-9)
                print(f"  已处理 {n_docs:,} 篇文档 / {total_bytes / 1e9:.2f}GB ({speed:.1f} MB/s)")
            if args.max_bytes and total_bytes >= args.max_bytes:
                print(f"达到 --max_bytes={args.max_bytes}, 提前结束")
                break
    flush_batch()

    train_writer.finalize()
    val_writer.finalize()
    dt = time.time() - t0
    n_tokens = train_writer.n_tokens + val_writer.n_tokens
    chars_per_token = max(total_bytes / max(n_tokens, 1), 1e-9)
    print(f"\n完成: {n_docs:,} 篇文档, 共 {n_tokens:,} tokens, "
          f"耗时 {dt / 60:.1f} 分钟 ({total_bytes / 1e6 / dt:.1f} MB/s), "
          f"压缩比 ≈ {chars_per_token:.2f} 字节/token")


if __name__ == "__main__":
    main()
