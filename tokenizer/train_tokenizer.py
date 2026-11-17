"""训练 BBPE (byte-level BPE) 分词器。

对大文件 (如 12.7GB 的 train.txt) 采用「顺序采样前 N 字节」的方式读取,
采样时保证按行截断, 不产生半行噪声。训练完成后保存为
tokenizer/tokenizer.json, 并打印特殊 token 的 id 与压缩比。

用法:
    python tokenizer/train_tokenizer.py \
        --input datasets/train.txt --sample_size 100000000 --vocab_size 16384
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOKENIZER_PATH, BASE_DIR  # noqa: E402

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders  # noqa: E402

# 特殊 token: BBPE 词表中预留, SFT 聊天模板使用 ChatML 风格
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<|im_start|>", "<|im_end|>"]

PAD, BOS, EOS, IM_START, IM_END = SPECIAL_TOKENS


def read_sample(input_path: str, sample_size: int) -> str:
    """顺序读取前 sample_size 字节, 丢掉最后一个不完整行。"""
    if sample_size <= 0:
        return None  # None 表示读全量 (交给 tokenizers 自己迭代)
    print(f"采样前 {sample_size / 1e6:.0f}MB 文本用于训练分词器 ...")
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read(sample_size)
    text = text[: text.rfind("\n") + 1]
    return text


def train_bbpe(input_path: str, vocab_size: int, sample_size: int) -> Tokenizer:
    tokenizer = Tokenizer(models.BPE(unk_token=None))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
        show_progress=True,
    )
    sample = read_sample(input_path, sample_size)
    if sample is not None:
        tokenizer.train_from_iterator([sample], trainer=trainer)
    else:
        tokenizer.train([input_path], trainer=trainer)
    return tokenizer


def main():
    ap = argparse.ArgumentParser(description="训练 BBPE 分词器")
    ap.add_argument("--input", type=str, default=os.path.join(BASE_DIR, "datasets", "train.txt"))
    ap.add_argument("--vocab_size", type=int, default=16384)
    ap.add_argument("--sample_size", type=int, default=100 * 1024 * 1024,
                    help="用于训练分词器的采样字节数; 0 表示全量 (12.7GB 会很慢)")
    args = ap.parse_args()

    tokenizer = train_bbpe(args.input, args.vocab_size, args.sample_size)

    os.makedirs(os.path.dirname(TOKENIZER_PATH), exist_ok=True)
    tokenizer.save(TOKENIZER_PATH)

    vocab = tokenizer.get_vocab()
    print(f"\n词表实际大小: {tokenizer.get_vocab_size()}")
    for tok in SPECIAL_TOKENS:
        print(f"  {tok!r}: id={vocab.get(tok)}")
    # 压缩比测试: 中文语料上 BBPE 一般 1 token ≈ 1.4~1.8 个汉字
    probe = "今天天气怎么样？北京明天的气温大约是十度左右，适合出门散步。"
    ids = tokenizer.encode(probe).ids
    print(f"压缩比示例: {probe!r} ({len(probe)} 字符) -> {len(ids)} tokens "
          f"(≈{len(probe) / len(ids):.2f} 字符/token)")


if __name__ == "__main__":
    main()
