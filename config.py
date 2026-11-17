"""Mini-Chinese-LLM 全局配置。

模型目标: ~20M 参数的中文 decoder-only Transformer (Llama/Qwen 风格:
RoPE + RMSNorm + SwiGLU, 权重绑定 embedding/lm_head, 无 bias)。

分词器: BBPE (byte-level BPE), 词表 16384, 由 tokenizer/train_tokenizer.py 训练。
"""
import json
import os
from dataclasses import dataclass, asdict, field

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------- 路径默认值 ----------
TOKENIZER_PATH = os.path.join(BASE_DIR, "tokenizer", "tokenizer.json")
PRETRAIN_BIN = os.path.join(BASE_DIR, "data_cache", "pretrain", "pretrain.bin")
PRETRAIN_VAL_BIN = os.path.join(BASE_DIR, "data_cache", "pretrain", "pretrain_val.bin")
SFT_TOKENS_BIN = os.path.join(BASE_DIR, "data_cache", "sft", "sft_tokens.bin")
SFT_MASK_BIN = os.path.join(BASE_DIR, "data_cache", "sft", "sft_mask.bin")
SFT_LENGTHS_NPY = os.path.join(BASE_DIR, "data_cache", "sft", "sft_lengths.npy")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")


@dataclass
class ModelConfig:
    """~20M 参数的小型中文 LLM。vocab_size 在加载分词器后由外部覆盖。"""
    vocab_size: int = 16384          # 由 BBPE 分词器决定
    hidden_size: int = 384           # d_model
    num_layers: int = 8              # transformer block 数
    num_heads: int = 6               # head_dim = 384/6 = 64
    intermediate_size: int = 1024    # SwiGLU 隐层维度
    max_seq_len: int = 512           # RoPE 缓存 / 上下文长度
    rope_base: float = 10000.0
    dropout: float = 0.0
    tie_word_embeddings: bool = True
    norm_eps: float = 1e-5

    def __post_init__(self):
        assert self.hidden_size % self.num_heads == 0

    def to_json(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, ensure_ascii=False)

    @classmethod
    def from_json(cls, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return cls(**json.load(f))


@dataclass
class PretrainConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    # 数据
    bin_path: str = PRETRAIN_BIN
    val_bin_path: str = PRETRAIN_VAL_BIN
    batch_size: int = 32             # 每卡样本数 (4090 约 24G 显存可用 64)
    grad_accum_steps: int = 1
    # 优化
    max_steps: int = 60000
    lr: float = 6e-4
    min_lr_ratio: float = 0.1        # cosine 退火到 lr 的 10%
    warmup_steps: int = 500
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    grad_clip: float = 1.0
    # 其它
    eval_interval: int = 1000
    eval_steps: int = 100
    log_interval: int = 20
    save_interval: int = 5000
    dtype: str = "auto"              # auto/bfloat16/float16/float32
    seed: int = 42
    out_dir: str = CKPT_DIR


@dataclass
class SFTConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    tokens_bin: str = SFT_TOKENS_BIN
    mask_bin: str = SFT_MASK_BIN
    lengths_npy: str = SFT_LENGTHS_NPY
    init_from: str = ""              # 预训练 checkpoint 路径, 为空则随机初始化
    batch_size: int = 32
    grad_accum_steps: int = 1
    epochs: int = 2
    max_steps: int = -1              # >0 时覆盖 epochs
    lr: float = 1e-4
    min_lr_ratio: float = 0.1
    warmup_steps: int = 100
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    eval_interval: int = 200
    eval_steps: int = 50
    log_interval: int = 10
    save_interval: int = 1000
    dtype: str = "auto"
    seed: int = 42
    out_dir: str = CKPT_DIR


def count_params(model_cfg: ModelConfig) -> int:
    """按配置静态估算参数量 (用于建模型前确认规模)。"""
    h, i = model_cfg.hidden_size, model_cfg.intermediate_size
    v = model_cfg.vocab_size
    emb = v * h
    per_layer = (
        3 * h * h          # qkv
        + h * h            # attn out
        + 3 * h * i        # SwiGLU: gate/up/down
        + 2 * h            # 两个 RMSNorm
    )
    total = emb + per_layer * model_cfg.num_layers + h
    return total


if __name__ == "__main__":
    cfg = ModelConfig()
    print(f"模型配置: {asdict(cfg)}")
    print(f"参数量(含词表, embedding/lm_head 绑定): {count_params(cfg):,}")
