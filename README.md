# Mini-Chinese-LLM

从零实现的 **~20M 参数中文大语言模型**：包含 BBPE 分词器、大规模预训练、SFT 微调与对话生成，完整覆盖「预训练 → SFT → 推理」全流程。数据集为 12.7GB 中文语料（`datasets/train.txt`，约 2411 万篇文档）与 27.4 万条中文问答对（`datasets/train.jsonl`）。

## 模型结构

Llama/Qwen 风格的 decoder-only Transformer：

| 配置 | 值 |
|---|---|
| 分词器 | BBPE (byte-level BPE)，词表 16384 |
| 参数量 | ≈ 20.5M（embedding 与 lm_head 权重绑定） |
| hidden_size | 384 |
| num_layers | 8 |
| num_heads | 6 (head_dim=64) |
| FFN | SwiGLU，intermediate=1024 |
| 位置编码 | RoPE |
| 归一化 | RMSNorm (pre-norm)，无 bias |
| 上下文长度 | 512 |

`python config.py` 可打印参数量。

## 目录结构

```
├── config.py                    # 模型/训练配置 (dataclass)
├── model.py                     # MiniChineseLLM: RoPE + RMSNorm + SwiGLU
├── run_all.sh                   # 一键训练脚本 (Ubuntu/GPU: 分词器→.bin→预训练→SFT)
├── pretrain.py                  # 预训练 (从 .bin 采样训练窗口)
├── sft.py                       # SFT 微调 (ChatML 模板, 仅对回答计算损失)
├── generate.py                  # 对话生成 / 交互式聊天
├── train_utils.py               # lr 调度 / optimizer / checkpoint 工具
├── tokenizer/
│   ├── train_tokenizer.py       # 训练 BBPE 分词器
│   └── tokenizer.json           # 训练产出的分词器 (已提交)
├── data/
│   ├── prepare_pretrain_data.py # train.txt -> pretrain.bin (+ val split)
│   └── prepare_sft_data.py      # train.jsonl -> sft_tokens/mask/lengths
├── samples/                     # 小样本数据 (可直接跑通全流程)
└── datasets/                    # 原始数据 (不入库, 见 .gitignore)
```

## 快速开始

### 一键训练（Ubuntu / 云 GPU）

`run_all.sh` 依次完成：依赖检查（直接使用当前 python3，仅缺依赖时才安装）→
BBPE 分词器训练 → 数据转 .bin → 预训练 → SFT，并在各阶段间自动跳过已完成步骤
（checkpoint 存在时自动断点续训，适合抢占式云实例）。日志写入 `logs/`。

```bash
bash run_all.sh                      # 全量训练 (默认: 16384 词表 / 60000 步预训练 / 2 轮 SFT)
bash run_all.sh --demo               # 训练结束后附带一次生成演示
bash run_all.sh --force              # 忽略已有产物全部重跑
bash run_all.sh --help               # 查看全部参数 (数据目录/步数/batch/学习率等均可覆盖)
```

小语料快速验证（几百 KB 数据 + 50 步训练，CPU 也能跑）：

```bash
bash run_all.sh --data-dir samples \
    --tokenizer-sample-mb 0 --val-fraction 0.1 \
    --pretrain-steps 50 --pretrain-batch-size 8 \
    --sft-epochs 1 --save-interval 50 --demo
```

### 分步执行

#### 0. 环境

```bash
pip install -r requirements.txt
# GPU 训练请按官方地址安装 CUDA 版 torch, 如:
# pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### 1. 训练 BBPE 分词器

对 12.7GB 语料顺序采样 100MB 训练词表（`--sample_size 0` 读全量）：

```bash
python tokenizer/train_tokenizer.py --input datasets/train.txt \
    --vocab_size 16384 --sample_size 100000000
```

### 2. 数据集转 .bin（训练前预处理）

预训练语料按文档切分、BBPE 编码后以 uint16 流式写入 `.bin`（约 3B tokens，6GB），
自动按 0.5% 抽出验证集：

```bash
python data/prepare_pretrain_data.py --input datasets/train.txt
# 调试时可只读前 N 字节: --max_bytes 200000000
```

SFT 数据转 `.bin`：token 存 `sft_tokens.bin`（uint16），损失掩码存 `sft_mask.bin`
（uint8，1 = 该 token 作为训练目标），样本长度存 `sft_lengths.npy`：

```bash
python data/prepare_sft_data.py --input datasets/train.jsonl
```

### 3. 预训练

```bash
# 4090 单卡参考配置
python pretrain.py --batch_size 64 --grad_accum 2 --max_steps 60000

# 恢复训练
python pretrain.py --max_steps 60000 --init_from checkpoints/pretrain_latest.pt
```

默认超参：AdamW(β2=0.95)，lr=6e-4，warmup 500 步 + cosine 退火，grad clip 1.0，
每 1000 步验证并保存 `pretrain_best.pt`。

### 4. SFT 微调

聊天模板（损失只计算回答部分）：

```
<|im_start|>user
{question}<|im_end|>
<|im_start|>assistant
{answer}<|im_end|>
```

```bash
python sft.py --init_from checkpoints/pretrain_latest.pt \
    --batch_size 32 --epochs 2 --lr 1e-4
```

### 5. 对话生成

```bash
python generate.py --ckpt checkpoints/sft_best.pt --prompt "你好，介绍一下你自己"
python generate.py --ckpt checkpoints/sft_best.pt --chat   # 交互模式
```

## 用 samples/ 单独跑各阶段（无需大数据集）

仓库自带小样本数据（预训练语料前 200KB + SFT 前 500 条），也可直接用上面的
一键脚本 `--data-dir samples` 跑通。单独调试某个阶段时：

```bash
python tokenizer/train_tokenizer.py --input samples/train.txt --sample_size 0
python data/prepare_pretrain_data.py --input samples/train.txt --val_fraction 0.1
python data/prepare_sft_data.py --input samples/train.jsonl
python pretrain.py --batch_size 8 --max_steps 50 --eval_interval 25 --save_interval 50
python sft.py --init_from checkpoints/pretrain_latest.pt --batch_size 8 --max_steps 30
python generate.py --ckpt checkpoints/sft_best.pt --prompt "你好"
```

## 训练成本参考

| 阶段 | 数据 | 硬件 | 预计耗时 |
|---|---|---|---|
| 分词器训练 | 100MB 采样 | CPU | ~10 分钟 |
| 数据转 .bin | 12.7GB 全量 | CPU | ~1 小时 |
| 预训练 | ~3B tokens | 1×4090 | ~2-3 天 (60000 步 × 64×512 tokens) |
| SFT | 27.4 万条 QA | 1×4090 | ~1 小时 |

无 GPU 时可把 `--max_steps` 调小（如 2000）在 CPU 上先验证流程。
