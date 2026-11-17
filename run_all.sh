#!/usr/bin/env bash
# ============================================================================
# Mini-Chinese-LLM 一键训练脚本 (Ubuntu / 云 GPU, 请用 bash 运行)
#
# 流程: 依赖检查 -> BBPE 分词器训练 -> 数据转 .bin -> 预训练 -> SFT -> (可选) 生成演示
#
# 环境: 默认直接使用当前 python3 (云服务器环境通常已就绪), 仅在缺少
#       torch/tokenizers/numpy/tqdm 时才 pip install; 如需隔离环境用 --venv。
#
# 用法:
#   bash run_all.sh                                  # 全量训练 (默认参数)
#   bash run_all.sh --force                          # 忽略已有产物, 全部重跑
#   bash run_all.sh --demo                           # 训练完后附带生成演示
#   bash run_all.sh --pretrain-steps 2000 --pretrain-batch-size 32
#
# 常用参数:
#   --data-dir DIR            数据目录, 含 train.txt / train.jsonl (默认 datasets;
#                             仓库自带小样本可直接 --data-dir samples)
#   --tokenizer-sample-mb N   分词器训练采样 MB 数, 0=全量 (默认 100)
#   --vocab-size N            BBPE 词表大小 (默认 16384)
#   --pretrain-steps N        预训练步数 (默认 60000)
#   --pretrain-batch-size N   预训练 batch (默认 64)
#   --pretrain-accum N        预训练梯度累积 (默认 2)
#   --pretrain-lr LR          预训练学习率 (默认 6e-4)
#   --sft-epochs N            SFT 轮数 (默认 2)
#   --sft-batch-size N        SFT batch (默认 32)
#   --sft-lr LR               SFT 学习率 (默认 1e-4)
#   --save-interval N         checkpoint 保存间隔步数 (默认 5000)
#   --val-fraction F          预训练验证集比例 (默认 0.005; 小语料建议 0.1)
#   --venv                    在 .venv 虚拟环境中运行 (默认直接用当前 python3)
#   --skip-install            跳过依赖安装 (依赖缺失时会直接报错)
#   --require-gpu             无可用 GPU 时直接报错退出
#   --force                   忽略已有产物全部重跑 (会覆盖 checkpoint)
#   --demo                    最后运行一次生成演示
#
# 断点续跑: 各阶段检测到产物已存在时自动跳过 (分词器/数据/checkpoint),
# 预训练 checkpoint 存在时自动 --init_from 续训, 适合抢占式云实例。
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

# ---------------- 默认参数 ----------------
DATA_DIR="datasets"
TOKENIZER_SAMPLE_MB=100
VOCAB_SIZE=16384
PRETRAIN_STEPS=60000
PRETRAIN_BS=64
PRETRAIN_ACC=2
PRETRAIN_LR=6e-4
SFT_EPOCHS=2
SFT_BS=32
SFT_LR=1e-4
SAVE_INTERVAL=5000
VAL_FRACTION=0.005
USE_VENV=0
SKIP_INSTALL=0
REQUIRE_GPU=0
FORCE=0
DEMO=0

usage() { grep -E '^#   (--|-h)' "$0" | sed 's/^#   //'; exit 1; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data-dir)             DATA_DIR="$2"; shift 2 ;;
        --tokenizer-sample-mb)  TOKENIZER_SAMPLE_MB="$2"; shift 2 ;;
        --vocab-size)           VOCAB_SIZE="$2"; shift 2 ;;
        --pretrain-steps)       PRETRAIN_STEPS="$2"; shift 2 ;;
        --pretrain-batch-size)  PRETRAIN_BS="$2"; shift 2 ;;
        --pretrain-accum)       PRETRAIN_ACC="$2"; shift 2 ;;
        --pretrain-lr)          PRETRAIN_LR="$2"; shift 2 ;;
        --sft-epochs)           SFT_EPOCHS="$2"; shift 2 ;;
        --sft-batch-size)       SFT_BS="$2"; shift 2 ;;
        --sft-lr)               SFT_LR="$2"; shift 2 ;;
        --save-interval)        SAVE_INTERVAL="$2"; shift 2 ;;
        --val-fraction)         VAL_FRACTION="$2"; shift 2 ;;
        --venv)                 USE_VENV=1; shift ;;
        --skip-install)         SKIP_INSTALL=1; shift ;;
        --require-gpu)          REQUIRE_GPU=1; shift ;;
        --force)                FORCE=1; shift ;;
        --demo)                 DEMO=1; shift ;;
        -h|--help)              usage ;;
        *) echo "未知参数: $1"; usage ;;
    esac
done

# ---------------- 日志 ----------------
mkdir -p logs
LOG_FILE="logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

banner() {
    echo ""
    echo "======================================================================"
    echo "[$(date '+%F %T')] $1"
    echo "======================================================================"
}
exists() { [[ -f "$1" ]]; }

# ---------------- 校验数据 ----------------
for f in "$DATA_DIR/train.txt" "$DATA_DIR/train.jsonl"; do
    if [[ ! -f "$f" ]]; then
        echo "[错误] 找不到数据文件: $f (--data-dir 指向的目录需包含 train.txt 与 train.jsonl)"
        exit 1
    fi
done

T0=$SECONDS
banner "阶段 0/5: 环境检查"

# python 解释器: Ubuntu 默认只有 python3; 有 python 时优先
if [[ "$USE_VENV" -eq 1 ]]; then
    if [[ ! -d .venv ]]; then
        echo "创建虚拟环境 .venv ..."
        "$(command -v python3 || command -v python)" -m venv .venv || {
            echo "[错误] venv 创建失败, 请安装 python3-venv 或改用当前环境 (去掉 --venv)"; exit 1; }
    fi
    # shellcheck disable=SC1091
    source .venv/bin/activate
    PY="$(command -v python)"
else
    PY="$(command -v python3 || command -v python || true)"
    if [[ -z "$PY" ]]; then
        echo "[错误] 未找到 python3/python, 请先安装 Python >= 3.10"
        exit 1
    fi
fi
echo "Python: $("$PY" --version) @ $PY"

# 依赖检查: 已装齐就完全不动环境
if "$PY" -c "import torch, tokenizers, numpy, tqdm" 2>/dev/null; then
    echo "依赖 (torch/tokenizers/numpy/tqdm) 已就绪, 跳过安装"
else
    if [[ "$SKIP_INSTALL" -eq 1 ]]; then
        echo "[错误] 依赖缺失但指定了 --skip-install, 请手动安装: $PY -m pip install -r requirements.txt"
        exit 1
    fi
    echo "缺少依赖, 执行 $PY -m pip install -r requirements.txt ..."
    "$PY" -m pip install -r requirements.txt
fi

# GPU 检查
if "$PY" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)"; then
    echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo '未知型号')"
else
    if [[ "$REQUIRE_GPU" -eq 1 ]]; then
        echo "[错误] 未检测到可用 GPU (--require-gpu), 退出"
        exit 1
    fi
    echo "[警告] 未检测到 GPU, 将在 CPU 上训练 (仅建议小规模验证)"
fi

# ---------------- 阶段 1: 分词器 ----------------
banner "阶段 1/5: BBPE 分词器训练"
if exists tokenizer/tokenizer.json && [[ "$FORCE" -eq 0 ]]; then
    echo "tokenizer/tokenizer.json 已存在, 跳过 (重跑请加 --force)"
else
    "$PY" tokenizer/train_tokenizer.py \
        --input "$DATA_DIR/train.txt" \
        --vocab_size "$VOCAB_SIZE" \
        --sample_size $((TOKENIZER_SAMPLE_MB * 1000000))
fi

# ---------------- 阶段 2: 预训练数据转 .bin ----------------
banner "阶段 2/5: 预训练语料 -> .bin"
if exists data_cache/pretrain/pretrain.bin && exists data_cache/pretrain/pretrain_val.bin \
   && [[ "$FORCE" -eq 0 ]]; then
    echo "data_cache/pretrain/*.bin 已存在, 跳过 (重跑请加 --force)"
else
    "$PY" data/prepare_pretrain_data.py --input "$DATA_DIR/train.txt" \
        --val_fraction "$VAL_FRACTION"
fi

# ---------------- 阶段 3: SFT 数据转 .bin ----------------
banner "阶段 3/5: SFT 数据 -> .bin"
if exists data_cache/sft/sft_tokens.bin && exists data_cache/sft/sft_mask.bin \
   && exists data_cache/sft/sft_lengths.npy && [[ "$FORCE" -eq 0 ]]; then
    echo "data_cache/sft/* 已存在, 跳过 (重跑请加 --force)"
else
    "$PY" data/prepare_sft_data.py --input "$DATA_DIR/train.jsonl"
fi

# ---------------- 阶段 4: 预训练 ----------------
banner "阶段 4/5: 预训练 (${PRETRAIN_STEPS} 步, batch ${PRETRAIN_BS} x accum ${PRETRAIN_ACC})"
PRETRAIN_ARGS=(--batch_size "$PRETRAIN_BS" --grad_accum "$PRETRAIN_ACC"
               --max_steps "$PRETRAIN_STEPS" --lr "$PRETRAIN_LR"
               --save_interval "$SAVE_INTERVAL")
if exists checkpoints/pretrain_latest.pt && [[ "$FORCE" -eq 0 ]]; then
    echo "发现 checkpoints/pretrain_latest.pt, 从断点续训 ..."
    PRETRAIN_ARGS+=(--init_from checkpoints/pretrain_latest.pt)
fi
"$PY" pretrain.py "${PRETRAIN_ARGS[@]}"

# ---------------- 阶段 5: SFT ----------------
banner "阶段 5/5: SFT 微调 (epochs ${SFT_EPOCHS}, batch ${SFT_BS})"
if exists checkpoints/sft_best.pt && [[ "$FORCE" -eq 0 ]]; then
    echo "checkpoints/sft_best.pt 已存在, 跳过 SFT (重跑请加 --force)"
else
    "$PY" sft.py \
        --init_from checkpoints/pretrain_latest.pt \
        --batch_size "$SFT_BS" \
        --epochs "$SFT_EPOCHS" \
        --lr "$SFT_LR" \
        --save_interval "$SAVE_INTERVAL"
fi

# ---------------- 结束 ----------------
banner "全部完成"
echo "产物位置:"
echo "  分词器:     tokenizer/tokenizer.json"
echo "  预训练数据: data_cache/pretrain/*.bin"
echo "  SFT 数据:   data_cache/sft/*"
echo "  checkpoints: checkpoints/pretrain_latest.pt, checkpoints/sft_best.pt"
echo "  日志:       $LOG_FILE"
echo ""
echo "对话测试: $PY generate.py --ckpt checkpoints/sft_best.pt --prompt \"你好\""

if [[ "$DEMO" -eq 1 ]]; then
    echo ""
    "$PY" generate.py --ckpt checkpoints/sft_best.pt --prompt "你好，介绍一下你自己"
fi

echo ""
echo "总耗时: $(( (SECONDS - T0) / 60 )) 分 $(( (SECONDS - T0) % 60 )) 秒"
