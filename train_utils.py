"""pretrain.py 与 sft.py 共用的训练工具。"""
import json
import math
import os

import torch


def get_lr(step: int, max_steps: int, base_lr: float, warmup_steps: int,
           min_lr_ratio: float = 0.1) -> float:
    """warmup + cosine 退火。"""
    if step < warmup_steps:
        return base_lr * (step + 1) / max(warmup_steps, 1)
    progress = (step - warmup_steps) / max(max_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    return base_lr * (min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress)))


def autocast_ctx(device: str, dtype: str):
    """根据设备与 --dtype 返回 autocast 上下文; CPU 上保持 fp32。"""
    if device == "cpu":
        return torch.autocast("cpu", enabled=False)
    if dtype == "auto":
        if device == "cuda" and torch.cuda.is_bf16_supported():
            dt = torch.bfloat16
        else:
            dt = torch.float16
    else:
        dt = {"float32": torch.float32, "bfloat16": torch.bfloat16,
              "float16": torch.float16}[dtype]
    return torch.autocast(device_type=device, dtype=dt)


def configure_optimizer(model: torch.nn.Module, lr: float, weight_decay: float,
                        betas=(0.9, 0.95)) -> torch.optim.Optimizer:
    """embedding 与 norm 不做 weight decay。"""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "emb" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=False)


def save_checkpoint(path: str, model: torch.nn.Module, model_cfg_dict: dict,
                    step: int, optimizer=None, extra: dict = None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "model_cfg": model_cfg_dict,
        "step": step,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if extra:
        payload.update(extra)
    torch.save(payload, path)
    print(f"[ckpt] 已保存 {path} (step={step})")


def load_checkpoint(path: str, model: torch.nn.Module, device: str):
    """返回 (step, extra_dict)。只加载权重 (sft 初始化/生成用)。"""
    payload = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    return payload.get("step", 0), {k: v for k, v in payload.items()
                                    if k not in ("model", "model_cfg", "optimizer")}


def save_model_cfg(out_dir: str, model_cfg_dict: dict):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "model_config.json"), "w", encoding="utf-8") as f:
        json.dump(model_cfg_dict, f, indent=2, ensure_ascii=False)


def estimate_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"
