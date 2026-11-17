"""加载 SFT 后的 checkpoint 进行中文对话生成。

用法:
    # 单轮
    python generate.py --ckpt checkpoints/sft_best.pt --prompt "你好，介绍一下你自己"
    # 多轮交互
    python generate.py --ckpt checkpoints/sft_best.pt --chat
参数: --temperature 0.8 --top_p 0.9 --top_k 50 --max_new_tokens 256
"""
import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import TOKENIZER_PATH, ModelConfig  # noqa: E402
from model import MiniChineseLLM  # noqa: E402
from train_utils import estimate_device  # noqa: E402

PROMPT_TEMPLATE = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"


def load_model(ckpt_path: str, device: str):
    payload = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**payload["model_cfg"])
    model = MiniChineseLLM(cfg).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model, cfg


def chat_once(model, tok, question: str, args, device) -> str:
    im_end = tok.token_to_id("<|im_end|>")
    prompt_ids = tok.encode(PROMPT_TEMPLATE.format(q=question)).ids
    idx = torch.tensor([prompt_ids], dtype=torch.int64, device=device)
    out = model.generate(
        idx, max_new_tokens=args.max_new_tokens,
        temperature=args.temperature, top_k=args.top_k, top_p=args.top_p,
        eos_id=im_end,
    )
    new_ids = out[0, len(prompt_ids):].tolist()
    if im_end in new_ids:
        new_ids = new_ids[: new_ids.index(im_end)]
    return tok.decode(new_ids)


def main():
    ap = argparse.ArgumentParser(description="对话生成")
    ap.add_argument("--ckpt", type=str, default="checkpoints/sft_best.pt")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--chat", action="store_true", help="多轮交互模式")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--top_k", type=int, default=50)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()
    device = args.device or estimate_device()

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOKENIZER_PATH)
    model, cfg = load_model(args.ckpt, device)
    print(f"已加载 {args.ckpt} (参数量 {sum(p.numel() for p in model.parameters()):,}, "
          f"上下文 {cfg.max_seq_len})")

    if args.prompt:
        print(f"\nUser: {args.prompt}\nAssistant: "
              f"{chat_once(model, tok, args.prompt, args, device)}")
        return

    history = ""
    print("进入交互模式 (输入 /quit 退出):")
    while True:
        try:
            q = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q or q == "/quit":
            break
        # 多轮: 把历史拼接进 user 内容尾部 (上下文有限, 仅做简单拼接)
        print(f"Assistant: {chat_once(model, tok, (history + q) if history else q, args, device)}")
        history = f"{history}{q} " if len(history) + len(q) < 800 else f"{q} "


if __name__ == "__main__":
    main()
