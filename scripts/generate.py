from __future__ import annotations

import argparse
from pathlib import Path

import torch

from minillm.model import GPTConfig, MiniGPT
from minillm.tokenizer import ByteTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/minigpt.pt")
    p.add_argument("--prompt", type=str, default="Once upon a time")
    p.add_argument("--tokens", type=int, default=120)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tok = ByteTokenizer()

    ckpt = torch.load(Path(args.ckpt), map_location=args.device)
    cfg = GPTConfig(**ckpt["config"])
    model = MiniGPT(cfg).to(args.device)
    model.load_state_dict(ckpt["state_dict"])

    ids = tok.encode(args.prompt)
    x = torch.tensor([ids], dtype=torch.long, device=args.device)
    y = model.generate(
        x,
        max_new_tokens=args.tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None,
    )
    out = tok.decode(y[0].tolist())
    print(out)


if __name__ == "__main__":
    main()
