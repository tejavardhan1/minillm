from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tqdm import trange

from minillm.data import get_batch, load_text, make_dataset
from minillm.model import GPTConfig, MiniGPT
from minillm.tokenizer import ByteTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/tiny_corpus.txt")
    p.add_argument("--out", type=str, default="checkpoints/minigpt.pt")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--block-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=200)
    return p.parse_args()


@torch.no_grad()
def estimate_loss(model: MiniGPT, ds, *, device: str, block_size: int, batch_size: int, iters: int = 25) -> dict[str, float]:
    out = {}
    model.eval()
    for split_name, data in [("train", ds.train), ("val", ds.val)]:
        losses = []
        for _ in range(iters):
            x, y = get_batch(data, block_size=block_size, batch_size=batch_size, device=device)
            _, loss = model(x, y)
            losses.append(float(loss.item()))
        out[split_name] = sum(losses) / len(losses)
    model.train()
    return out


def main() -> None:
    args = parse_args()
    device = args.device
    torch.manual_seed(1337)

    tok = ByteTokenizer()
    text = load_text(Path(args.data))
    ds = make_dataset(text, tokenizer=tok)

    block_size = min(args.block_size, len(ds.train) - 2, len(ds.val) - 2)
    if block_size < 8:
        raise ValueError(f"Dataset too small for training: need at least 10 tokens per split, got train={len(ds.train)}, val={len(ds.val)}")
    cfg = GPTConfig(vocab_size=tok.vocab_size, block_size=block_size)
    model = MiniGPT(cfg).to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pbar = trange(args.steps, desc="training")
    for step in pbar:
        x, y = get_batch(ds.train, block_size=block_size, batch_size=args.batch_size, device=device)
        _, loss = model(x, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if (step + 1) % args.eval_every == 0 or step == 0:
            losses = estimate_loss(model, ds, device=device, block_size=block_size, batch_size=args.batch_size)
            pbar.write(f"step {step+1}: train {losses['train']:.4f}  val {losses['val']:.4f}")

    torch.save({"config": cfg.__dict__, "state_dict": model.state_dict()}, out_path)
    print(f"Saved checkpoint to {out_path}")


if __name__ == "__main__":
    main()
