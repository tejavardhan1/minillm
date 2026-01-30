from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ByteTokenizer:
    """
    Minimal tokenizer: bytes 0..255 plus 1 special token (eos=256).
    Pros: no training required, works on any text.
    Cons: longer sequences than subword tokenizers.
    """

    eos_id: int = 256

    @property
    def vocab_size(self) -> int:
        return 257

    def encode(self, text: str) -> list[int]:
        b = text.encode("utf-8", errors="ignore")
        return list(b)

    def decode(self, ids: list[int]) -> str:
        b = bytes([i for i in ids if 0 <= i <= 255])
        return b.decode("utf-8", errors="ignore")
