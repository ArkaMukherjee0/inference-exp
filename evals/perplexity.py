"""WikiText-2 sliding-window perplexity.

Perplexity is the sensitive instrument in this study: it moves under quantization well
before GSM8K exact match does, and figure 07 is built on the two disagreeing.

Window and stride are explicit parameters recorded in the config, never defaults chosen
inside this module. Perplexity is not comparable across different strides -- a shorter
stride gives every token more left-context and lowers the number -- so a stride that
drifted between conditions would produce a quality difference that is purely an artifact
of the measurement.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class PPLConfig:
    max_length: int
    stride: int
    split: str = "test"
    dataset: str = "wikitext"
    subset: str = "wikitext-2-raw-v1"

    def __post_init__(self) -> None:
        if self.stride < 1:
            raise ValueError(f"stride must be >= 1 (got {self.stride})")
        if self.max_length < 2:
            raise ValueError(f"max_length must be >= 2 (got {self.max_length})")
        if self.stride > self.max_length:
            raise ValueError(
                f"stride ({self.stride}) > max_length ({self.max_length}): windows would "
                "skip tokens, so the reported perplexity would cover only part of the text."
            )


@dataclass(frozen=True)
class PPLResult:
    perplexity: float
    mean_nll: float
    n_tokens: int
    n_windows: int
    config: dict[str, Any]
    # Per-window NLL, kept so quality deltas can be paired window-by-window the same way
    # GSM8K deltas are paired example-by-example.
    window_nll: list[float]

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")


def load_text(cfg: PPLConfig) -> str:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("the 'datasets' package is required to load WikiText-2.") from exc
    ds = load_dataset(cfg.dataset, cfg.subset, split=cfg.split)
    return "\n\n".join(ds["text"])


def compute_perplexity(model: Any, tokenizer: Any, cfg: PPLConfig, text: str | None = None) -> PPLResult:
    """Sliding-window perplexity over WikiText-2.

    The standard windowed estimator: slide a window of ``max_length`` tokens forward by
    ``stride``, and score only the ``stride`` newly-revealed tokens in each window by
    masking the rest of the target. Scoring the whole window every time would count most
    tokens repeatedly, with more left-context each time, and report a number that is
    lower than the model deserves.
    """
    import torch

    if text is None:
        text = load_text(cfg)

    encodings = tokenizer(text, return_tensors="pt")
    input_ids_all = encodings.input_ids
    seq_len = int(input_ids_all.size(1))
    if seq_len < cfg.max_length:
        raise ValueError(
            f"corpus is {seq_len} tokens but max_length is {cfg.max_length}; a single "
            "window would not fill. Refusing to silently shrink the window."
        )

    device = next(model.parameters()).device
    nlls: list[float] = []
    counts: list[int] = []
    prev_end = 0

    model.eval()
    for begin in range(0, seq_len - cfg.max_length + 1, cfg.stride):
        end = begin + cfg.max_length
        trg_len = end - prev_end  # newly revealed tokens
        input_ids = input_ids_all[:, begin:end].to(device)
        targets = input_ids.clone()
        targets[:, :-trg_len] = -100  # mask everything but the new tokens

        with torch.no_grad():
            out = model(input_ids, labels=targets)

        # HF averages the loss over unmasked target positions; the count of scored
        # positions is trg_len - 1 because the first target has no preceding context
        # inside the masked region.
        n_scored = max(int(trg_len) - 1, 1)
        nlls.append(float(out.loss) * n_scored)
        counts.append(n_scored)
        prev_end = end

    if not nlls:
        raise ValueError("no windows were scored; check max_length and stride against the corpus")

    total_nll = float(np.sum(nlls))
    total_tokens = int(np.sum(counts))
    mean_nll = total_nll / total_tokens
    return PPLResult(
        perplexity=float(np.exp(mean_nll)),
        mean_nll=mean_nll,
        n_tokens=total_tokens,
        n_windows=len(nlls),
        config=asdict(cfg),
        window_nll=[n / c for n, c in zip(nlls, counts)],
    )


def paired_window_nll(base: PPLResult, opt: PPLResult) -> tuple[np.ndarray, np.ndarray]:
    """Align two perplexity results window-by-window for a paired delta.

    Raises unless both were computed with the identical window/stride configuration --
    perplexity is not comparable across strides, and a paired test across two different
    estimators would be meaningless.
    """
    if base.config != opt.config:
        raise ValueError(
            "perplexity configs differ, so the two numbers are not comparable:\n"
            f"  base: {base.config}\n  opt:  {opt.config}"
        )
    if len(base.window_nll) != len(opt.window_nll):
        raise ValueError(
            f"window counts differ ({len(base.window_nll)} vs {len(opt.window_nll)}) "
            "despite identical configs; the corpora must differ."
        )
    return np.asarray(base.window_nll, dtype=float), np.asarray(opt.window_nll, dtype=float)
