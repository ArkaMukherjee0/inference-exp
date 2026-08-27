"""Compute WikiText-2 perplexity for one checkpoint, and write it as JSON.

Figure 07 asks whether the precision axis moves perplexity and GSM8K exact-match in the
same direction. That needs a PPL number per precision, measured on the *same* windows,
and nothing else in the repo produces one -- ``evals/perplexity.py`` has the estimator
but no way to run it.

Run once per precision, with identical window settings, then hand the results to the
report:

    for m in meta-llama/Llama-3.1-8B-Instruct \\
             RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8 \\
             RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16 ; do
        python -m scripts.run_perplexity --model "$m" --out logs/ppl-$(basename "$m").json
    done

    python -m scripts.build_report --logs logs/instance-1.jsonl \\
        --perplexity bf16=logs/ppl-Llama-3.1-8B-Instruct.json \\
                     fp8=logs/ppl-Meta-Llama-3.1-8B-Instruct-FP8.json \\
                     w4a16=logs/ppl-Meta-Llama-3.1-8B-Instruct-quantized.w4a16.json

This is a quality measurement, not a latency one. It runs on HF transformers, and the
timings it happens to produce are not recorded anywhere -- HF Python overhead must never
reach a speed figure (§WP4).

The window settings are recorded inside every result file, and figure 07 refuses to take
deltas between results whose configs differ: perplexity is only comparable window for
window, and two checkpoints scored under different strides are two different numbers.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from evals.perplexity import PPLConfig, compute_perplexity, load_text

REPO_ROOT = Path(__file__).resolve().parent.parent


def _tolerant_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    _tolerant_stdout()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF repo id or local path")
    ap.add_argument("--out", required=True, help="where to write the PPLResult JSON")
    ap.add_argument("--max-length", type=int, default=2048)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--split", default="test")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="auto",
                    help="torch dtype for loading; 'auto' honours the checkpoint")
    args = ap.parse_args(argv)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = PPLConfig(max_length=args.max_length, stride=args.stride, split=args.split)

    print(f"model   : {args.model}")
    print(f"windows : max_length={cfg.max_length} stride={cfg.stride} split={cfg.split}")
    print(f"corpus  : {cfg.dataset}/{cfg.subset}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=dtype, device_map=args.device
    )
    model.eval()

    text = load_text(cfg)
    result = compute_perplexity(model, tokenizer, cfg, text=text)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_json(out)

    print(f"\nperplexity : {result.perplexity:.4f}")
    print(f"mean NLL   : {result.mean_nll:.6f}")
    print(f"tokens     : {result.n_tokens} over {result.n_windows} windows")
    print(f"written    : {out}")
    print("\nPass this to build_report as --perplexity <target_dtype>=%s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
