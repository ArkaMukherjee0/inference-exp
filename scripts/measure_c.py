"""Measure c -- the draft/target per-step cost ratio -- for figure 04.

``analysis.model.expected_speedup`` needs c, and figure 04 refuses to draw its predicted
curve without one. Nothing in the repo produced it, so the GPU arm could measure the
speedup but never the prediction it was supposed to be tested against.

c is measured, never derived. Two things it is deliberately NOT:

* a parameter-count ratio. A 1B drafting for an 8B is nowhere near 8x cheaper per step at
  batch 1, because kernel launches, sampling and scheduler work do not shrink with the
  model and come to dominate for small ones. See ``analysis.model.measure_c``.
* anything read off the speculative runs themselves. Deriving c from observed speedup
  would make the predicted curve fit by construction, and figure 04 exists precisely to
  test whether it fits.

So each model is timed **alone**, speculation off, batch 1, through the same VLLMRunner
the sweep uses -- same submission overhead, same sampler, same fences on both sides of
the ratio. The drafter is timed once; every target is timed separately, because c is a
ratio against *that* target and a w4a16 target does not have a bf16 target's step time.

    python -m scripts.measure_c \\
        --draft meta-llama/Llama-3.2-1B-Instruct \\
        --target meta-llama/Llama-3.1-8B-Instruct=bf16 \\
        --target RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8=fp8 \\
        --target RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16=w4a16 \\
        --out logs/measured_c.json

Needs the GPU to itself: run_sweep asserts GPU exclusivity before every condition block,
so running this beside a live sweep would abort the sweep.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path
from typing import Any

from core.schema import RunConfig

REPO_ROOT = Path(__file__).resolve().parent.parent


def _tolerant_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def _prompts(n: int) -> list[str]:
    from evals.gsm8k import load_subset

    examples = load_subset()
    if len(examples) < n:
        raise ValueError(f"frozen subset holds {len(examples)} prompts, asked for {n}")
    return [e.prompt for e in examples[:n]]


def step_time_ms(model: str, dtype: str, prompts: list[str], *, max_tokens: int,
                 gpu_memory_utilization: float, warmup: int) -> dict[str, Any]:
    """Median per-token decode time for one model, alone, batch 1, speculation off."""
    from runners.vllm_runner import VLLMRunner

    cfg = RunConfig(
        target_model=model,
        target_dtype=dtype,
        draft_model=None,
        spec_method="none",
        num_speculative_tokens=None,
        stack="vllm",
        platform="h100",
        batch_size=1,
        max_tokens=max_tokens,
        ignore_eos=True,
        temperature=0.0,
        seed=42,
        gpu_memory_utilization=gpu_memory_utilization,
    )
    runner = VLLMRunner(cfg)
    runner.setup()
    try:
        # Warmup generations are discarded, exactly as the sweep discards them: the first
        # requests through a fresh engine pay JIT and allocator costs that are not part of
        # the steady-state step time this ratio is about.
        for p in prompts[:warmup]:
            runner.generate(p, [])
        results = [runner.generate(p, []) for p in prompts]
    finally:
        runner.close()

    # tpot, by the same definition core/schema.py validates every record against.
    values = [(r.total_ms - r.ttft_ms) / (r.output_tokens - 1) for r in results]
    return {
        "model": model,
        "target_dtype": dtype,
        "tpot_ms_median": statistics.median(values),
        "tpot_ms_mean": statistics.fmean(values),
        "n_prompts": len(values),
        "max_tokens": max_tokens,
        "resolved_model_path": runner.resolved.get("model_path"),
    }


def main(argv: list[str] | None = None) -> int:
    _tolerant_stdout()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draft", required=True, help="draft repo id (timed once)")
    ap.add_argument("--draft-dtype", default="bf16")
    ap.add_argument("--target", action="append", required=True, metavar="MODEL=DTYPE",
                    help="repeatable; dtype is explicit, never inferred from the name")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-prompts", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = ap.parse_args(argv)

    from analysis.model import measure_c

    targets = []
    for spec in args.target:
        if "=" not in spec:
            raise SystemExit(f"--target must be MODEL=DTYPE (got {spec!r})")
        model, dtype = spec.rsplit("=", 1)
        targets.append((model, dtype))

    prompts = _prompts(args.n_prompts)
    common = dict(max_tokens=args.max_tokens, warmup=args.warmup,
                  gpu_memory_utilization=args.gpu_memory_utilization)

    print(f"prompts: {len(prompts)} (frozen subset prefix), max_tokens={args.max_tokens}\n")

    print(f"== draft: {args.draft} ({args.draft_dtype})")
    draft = step_time_ms(args.draft, args.draft_dtype, prompts, **common)
    print(f"   median tpot {draft['tpot_ms_median']:.4f} ms\n")

    results, c_by_pair = [], {}
    for model, dtype in targets:
        print(f"== target: {model} ({dtype})")
        t = step_time_ms(model, dtype, prompts, **common)
        c = measure_c(draft["tpot_ms_median"], t["tpot_ms_median"])
        t["c"] = c
        t["draft_model"] = args.draft
        results.append(t)
        c_by_pair[f"{model}|{args.draft}"] = c
        print(f"   median tpot {t['tpot_ms_median']:.4f} ms   ->  c = {c:.4f}\n")

    payload = {"draft": draft, "targets": results, "c_by_pair": c_by_pair,
               "method": "isolated batch-1 decode, speculation off, VLLMRunner"}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"written: {out}")
    return 0


def load_c_by_pair(path: str | Path) -> dict[tuple[str, str], float]:
    """Read a measure_c result into figure 04's ``c_by_pair`` keying."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return {tuple(k.split("|", 1)): float(v) for k, v in payload["c_by_pair"].items()}


if __name__ == "__main__":
    raise SystemExit(main())
