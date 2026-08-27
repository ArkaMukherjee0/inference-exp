"""Estimate a sweep's wall clock, and size one to fit a time budget.

Written because "will this finish tonight?" is a question you must answer *before*
starting, and because the honest answer depends on numbers only your machine knows.

Two modes:

* ``--from-log`` derives the per-unit cost from measurements you already have. This is
  the accurate one: it reads real tpot and real prompt lengths out of a JSONL log.
* ``--tok-s`` takes a decode rate you supply, for when no log exists yet.

Neither invents anything. With no log and no rate, it refuses rather than guessing --
a runtime estimate presented with false confidence is how people start 30-hour sweeps
believing they are 8-hour ones.

    python -m scripts.estimate_runtime --config configs/local_cpu.yaml --from-log logs/local-cpu-smoke.jsonl
    python -m scripts.estimate_runtime --config configs/instance1_precision_spec.yaml --tok-s 60 --budget-hours 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from core.config import SweepConfig, load_sweep

REPO_ROOT = Path(__file__).resolve().parent.parent

# Per condition-visit: model load, compile, CUDA graph capture. Paid once per block on
# the GPU arm; the llama.cpp arm reloads per prompt instead and that is folded into the
# per-unit cost below.
DEFAULT_STARTUP_S = {"h100": 75.0, "cpu": 0.0}
# llama.cpp spawns a fresh process per prompt, so every unit pays a model load.
DEFAULT_PER_UNIT_OVERHEAD_S = {"h100": 0.3, "cpu": 3.0}


def _tolerant_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def rate_from_log(path: Path) -> tuple[float, float]:
    """(decode tokens/sec, mean prefill seconds) from a real log.

    Uses the median tpot across measured records, so one slow outlier does not set the
    estimate for a whole sweep.
    """
    import numpy as np

    from analysis.load import load_runs

    df = load_runs(path)
    tpot_ms = float(np.median(df["tpot_ms"].to_numpy(dtype=float)))
    ttft_ms = float(np.median(df["ttft_ms"].to_numpy(dtype=float)))
    if tpot_ms <= 0:
        raise ValueError(f"{path}: median tpot_ms is not positive")
    return 1000.0 / tpot_ms, ttft_ms / 1000.0


def estimate(
    cfg: SweepConfig,
    *,
    tok_s: float,
    prefill_s: float,
    startup_s: float | None = None,
    per_unit_overhead_s: float | None = None,
) -> dict[str, float]:
    startup_s = DEFAULT_STARTUP_S[cfg.platform] if startup_s is None else startup_s
    overhead = (
        DEFAULT_PER_UNIT_OVERHEAD_S[cfg.platform]
        if per_unit_overhead_s is None else per_unit_overhead_s
    )

    ids = [f"gsm8k-test-{i}" for i in range(cfg.prompts.n)]
    queue = cfg.build_queue(ids)
    blocks = sum(1 for _ in cfg.condition_visits(queue))

    # Every unit: prefill, then max_tokens decoded, plus fixed overhead. The GPU arm
    # additionally pays a second prefill for the matched TTFT request.
    max_tokens = max(c.max_tokens for c in cfg.conditions)
    ttft_extra = prefill_s if cfg.platform != "cpu" else 0.0
    per_unit = prefill_s + ttft_extra + max_tokens / tok_s + overhead

    unit_s = len(queue) * per_unit
    startup_total_s = blocks * startup_s
    return {
        "units": len(queue),
        "blocks": blocks,
        "per_unit_s": per_unit,
        "generation_h": unit_s / 3600.0,
        "startup_h": startup_total_s / 3600.0,
        "total_h": (unit_s + startup_total_s) / 3600.0,
    }


def _fmt(cfg: SweepConfig, est: dict[str, float], label: str) -> str:
    return (
        f"  {label:34s} {est['units']:6.0f} units, {est['blocks']:4.0f} blocks"
        f"  ->  {est['total_h']:5.2f} h"
        f"  (gen {est['generation_h']:.2f} + startup {est['startup_h']:.2f})"
    )


def suggest(cfg: SweepConfig, budget_h: float, *, tok_s: float, prefill_s: float) -> None:
    """Show what fits the budget, cheapest-information-loss first.

    The order matters and is not arbitrary. Repeats guard against a hiccup becoming a
    data point; they do NOT narrow confidence intervals -- prompts do. So repeats are cut
    first and prompt count last, because prompt count is the one that costs statistical
    power. max_tokens is never suggested: it is a fixed measurement constraint, and
    shortening it raises the share of each run spent in prefill.
    """
    import copy

    base = estimate(cfg, tok_s=tok_s, prefill_s=prefill_s)
    print(f"\nbudget: {budget_h:.1f} h")
    print(_fmt(cfg, base, "as configured"))
    if base["total_h"] <= budget_h:
        print("\n  Fits as-is.")
        return

    print("\n  Does not fit. Best-fitting options:\n")
    max_repeats = max(c.repeats for c in cfg.conditions)
    gammas = sorted({c.num_speculative_tokens for c in cfg.conditions
                     if c.num_speculative_tokens})

    # Enumerate everything that fits, then rank by what we are least willing to give up.
    # Prompt count comes first because it is the only one of the three that sets the
    # width of a confidence interval; repeats second (3 still absorbs a hiccup); the
    # gamma axis last, since dropping interior gamma points costs curve resolution in
    # figure 04 but no precision anywhere else.
    fitting = []
    for n in (cfg.prompts.n, 250, 200, 150, 120, 100, 80, 60, 50):
        if n > cfg.prompts.n:
            continue
        for repeats in range(max_repeats, 0, -1):
            for keep_gamma in range(len(gammas), 0, -1):
                trial = _variant(cfg, repeats=repeats, n=n, keep_gamma=keep_gamma,
                                 gammas=gammas)
                est = estimate(trial, tok_s=tok_s, prefill_s=prefill_s)
                if est["total_h"] <= budget_h:
                    fitting.append((n, repeats, keep_gamma, trial, est))

    if not fitting:
        print("  Nothing in the suggested grid fits. Cut conditions, not measurements:")
        print("  drop an axis (a precision level, the TP arm) rather than shrinking n")
        print("  below ~50, where the intervals stop being worth reporting.")
        return

    # Two hard floors before ranking, because below either one an *experiment* stops
    # existing rather than merely getting noisier:
    #
    #   * gamma points -- a single gamma cannot show a turnover, so figure 04 (model vs
    #     measured) and the optimal-gamma result simply do not happen. Four points is the
    #     minimum that shows a rise and a fall.
    #   * repeats -- with fewer than 3 there is nothing to absorb one bad run, and the
    #     median over repeats degenerates.
    #
    # Only once both are satisfied does prompt count get maximized, because n is what
    # sets interval width.
    min_gammas = min(4, len(gammas)) if gammas else 0
    min_repeats = min(3, max_repeats)

    viable = [r for r in fitting if r[2] >= min_gammas and r[1] >= min_repeats]
    relaxed = False
    if not viable:
        relaxed = True
        viable = fitting

    viable.sort(key=lambda row: (row[0], row[1], row[2]), reverse=True)
    seen: set[tuple[int, int]] = set()
    shown = 0
    for n, repeats, keep_gamma, trial, est in viable:
        key = (n, keep_gamma)
        if key in seen:
            continue
        seen.add(key)
        print(_fmt(trial, est, f"n={n}, repeats={repeats}, gamma={gammas[:keep_gamma]}"))
        shown += 1
        if shown >= 5:
            break

    best = viable[0]
    print(f"\n  Recommended: n={best[0]}, repeats={best[1]}, "
          f"gamma={gammas[:best[2]]}  ({best[4]['total_h']:.2f} h)")
    if relaxed:
        print(f"  WARNING: nothing met the floors (>={min_gammas} gamma points, "
              f">={min_repeats} repeats).")
        print("  What is shown sacrifices an experiment, not just precision. Prefer")
        print("  dropping a whole axis -- a precision level, the TP arm -- and keeping")
        print("  the gamma sweep intact on what remains.")
    if best[0] < 100:
        print("  NOTE: below ~100 prompts the paired intervals widen enough that a small")
        print("        composition effect may not clear 1.0. Report intervals, not point")
        print("        estimates alone.")


def _variant(cfg: SweepConfig, *, repeats: int, n: int, keep_gamma: int, gammas: list[int]) -> SweepConfig:
    """A copy of the config with repeats/prompt-count/gamma-axis trimmed."""
    import dataclasses

    keep = set(gammas[:keep_gamma]) if keep_gamma < len(gammas) else set(gammas)
    conditions = tuple(
        dataclasses.replace(c, repeats=repeats)
        for c in cfg.conditions
        if c.num_speculative_tokens is None or c.num_speculative_tokens in keep
    )
    prompts = dataclasses.replace(cfg.prompts, n=n)
    return dataclasses.replace(cfg, conditions=conditions, prompts=prompts)


def main(argv: list[str] | None = None) -> int:
    _tolerant_stdout()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--from-log", help="derive tok/s and prefill from a real log")
    ap.add_argument("--tok-s", type=float, help="decode tokens/sec, if no log exists yet")
    ap.add_argument("--prefill-s", type=float, default=None, help="seconds of prefill per prompt")
    ap.add_argument("--startup-s", type=float, default=None, help="per-condition engine startup")
    ap.add_argument("--budget-hours", type=float, default=None)
    args = ap.parse_args(argv)

    cfg = load_sweep(args.config)

    if args.from_log:
        tok_s, prefill_s = rate_from_log(Path(args.from_log))
        source = f"measured from {args.from_log}"
    elif args.tok_s:
        tok_s = args.tok_s
        prefill_s = args.prefill_s if args.prefill_s is not None else 0.5
        source = "supplied on the command line"
    else:
        print(
            "need either --from-log (accurate) or --tok-s (estimate).\n"
            "Refusing to invent a decode rate: a runtime estimate with a made-up input "
            "is how a 30-hour sweep gets started as an 8-hour one.",
            file=sys.stderr,
        )
        return 2

    if args.prefill_s is not None:
        prefill_s = args.prefill_s

    print(f"config   : {args.config}")
    print(f"platform : {cfg.platform} / {cfg.stack}")
    print(f"rate     : {tok_s:.1f} tok/s decode, {prefill_s * 1000:.0f} ms prefill ({source})")
    print()
    est = estimate(cfg, tok_s=tok_s, prefill_s=prefill_s, startup_s=args.startup_s)
    print(_fmt(cfg, est, "as configured"))

    if args.budget_hours:
        suggest(cfg, args.budget_hours, tok_s=tok_s, prefill_s=prefill_s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
