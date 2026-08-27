"""End-to-end smoke test: a tiny real sweep, all the way through figure generation.

**This is a pipeline test, not a runner test.** Its job is to prove that a record written
by a runner survives validation, loading, collapsing, bootstrapping and plotting without
anything in the chain disagreeing about the schema. It uses real models and real
measurements -- just very few of them.

It deliberately does *not* mock a runner. A mocked smoke test proves the plumbing accepts
mock data, which is the failure mode this whole codebase is built to prevent.

Run:
    python -m scripts.smoke --config configs/smoke.yaml
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from core.config import load_sweep
from scripts import build_report, run_sweep

REPO_ROOT = Path(__file__).resolve().parent.parent
EXPECTED_FIGURES = ["fig01", "fig02", "fig03", "fig04", "fig05", "fig06", "fig07", "fig08"]


def _tolerant_stdout() -> None:
    """Never let a console encoding kill a run.

    Windows consoles default to cp1252 and raise UnicodeEncodeError on characters the
    figures use freely. A benchmark sweep must not die four hours in because a status
    line contained a Greek letter.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def main(argv: list[str] | None = None) -> int:
    _tolerant_stdout()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/smoke.yaml")
    ap.add_argument("--outdir", default="report/smoke")
    ap.add_argument("--fresh", action="store_true", help="delete the smoke log first")
    ap.add_argument("--require", nargs="*", default=None,
                    help="figures that must render (default: those the config supports)")
    args = ap.parse_args(argv)

    cfg = load_sweep(args.config)
    log_path = cfg.log_path if cfg.log_path.is_absolute() else REPO_ROOT / cfg.log_path
    outdir = Path(args.outdir) if Path(args.outdir).is_absolute() else REPO_ROOT / args.outdir

    if args.fresh:
        log_path.unlink(missing_ok=True)
        if outdir.exists():
            shutil.rmtree(outdir)

    print("=" * 70)
    print("SMOKE 1/3 -- sweep")
    print("=" * 70)
    rc = run_sweep.main(["--config", args.config])
    if rc != 0:
        print("smoke failed: the sweep did not complete.", file=sys.stderr)
        return rc

    print("\n" + "=" * 70)
    print("SMOKE 2/3 -- report + figures")
    print("=" * 70)
    result = build_report.build([log_path], outdir, score_quality=True)

    print("\n" + "=" * 70)
    print("SMOKE 3/3 -- verification")
    print("=" * 70)
    required = args.require if args.require is not None else _supported(cfg)
    missing = []
    for name in EXPECTED_FIGURES:
        outcome = result["figures"].get(name)
        rendered = isinstance(outcome, Path) or (
            isinstance(outcome, str) and not outcome.startswith("SKIPPED")
        )
        status = "ok" if rendered else str(outcome)
        print(f"  {name}: {status}")
        if name in required and not rendered:
            missing.append(name)

    if missing:
        print(f"\nsmoke FAILED: required figure(s) did not render: {missing}", file=sys.stderr)
        return 1

    print(f"\nsmoke passed: {result['n_records']} real records -> {outdir}")
    return 0


def _supported(cfg) -> list[str]:
    """Which figures this config can actually produce.

    A smoke config with one batch size cannot make figure 03, and demanding it would
    make the smoke test fail for a reason that is not a bug. Figures whose inputs the
    config does not sweep are not required -- but they are still attempted and their
    skip reason is printed.
    """
    dtypes = {c.target_dtype for c in cfg.conditions}
    batches = {c.batch_size for c in cfg.conditions}
    gammas = {c.num_speculative_tokens for c in cfg.conditions if c.num_speculative_tokens}
    models = {c.target_model for c in cfg.conditions}
    has_spec = any(c.spec_method != "none" for c in cfg.conditions)

    required = []
    if has_spec:
        required += ["fig02", "fig06"]
    if has_spec and len(batches) > 1:
        required.append("fig03")
    if has_spec and len(gammas) > 1:
        required.append("fig04")
    if len(dtypes) > 1:
        required.append("fig07")
    if has_spec and len(models) > 1:
        required.append("fig08")
    required.append("fig01")
    return required


if __name__ == "__main__":
    raise SystemExit(main())
