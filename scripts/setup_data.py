"""Dataset setup: freeze the GSM8K subset, warm the WikiText-2 cache.

Two jobs, and one deliberate non-job.

**Freeze the GSM8K subset.** The 250 questions are chosen once by seed and written to
``evals/gsm8k_subset_ids.json``, which is then committed. After that the file is the
authority and the seed is only provenance. Nothing else in the codebase will select
questions: the loaders raise if the file is missing rather than quietly picking their own,
because two instances scoring different exams produces a quality difference that is
entirely an artifact of the setup step.

Re-running without ``--force`` refuses. Reselecting the subset invalidates every quality
number already measured against the old list, so it has to be an explicit act.

**Warm the WikiText-2 cache.** Optional, and purely about not discovering a download
failure four hours into a sweep. It changes no measurement.

**Not a job: downloading model weights.** There is no code here that fetches a checkpoint
or a GGUF. Which weights a condition runs is the single most important fact about it, and
a setup script that silently pulled "whatever matches this name" is exactly how a study
ends up reporting numbers from a model nobody chose. Model paths go in the sweep YAML by
hand; see the notes printed at the end of a run.

Run:
    python -m scripts.setup_data --gsm8k                # the required step
    python -m scripts.setup_data --gsm8k --wikitext     # plus cache warming
    python -m scripts.setup_data --check                # report status, change nothing
    python -m scripts.setup_data --check-models configs/instance1_precision_spec.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _datasets_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("datasets") is not None


def check() -> int:
    """Report what exists. Touches nothing."""
    from evals.gsm8k import SUBSET_PATH

    print("dataset status")
    print("-" * 60)

    if SUBSET_PATH.exists():
        meta = json.loads(SUBSET_PATH.read_text(encoding="utf-8"))
        print(f"  GSM8K subset : PRESENT  {SUBSET_PATH}")
        print(f"                 {meta['n']} ids, split={meta['split']!r}, seed={meta['seed']}")
        print(f"                 first: {meta['ids'][0]}   last: {meta['ids'][-1]}")
    else:
        print(f"  GSM8K subset : MISSING  {SUBSET_PATH}")
        print("                 nothing can run until this exists:")
        print("                 python -m scripts.setup_data --gsm8k")

    print(f"  datasets pkg : {'installed' if _datasets_available() else 'NOT INSTALLED'}")
    return 0 if SUBSET_PATH.exists() else 1


def check_models(config_path: str) -> int:
    """Preflight every model a sweep config names. Downloads nothing.

    Worth its own step because the alternative is discovering a typo'd repo id or a
    missing GGUF four hours into a sweep, on the one condition that happened to run last.
    """
    from core.config import load_sweep

    cfg = load_sweep(config_path)
    print(f"model preflight: {config_path}")
    print(f"  stack={cfg.stack}  platform={cfg.platform}  {len(cfg.conditions)} conditions")
    print("-" * 68)

    ok = True
    if cfg.stack == "llamacpp":
        ok &= _check_llamacpp(cfg)
    else:
        ok &= _check_hf_repos(cfg)

    print("-" * 68)
    print("PREFLIGHT PASSED" if ok else "PREFLIGHT FAILED -- fix the above before sweeping")
    return 0 if ok else 1


def _check_hf_repos(cfg) -> bool:
    """Report each distinct checkpoint and whether it is already in the local cache."""
    import shutil

    repos: dict[str, set[str]] = {}
    for cond in cfg.conditions:
        repos.setdefault(cond.target_model, set()).add(f"target/{cond.target_dtype}")
        if cond.draft_model:
            repos.setdefault(cond.draft_model, set()).add("draft")

    try:
        from huggingface_hub import scan_cache_dir

        cached = {r.repo_id for r in scan_cache_dir().repos}
    except ImportError:
        print("  (huggingface_hub not installed -- cannot report cache state)")
        cached = set()
    except Exception:  # noqa: BLE001 -- an unreadable cache is not fatal to the report
        cached = set()

    ok = True
    for repo, roles in sorted(repos.items()):
        local = Path(repo).expanduser()
        if local.exists():
            state = "LOCAL PATH"
        elif repo in cached:
            state = "cached"
        else:
            state = "NOT CACHED (will download on first use)"
        print(f"  {state:38s} {repo}")
        print(f"  {'':38s}   roles: {', '.join(sorted(roles))}")

    print()
    print("  Gated repos (Meta Llama, Mistral) need a license accepted on the model page")
    print("  and `huggingface-cli login` on this machine, or the download 401s.")
    if shutil.which("huggingface-cli") is None:
        print("  NOTE: huggingface-cli not on PATH (pip install -U 'huggingface_hub[cli]')")

    # Speculative decoding needs a shared vocabulary between draft and target; a mismatch
    # is not a slowdown, it is a wrong answer.
    pairs = {(c.target_model, c.draft_model) for c in cfg.conditions if c.draft_model}
    if pairs:
        print()
        print("  Draft/target pairs (must share a tokenizer and vocabulary):")
        for target, draft in sorted(pairs):
            print(f"    {draft}\n      -> {target}")
    return ok


def _check_llamacpp(cfg) -> bool:
    import shutil

    model = cfg.raw.get("model") or {}
    ok = True

    binary = model.get("binary")
    resolved = shutil.which(binary) if binary else None
    if resolved or (binary and Path(binary).is_file()):
        print(f"  binary       OK        {resolved or binary}")
    else:
        print(f"  binary       MISSING   {binary!r} not on PATH and not a file")
        ok = False

    for key in ("target_gguf", "draft_gguf"):
        raw = model.get(key)
        if not raw:
            print(f"  {key:12s} MISSING   not set in config")
            ok = False
            continue
        path = Path(raw)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if path.is_file():
            size_gb = path.stat().st_size / 1e9
            print(f"  {key:12s} OK        {path}  ({size_gb:.1f} GB)")
        else:
            print(f"  {key:12s} MISSING   {path}")
            ok = False

    quants = {c.gguf_quant for c in cfg.conditions if c.gguf_quant}
    threads = {c.num_threads for c in cfg.conditions}
    print(f"\n  gguf_quant recorded: {sorted(quants) or 'NOT SET'}")
    print(f"  num_threads pinned : {sorted(t for t in threads if t)}")
    if None in threads:
        print("  WARNING: a condition has no num_threads; the CPU runner will refuse it.")
        ok = False
    print("\n  Draft and target must use the SAME quantization, and must share a")
    print("  tokenizer -- speculative decoding across different vocabularies is wrong,")
    print("  not merely slow.")
    return ok


def setup_gsm8k(*, n: int, seed: int, split: str, force: bool) -> None:
    from evals.gsm8k import SUBSET_PATH, build_subset, load_subset

    if SUBSET_PATH.exists() and not force:
        meta = json.loads(SUBSET_PATH.read_text(encoding="utf-8"))
        print(f"GSM8K subset already frozen: {meta['n']} ids, seed={meta['seed']}.")
        print("Leaving it alone. Re-selecting would invalidate every quality number")
        print("already measured against it; pass --force only if you intend that.")
        return

    print(f"selecting {n} GSM8K {split} examples with seed {seed} ...")
    ids = build_subset(n=n, seed=seed, split=split, force=force)
    print(f"wrote {SUBSET_PATH} ({len(ids)} ids)")

    # Load it straight back through the real loader: a subset that cannot be read by the
    # thing that will read it is not set up, and better to find out now.
    examples = load_subset()
    print(f"verified: {len(examples)} examples load, gold answers extracted")
    print(f"  e.g. {examples[0].prompt_id} -> gold {examples[0].answer!r}")
    print("\nCOMMIT THIS FILE. Every instance must score the same exam.")


def setup_wikitext(*, split: str) -> None:
    from evals.perplexity import PPLConfig, load_text

    cfg = PPLConfig(max_length=2048, stride=512, split=split)
    print(f"warming WikiText-2 cache ({cfg.dataset}/{cfg.subset}, split={split}) ...")
    text = load_text(cfg)
    print(f"cached: {len(text):,} characters")
    print("(cache warming only -- window and stride are set per-run in the eval config)")


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
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--gsm8k", action="store_true",
                    help="freeze the GSM8K subset (required before anything runs)")
    ap.add_argument("--wikitext", action="store_true",
                    help="pre-download WikiText-2 so a sweep does not stall on it")
    ap.add_argument("--check", action="store_true",
                    help="report what exists and exit, changing nothing")
    ap.add_argument("--check-models", metavar="CONFIG",
                    help="preflight every model a sweep config names; downloads nothing")
    ap.add_argument("--n", type=int, default=250, help="subset size (default 250)")
    ap.add_argument("--seed", type=int, default=0, help="selection seed (default 0)")
    ap.add_argument("--split", default="test", help="dataset split (default test)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing frozen subset -- invalidates measured quality")
    args = ap.parse_args(argv)

    if args.check_models:
        return check_models(args.check_models)

    if args.check:
        return check()

    if not (args.gsm8k or args.wikitext):
        ap.print_help()
        print("\nnothing to do: pass --gsm8k, --wikitext or --check")
        return 2

    if not _datasets_available():
        print(
            "the 'datasets' package is required and is not installed.\n"
            "  pip install datasets\n"
            "Refusing to substitute a local copy of unknown provenance.",
            file=sys.stderr,
        )
        return 1

    if args.gsm8k:
        setup_gsm8k(n=args.n, seed=args.seed, split=args.split, force=args.force)
    if args.wikitext:
        print()
        setup_wikitext(split=args.split)

    print("\n" + "-" * 60)
    print("Still to do by hand (deliberately not automated):")
    print("  * model checkpoints -- set target_model / draft_model in configs/*.yaml")
    print("  * GGUF files for the CPU arm -- set model.target_gguf / model.draft_gguf")
    print("  * llama.cpp speculative binary -- set model.binary")
    print("Which weights ran is the most important fact about a condition; it is chosen,")
    print("never inferred.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
