"""Find where THIS vLLM build exposes speculative acceptance counts.

vLLM has moved acceptance reporting repeatedly, and ``runners/vllm_runner.py`` refuses to
guess: an acceptance rate inferred from the wrong field is a number that looks fine and
is wrong, and every gamma conclusion in the study rests on it. So instead of probing
blindly, this script runs one tiny speculative request against the installed build and
reports every place acceptance-shaped data actually turns up.

It changes no measurement and writes no record. Run it once per box, paste the output
into the runner's ``_acceptance``, and the GPU arm is unblocked.

    python -m scripts.probe_vllm_acceptance \\
        --model meta-llama/Llama-3.2-3B-Instruct \\
        --draft meta-llama/Llama-3.2-1B-Instruct
"""

from __future__ import annotations

import argparse
import sys

KEYWORDS = ("accept", "spec", "draft", "num_drafts", "reject")
# Attributes whose repr is huge or whose access has side effects.
SKIP = {"prompt_token_ids", "token_ids", "logprobs", "prompt_logprobs", "embeddings",
        "multi_modal_data", "prompt", "text", "encoder_prompt", "lora_request"}


def _tolerant_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            pass


def _interesting(name: str) -> bool:
    low = name.lower()
    return any(k in low for k in KEYWORDS) and not low.startswith("__")


def _describe(value: object, limit: int = 220) -> str:
    try:
        text = repr(value)
    except Exception as exc:  # noqa: BLE001 -- probing unknown objects
        return f"<repr failed: {exc}>"
    return text if len(text) <= limit else text[:limit] + " ..."


def walk(obj: object, label: str, depth: int = 0, seen: set[int] | None = None) -> int:
    """Print every acceptance-shaped attribute reachable from obj. Returns hit count."""
    seen = seen if seen is not None else set()
    if obj is None or id(obj) in seen or depth > 6:
        return 0
    seen.add(id(obj))

    hits = 0
    for name in sorted(dir(obj)):
        if name.startswith("_") or name in SKIP:
            continue
        try:
            value = getattr(obj, name)
        except Exception:  # noqa: BLE001 -- properties can raise on partial state
            continue
        if callable(value):
            continue

        if _interesting(name):
            print(f"  [HIT] {label}.{name} = {_describe(value)}")
            hits += 1
            # Keep descending: a matched name is often a container whose *contents* are
            # the counts. Stopping here hid SpecDecodingLogging's internals.
            if hasattr(value, "__dict__"):
                hits += walk(value, f"{label}.{name}", depth + 1, seen)
        # Descend into containers too. vLLM V1 keeps stat loggers in a dict, and the
        # EngineCore process boundary means those loggers are the only thing in this
        # process that ever sees acceptance counts.
        elif depth < 6:
            if isinstance(value, dict):
                for key, item in list(value.items())[:8]:
                    hits += walk(item, f"{label}.{name}[{key!r}]", depth + 1, seen)
            elif isinstance(value, (list, tuple)) and not isinstance(value, (str, bytes)):
                for i, item in enumerate(list(value)[:8]):
                    if not isinstance(item, (str, bytes, int, float, bool)):
                        hits += walk(item, f"{label}.{name}[{i}]", depth + 1, seen)
            elif hasattr(value, "__dict__"):
                hits += walk(value, f"{label}.{name}", depth + 1, seen)
    return hits


def main(argv: list[str] | None = None) -> int:
    _tolerant_stdout()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", required=True)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    args = ap.parse_args(argv)

    import os

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    from vllm import LLM, SamplingParams
    from vllm.version import __version__ as vllm_version

    print(f"vLLM {vllm_version}")
    print(f"target={args.model}  draft={args.draft}  gamma={args.gamma}\n")

    llm = LLM(
        model=args.model,
        speculative_config={
            "model": args.draft,
            "num_speculative_tokens": args.gamma,
        },
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_prefix_caching=False,
        disable_log_stats=False,
        seed=42,
    )

    # A prompt with obvious repetition, so acceptance is high and the counts are
    # unmistakable rather than a row of zeros.
    prompt = "Count: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23"
    outputs = llm.generate(
        [prompt],
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens, ignore_eos=True),
        use_tqdm=False,
    )
    out = outputs[0]
    emitted = len(out.outputs[0].token_ids)
    print(f"generated {emitted} tokens\n")

    total = 0
    print("=" * 70)
    print("RequestOutput")
    print("=" * 70)
    total += walk(out, "out")
    total += walk(getattr(out, "metrics", None), "out.metrics")
    total += walk(out.outputs[0], "out.outputs[0]")

    print()
    print("=" * 70)
    print("Engine internals")
    print("=" * 70)
    engine = getattr(llm, "llm_engine", None)
    for attr in ("logger_manager", "stat_loggers", "engine_core", "output_processor"):
        total += walk(getattr(engine, attr, None), f"llm_engine.{attr}")
    total += walk(engine, "llm_engine", depth=2)

    print()
    print("=" * 70)
    if total:
        print(f"{total} candidate field(s) found.")
        print("Wire the one holding per-position or per-step counts into")
        print("runners/vllm_runner.py::_acceptance. What is needed is either:")
        print("  * per-position counts (non-increasing list, length gamma), or")
        print("  * per-step accepted counts,")
        print("plus enough to reconcile with output_tokens. Aggregate totals alone are")
        print("NOT enough for figure 06 -- a distribution synthesised from its own mean")
        print("is fabricated data, which is why the runner raises instead.")
    else:
        print("Nothing found. This build may expose acceptance only through the")
        print("Prometheus metrics path (vllm:spec_decode_*). In that case the options")
        print("are a custom StatLogger, or running the speculative arm on a build that")
        print("reports per-request counts. Do not estimate acceptance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
