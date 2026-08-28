"""Sweep driver.

Expands a YAML config into an interleaved queue and executes it, one subprocess per
(round, condition) block. Two structural decisions:

**Progress is derived from the log, not from a separate checkpoint file.** The log is
the only thing that matters, so it is also the only thing consulted: on restart the
driver reads back every ``(condition_id, prompt_id, repeat_idx, is_warmup)`` already
recorded and skips those units. A checkpoint file can disagree with the log; the log
cannot disagree with itself. Combined with uuid4 run ids, killing the sweep and
restarting produces no duplicate rows and no gaps.

**Each block is a fresh subprocess.** Not just a fresh engine object -- a fresh process,
so CUDA context, allocator state, autotuning caches and any leaked memory die with it.
A crash inside one condition also cannot take the sweep down with it: the driver records
the failure and moves on, leaving that cell missing, which is the correct outcome.

Run:
    python -m scripts.run_sweep --config configs/instance1_precision_spec.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from core.config import SweepConfig, WorkUnit, load_sweep
from core.env import assert_gpu_exclusive, capture_env, utc_now
from core.schema import RunConfig, append_record, read_records
from runners.base import RunnerError, build_record

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------------------
# Prompt loading
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptSet:
    ids: list[str]
    text: dict[str, str]

    def filler_pool(self, exclude: str) -> list[str]:
        """Distinct prompts for padding a batch, deterministic in order."""
        return [self.text[i] for i in self.ids if i != exclude]


def load_prompts(cfg: SweepConfig) -> PromptSet:
    if cfg.prompts.source != "gsm8k":
        raise ValueError(
            f"prompt source {cfg.prompts.source!r} is not implemented. Add a loader "
            "rather than substituting a different corpus."
        )
    from evals.gsm8k import load_subset

    examples = load_subset()
    if len(examples) != cfg.prompts.n:
        if cfg.prompts.n > len(examples):
            raise ValueError(
                f"config asks for {cfg.prompts.n} prompts but the frozen subset holds "
                f"only {len(examples)}. Re-freeze the subset deliberately, or lower n."
            )
        if not cfg.prompts.allow_partial:
            raise ValueError(
                f"frozen subset has {len(examples)} examples but config asks for "
                f"{cfg.prompts.n}. The exam and the config disagree.\n"
                "If that is deliberate (a smoke or debug run), set "
                "prompts.allow_partial: true in the config. Do not do this for a real "
                "sweep -- every instance must score the same exam."
            )
        # First n in committed order: deterministic, identical on every machine, and
        # identical across every condition in this sweep.
        examples = examples[: cfg.prompts.n]
        print(
            f"NOTE: partial prompt set -- {cfg.prompts.n} of the frozen exam "
            f"(allow_partial). Not comparable with a full-exam sweep."
        )
    text = {e.prompt_id: e.prompt for e in examples}
    if cfg.prompts.chat_template:
        text = _apply_chat_template(cfg, text)
    return PromptSet(
        ids=[e.prompt_id for e in examples],
        text=text,
    )


def _apply_chat_template(cfg: SweepConfig, text: dict[str, str]) -> dict[str, str]:
    """Wrap every prompt in the target model's chat template.

    Applied once, to the shared prompt set, so that every condition in the sweep sees
    byte-identical inputs -- the same invariant the frozen subset exists to protect.

    The template is a property of the tokenizer, so a sweep spanning two different target
    models has two different templates and no single correct answer here. That raises
    rather than picking one: silently templating both arms with the first model's
    template would corrupt the second arm's prompts in a way nothing downstream detects.
    """
    models = sorted({c.target_model for c in cfg.conditions})
    if len(models) != 1:
        raise ValueError(
            "prompts.chat_template is on, but this sweep spans multiple target models "
            f"({models}). A chat template belongs to one tokenizer; applying one model's "
            "template to another's arm silently corrupts that arm. Split this into one "
            "config per model, or turn chat_template off."
        )
    model = models[0]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model)
    if not getattr(tok, "chat_template", None):
        raise ValueError(
            f"prompts.chat_template is on but {model!r} defines no chat template. "
            "Refusing to invent one -- turn the flag off, or point at a model that has "
            "one."
        )

    out: dict[str, str] = {}
    for pid, question in text.items():
        rendered = tok.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
        if question.strip() not in rendered:
            raise ValueError(
                f"chat template for {model!r} dropped the question text for {pid!r}. "
                "Refusing to measure a prompt that no longer contains the prompt."
            )
        out[pid] = rendered
    print(f"prompts   : chat template applied ({model})")
    return out


# --------------------------------------------------------------------------------------
# Progress
# --------------------------------------------------------------------------------------


def completed_keys(log_path: Path) -> set[tuple[str, str, int, bool]]:
    """Every unit already in the log. Empty set when the log does not exist yet."""
    if not log_path.exists():
        return set()
    done = set()
    for rec in read_records(log_path, validate=False):
        done.add((rec["condition_id"], rec["prompt_id"], int(rec["repeat_idx"]),
                  bool(rec["is_warmup"])))
    return done


def remaining(queue: list[WorkUnit], done: set) -> list[WorkUnit]:
    return [u for u in queue if u.key not in done]


# --------------------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------------------


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
    ap.add_argument("--config", required=True)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit")
    ap.add_argument("--allow-shared-gpu", action="store_true",
                    help="skip the GPU exclusivity check (never for real measurements)")
    # Internal: the driver re-invokes this module in worker mode.
    ap.add_argument("--worker-condition", help=argparse.SUPPRESS)
    ap.add_argument("--worker-round", type=int, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)

    cfg = load_sweep(args.config)
    prompts = load_prompts(cfg)
    queue = cfg.build_queue(prompts.ids)

    if args.worker_condition is not None:
        return _worker_main(cfg, prompts, queue, args)

    log_path = _resolve(cfg.log_path)
    done = completed_keys(log_path)
    todo = remaining(queue, done)

    print(f"instance   : {cfg.instance}")
    print(f"platform   : {cfg.platform} / stack {cfg.stack}")
    print(f"conditions : {len(cfg.conditions)}")
    print(f"prompts    : {len(prompts.ids)}")
    print(f"queue      : {len(queue)} units ({len(done)} already logged, {len(todo)} to run)")
    print(f"log        : {log_path}")

    if args.dry_run:
        for round_idx, condition, block in cfg.condition_visits(queue):
            pending = len([u for u in block if u.key not in done])
            print(f"  round {round_idx}  {condition.condition_id}  "
                  f"{_describe(condition)}  ({pending}/{len(block)} pending)")
        return 0

    if not todo:
        print("nothing to do; sweep is complete.")
        return 0

    failures: list[dict[str, Any]] = []
    for round_idx, condition, block in cfg.condition_visits(queue):
        if all(u.key in done for u in block):
            continue
        if cfg.platform != "cpu" and not args.allow_shared_gpu:
            # Re-checked before every block: a neighbour that appears mid-sweep would
            # otherwise contaminate only some conditions, which is worse than failing.
            assert_gpu_exclusive()

        print(f"\n=== round {round_idx} | {condition.condition_id} | {_describe(condition)}")
        rc = _spawn_worker(args.config, condition.condition_id, round_idx)
        if rc != 0:
            print(f"    !! worker exited {rc}; leaving this cell missing and continuing",
                  file=sys.stderr)
            failures.append({
                "round": round_idx,
                "condition_id": condition.condition_id,
                "returncode": rc,
                "at": utc_now(),
            })

    _write_failures(log_path, failures)
    if failures:
        print(f"\n{len(failures)} condition block(s) failed; see "
              f"{log_path.with_suffix('.failures.json')}. Those cells are missing, "
              "not estimated.", file=sys.stderr)
        return 1
    print("\nsweep complete.")
    return 0


def _spawn_worker(config_path: str, condition_id: str, round_idx: int) -> int:
    cmd = [
        sys.executable, "-m", "scripts.run_sweep",
        "--config", config_path,
        "--worker-condition", condition_id,
        "--worker-round", str(round_idx),
    ]
    return subprocess.run(cmd, cwd=REPO_ROOT).returncode


def _worker_main(cfg: SweepConfig, prompts: PromptSet, queue: list[WorkUnit],
                 args: argparse.Namespace) -> int:
    """Run exactly one (round, condition) block, then exit so the process dies with it."""
    block = [
        u for u in queue
        if u.config.condition_id == args.worker_condition and u.round_idx == args.worker_round
    ]
    if not block:
        raise ValueError(
            f"no work units for condition {args.worker_condition} round {args.worker_round}"
        )
    condition = block[0].config
    log_path = _resolve(cfg.log_path)
    done = completed_keys(log_path)
    pending = [u for u in block if u.key not in done]
    if not pending:
        return 0

    env = capture_env(
        stack=condition.stack,
        platform_name=condition.platform,
        llamacpp_binary=_llamacpp_binary(cfg),
        require_exclusive=(cfg.platform != "cpu" and not args.allow_shared_gpu),
    )
    runner = build_runner(condition, cfg)
    runner.setup()
    try:
        for unit in pending:
            prompt = prompts.text[unit.prompt_id]
            fillers = prompts.filler_pool(unit.prompt_id) if condition.batch_size > 1 else None
            result = runner.generate(prompt, fillers)
            record = build_record(
                cfg=condition,
                env=env,
                prompt_id=unit.prompt_id,
                repeat_idx=unit.repeat_idx,
                is_warmup=unit.is_warmup,
                result=result,
                latency_valid=(condition.stack != "hf"),
                acceptance_unavailable=bool(getattr(runner, "acceptance_unavailable", False)),
                resolved=runner.resolved,
            )
            append_record(log_path, record)
            flag = " (warmup)" if unit.is_warmup else ""
            print(f"    {unit.prompt_id} r{unit.repeat_idx}{flag}: "
                  f"tpot {record['tpot_ms']:.2f} ms")
    finally:
        runner.close()
    return 0


def build_runner(condition: RunConfig, cfg: SweepConfig):
    """Pick the runner for a condition. No fallback between stacks, ever."""
    if condition.stack == "vllm":
        from runners.vllm_runner import VLLMRunner

        return VLLMRunner(
            condition,
            allow_missing_acceptance=bool(cfg.raw.get("allow_missing_acceptance", False)),
        )
    if condition.stack == "hf":
        from runners.hf_runner import HFRunner

        return HFRunner(condition)
    if condition.stack == "llamacpp":
        from runners.llamacpp_runner import LlamaCppRunner

        raw = cfg.raw
        binary = _llamacpp_binary(cfg)
        if not binary:
            raise RunnerError("llamacpp stack requires 'model.binary' in the sweep config")
        model = raw.get("model", {})
        return LlamaCppRunner(
            condition,
            binary=binary,
            baseline_binary=model.get("baseline_binary"),
            model_path=model.get("target_gguf", condition.target_model),
            draft_model_path=model.get("draft_gguf", condition.draft_model),
            extra_args=list(model.get("extra_args") or []),
            baseline_extra_args=list(model.get("baseline_extra_args") or []),
        )
    raise ValueError(f"unknown stack {condition.stack!r}")


def _llamacpp_binary(cfg: SweepConfig) -> str | None:
    model = cfg.raw.get("model")
    if isinstance(model, dict):
        return model.get("binary")
    return None


def _describe(condition: RunConfig) -> str:
    # ASCII only: this goes to a terminal, and a Windows console in cp1252 raises
    # UnicodeEncodeError on a gamma. Figures and written files keep the real symbol.
    bits = [condition.target_dtype, condition.spec_method]
    if condition.num_speculative_tokens:
        bits.append(f"gamma={condition.num_speculative_tokens}")
    if condition.batch_size != 1:
        bits.append(f"batch={condition.batch_size}")
    if condition.tensor_parallel_size != 1:
        bits.append(f"tp={condition.tensor_parallel_size}")
    return " ".join(bits)


def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def _write_failures(log_path: Path, failures: Iterable[dict]) -> None:
    failures = list(failures)
    if not failures:
        return
    out = log_path.with_suffix(".failures.json")
    existing = json.loads(out.read_text(encoding="utf-8")) if out.exists() else []
    out.write_text(json.dumps(existing + failures, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
