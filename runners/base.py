"""Runner protocol and the one place a RunRecord is assembled.

Every runner returns a ``GenResult`` holding only what it actually measured. The
config- and environment-derived half of the record, and every derived quantity
(``tpot_ms``, ``acceptance_rate``, ``mean_accept_length``), are computed here. A runner
that computed its own ``tpot_ms`` could disagree with the validator's definition, and
the resulting record would be rejected -- or worse, quietly accepted with a different
meaning from its neighbours.

Histogram convention (fixed, and shared by all three runners)
------------------------------------------------------------
``accept_length_histogram[k]`` is the number of verification steps in which exactly
``k`` drafted tokens were accepted, for ``k`` in ``[0, gamma]``. So:

* ``sum(hist)``              -- number of target forward passes (verification steps)
* ``sum(k * hist[k])``       -- ``accepted_tokens``
* ``accepted_tokens / steps + 1`` -- ``mean_accept_length``, tokens emitted per step,
  the ``+1`` being the bonus token the target itself emits on every step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from core.env import gpu_sample, utc_now
from core.schema import RunConfig, base_record


@dataclass
class GenResult:
    """What a runner measured for one prompt. Measured quantities only."""

    ttft_ms: float
    total_ms: float
    output_tokens: int
    output_text: str
    prompt_tokens: int
    # Speculative statistics: all None when spec_method == 'none'.
    accept_length_histogram: list[int] = field(default_factory=list)
    draft_tokens_proposed: int | None = None
    # Anything runner-specific worth keeping (resolved configs, engine metrics).
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Runner(Protocol):
    """One condition, one process, one model load.

    Lifecycle is strictly ``setup() -> generate()* -> close()``. Reusing a runner
    across conditions is forbidden: CUDA graph caches, allocator state and autotuning
    results persist asymmetrically, so the second condition would inherit warm state
    the first one paid for.
    """

    config: RunConfig
    resolved: dict[str, Any]

    def setup(self) -> None: ...

    def generate(self, prompt: str, fillers: list[str] | None = None) -> GenResult: ...

    def close(self) -> None: ...


# How batch conditions are measured
# ---------------------------------
# At batch size B the runner submits the target prompt alongside B-1 *distinct* filler
# prompts drawn from the same pool, and records only the target prompt's timings. That
# gives one record per work unit, measured under genuine batch-B load.
#
# The obvious alternative -- submitting B copies of one prompt -- shares a prefix across
# the batch and measures cache hits rather than batching. Prefix caching is disabled in
# the engines regardless, for the same reason.


class RunnerError(RuntimeError):
    """Raised when a runner cannot measure what it was asked to measure.

    Always fatal. There is no fallback path anywhere in this package: a missing cell
    is correct, an invented one is not.
    """


def build_record(
    *,
    cfg: RunConfig,
    env: dict[str, Any],
    prompt_id: str,
    repeat_idx: int,
    is_warmup: bool,
    result: GenResult,
    latency_valid: bool = True,
    resolved: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble a complete, schema-valid record. Raises if the measurement is unusable."""
    if result.output_tokens < 2:
        raise RunnerError(
            f"output_tokens={result.output_tokens}: tpot_ms is undefined for fewer than "
            "two tokens. Not recording."
        )
    if cfg.ignore_eos and result.output_tokens != cfg.max_tokens:
        raise RunnerError(
            f"output_tokens={result.output_tokens} != max_tokens={cfg.max_tokens} under "
            "ignore_eos=True. The generation stopped early, so this timing is not "
            "comparable with its condition-mates. Not recording."
        )

    rec = base_record(cfg, env)
    rec["timestamp"] = utc_now()
    rec["prompt_id"] = prompt_id
    rec["prompt_tokens"] = int(result.prompt_tokens)
    rec["repeat_idx"] = int(repeat_idx)
    rec["is_warmup"] = bool(is_warmup)

    rec["ttft_ms"] = float(result.ttft_ms)
    rec["total_ms"] = float(result.total_ms)
    rec["output_tokens"] = int(result.output_tokens)
    rec["tpot_ms"] = (rec["total_ms"] - rec["ttft_ms"]) / (rec["output_tokens"] - 1)
    rec["output_text"] = result.output_text

    rec.update(_spec_fields(cfg, result))

    sample = gpu_sample() if cfg.platform != "cpu" else {"clocks_sm_mhz": None, "power_draw_w": None}
    rec["clocks_sm_mhz"] = sample["clocks_sm_mhz"]
    rec["power_draw_w"] = sample["power_draw_w"]

    if not latency_valid:
        rec["latency_valid"] = False
    if resolved:
        rec["resolved_spec_config"] = _stringify(resolved.get("speculative_config"))
        if resolved.get("model_path"):
            rec["resolved_model_path"] = str(resolved["model_path"])
    if env.get("env") is not None:
        rec["env"] = env["env"]
    for key in ("gpu_memory_utilization", "num_threads", "gguf_quant"):
        value = getattr(cfg, key, None)
        if value is not None:
            rec[key] = value
    return rec


def _spec_fields(cfg: RunConfig, result: GenResult) -> dict[str, Any]:
    """Derive the speculative half of a record from measured counts."""
    if cfg.spec_method == "none":
        if result.accept_length_histogram or result.draft_tokens_proposed:
            raise RunnerError(
                "runner returned speculative statistics for a non-speculative condition; "
                "the engine is not running the configuration we asked for."
            )
        return {
            "accepted_tokens": None,
            "draft_tokens_proposed": None,
            "acceptance_rate": None,
            "mean_accept_length": None,
            "accept_length_histogram": [],
        }

    hist = list(result.accept_length_histogram)
    if not hist:
        raise RunnerError(
            "speculative condition produced an empty accept_length_histogram. Acceptance "
            "extraction failed silently -- fix the extraction rather than recording a gap."
        )
    steps = sum(hist)
    if steps <= 0:
        raise RunnerError("accept_length_histogram sums to zero verification steps.")

    accepted = sum(k * n for k, n in enumerate(hist))
    proposed = result.draft_tokens_proposed
    if proposed is None:
        if cfg.gamma_schedule != "constant":
            raise RunnerError(
                "a non-constant gamma schedule must report draft_tokens_proposed "
                "explicitly; it cannot be derived from gamma and step count."
            )
        proposed = int(cfg.num_speculative_tokens) * steps
    proposed = int(proposed)
    if accepted > proposed:
        raise RunnerError(
            f"accepted ({accepted}) > proposed ({proposed}); acceptance parsing is wrong."
        )
    return {
        "accepted_tokens": accepted,
        "draft_tokens_proposed": proposed,
        "acceptance_rate": (accepted / proposed) if proposed > 0 else 0.0,
        "mean_accept_length": accepted / steps + 1.0,
        "accept_length_histogram": hist,
    }


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return repr(value)
