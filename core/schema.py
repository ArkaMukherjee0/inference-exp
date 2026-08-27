"""Shared measurement contract for the inference-optimization study.

This module is the interface between every other module. Do not change it without
updating `implementation.md` first (spec rule 3).

Two objects live here:

* ``RunConfig``  -- one immutable measurement condition.
* the ``RunRecord`` schema -- one JSON object per prompt per repeat.

Plus the two functions everything writes through: ``validate_record`` and
``append_record``.

Design stance: this validator is deliberately hostile. It coerces nothing, fills
nothing in, and raises ``ValueError`` naming the offending field. A record that is
even slightly wrong must not reach the log, because a wrong record is
indistinguishable from a right one at analysis time.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import uuid
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Final, Literal

# --------------------------------------------------------------------------------------
# Enumerations (kept as plain tuples so validation and typing share one source)
# --------------------------------------------------------------------------------------

DTYPES: Final = ("bf16", "fp8", "w4a16")
SPEC_METHODS: Final = ("none", "draft_model", "mtp", "eagle", "eagle3", "ngram")
GAMMA_SCHEDULES: Final = ("constant", "heuristic")
STACKS: Final = ("vllm", "hf", "llamacpp")
PLATFORMS: Final = ("h100", "cpu")
PROVENANCES: Final = ("measured", "fixture")

# Fields excluded from the condition hash: they describe how many times a condition is
# sampled, not what the condition *is*. Including them would split one condition into
# two the moment a sweep resumed with a different repeat count.
_HASH_EXCLUDED: Final = frozenset({"condition_id", "repeats", "warmup"})


# --------------------------------------------------------------------------------------
# RunConfig
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunConfig:
    """One immutable object fully describing a single measurement condition."""

    # model
    target_model: str
    target_dtype: Literal["bf16", "fp8", "w4a16"]
    stack: Literal["vllm", "hf", "llamacpp"]
    platform: Literal["h100", "cpu"]
    draft_model: str | None = None
    spec_method: Literal["none", "draft_model", "mtp", "eagle", "eagle3", "ngram"] = "none"
    num_speculative_tokens: int | None = None
    gamma_schedule: Literal["constant", "heuristic"] = "constant"
    # parallelism
    tensor_parallel_size: int = 1
    draft_tensor_parallel_size: int = 1
    nccl_p2p_disabled: bool = False
    # generation
    batch_size: int = 1
    max_tokens: int = 256
    ignore_eos: bool = True
    temperature: float = 0.0
    seed: int = 42
    # execution
    repeats: int = 5
    warmup: int = 3
    # engine knobs that change the measurement and must therefore be recorded
    gpu_memory_utilization: float | None = None
    num_threads: int | None = None
    gguf_quant: str | None = None
    # identity -- derived in __post_init__, never passed by callers
    condition_id: str = field(default="", compare=False)

    def __post_init__(self) -> None:
        self._check()
        object.__setattr__(self, "condition_id", self._compute_condition_id())

    # -- validation ---------------------------------------------------------------

    def _check(self) -> None:
        if self.target_dtype not in DTYPES:
            raise ValueError(f"target_dtype: {self.target_dtype!r} not in {DTYPES}")
        if self.spec_method not in SPEC_METHODS:
            raise ValueError(f"spec_method: {self.spec_method!r} not in {SPEC_METHODS}")
        if self.gamma_schedule not in GAMMA_SCHEDULES:
            raise ValueError(f"gamma_schedule: {self.gamma_schedule!r} not in {GAMMA_SCHEDULES}")
        if self.stack not in STACKS:
            raise ValueError(f"stack: {self.stack!r} not in {STACKS}")
        if self.platform not in PLATFORMS:
            raise ValueError(f"platform: {self.platform!r} not in {PLATFORMS}")

        if self.spec_method == "none":
            if self.num_speculative_tokens is not None:
                raise ValueError(
                    "num_speculative_tokens must be None when spec_method == 'none' "
                    f"(got {self.num_speculative_tokens!r})"
                )
            if self.draft_model is not None:
                raise ValueError(
                    f"draft_model must be None when spec_method == 'none' (got {self.draft_model!r})"
                )
        else:
            if self.num_speculative_tokens is None:
                raise ValueError(f"num_speculative_tokens required for spec_method={self.spec_method!r}")
            if self.num_speculative_tokens < 1:
                raise ValueError(f"num_speculative_tokens must be >= 1 (got {self.num_speculative_tokens})")
            # ngram and mtp/eagle draw their draft from the target itself; only the
            # generic draft-model path needs a separate checkpoint.
            if self.spec_method == "draft_model" and not self.draft_model:
                raise ValueError("draft_model required for spec_method='draft_model'")

        if self.max_tokens < 2:
            # tpot_ms is undefined for a single token; see validate_record.
            raise ValueError(f"max_tokens must be >= 2 (got {self.max_tokens})")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1 (got {self.batch_size})")
        if self.repeats < 1:
            raise ValueError(f"repeats must be >= 1 (got {self.repeats})")
        if self.warmup < 0:
            raise ValueError(f"warmup must be >= 0 (got {self.warmup})")
        if self.tensor_parallel_size < 1 or self.draft_tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size and draft_tensor_parallel_size must be >= 1")
        if self.temperature < 0:
            raise ValueError(f"temperature must be >= 0 (got {self.temperature})")

    # -- identity -----------------------------------------------------------------

    def _compute_condition_id(self) -> str:
        """First 12 hex chars of sha256 over the sorted field dict.

        Deterministic across machines and processes: ``hash()`` is salted per process
        and must never be used for this.
        """
        payload = {f.name: getattr(self, f.name) for f in fields(self) if f.name not in _HASH_EXCLUDED}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# --------------------------------------------------------------------------------------
# RunRecord schema
# --------------------------------------------------------------------------------------

REQUIRED_FIELDS: Final[list[str]] = [
    "run_id", "condition_id", "provenance", "timestamp",
    "hostname", "platform", "stack", "stack_version", "driver",
    "target_model", "target_dtype", "draft_model", "spec_method",
    "num_speculative_tokens", "gamma_schedule",
    "tensor_parallel_size", "draft_tensor_parallel_size", "nccl_p2p_disabled",
    "batch_size", "prompt_id", "prompt_tokens", "max_tokens", "ignore_eos",
    "temperature", "seed", "repeat_idx", "is_warmup",
    "ttft_ms", "tpot_ms", "total_ms", "output_tokens",
    "accepted_tokens", "draft_tokens_proposed", "acceptance_rate",
    "mean_accept_length", "accept_length_histogram",
    "clocks_sm_mhz", "power_draw_w", "output_text",
]

# Extras that modules are permitted to attach. Anything outside REQUIRED + OPTIONAL is
# rejected: unknown keys are how schema drift between parallel agents starts.
OPTIONAL_FIELDS: Final[frozenset[str]] = frozenset({
    "latency_valid",          # WP4: False on every HF record; gates all speed figures
    "timing_method",          # how ttft/total were obtained; see runners/vllm_runner.py
    "acceptance_unavailable", # explicit exemption: engine exposes no acceptance counts
    "resolved_spec_config",   # WP2: what the engine actually resolved our request to
    "resolved_model_path",    # guard against silent model substitution
    "gpu_memory_utilization",
    "num_threads",
    "gguf_quant",
    "env",                    # core.env provenance blob
    "notes",
})

ALLOWED_FIELDS: Final[frozenset[str]] = frozenset(REQUIRED_FIELDS) | OPTIONAL_FIELDS

# Fields that must be null (never 0) when they do not apply.
_SPEC_ONLY_FIELDS: Final = (
    "num_speculative_tokens",
    "accepted_tokens",
    "draft_tokens_proposed",
    "acceptance_rate",
    "mean_accept_length",
)

# Tolerance for the tpot_ms consistency re-derivation, in milliseconds.
_TPOT_TOL_MS: Final = 1e-6


def _require_type(rec: dict, key: str, types: tuple[type, ...], *, allow_none: bool = False) -> Any:
    val = rec[key]
    if val is None:
        if allow_none:
            return None
        raise ValueError(f"{key}: must not be null")
    if isinstance(val, bool) and bool not in types:
        # bool is a subclass of int in Python; almost never what we mean here.
        raise ValueError(f"{key}: got bool, expected {[t.__name__ for t in types]}")
    if not isinstance(val, types):
        raise ValueError(f"{key}: got {type(val).__name__}, expected {[t.__name__ for t in types]}")
    return val


def _require_positive(rec: dict, key: str, *, allow_none: bool = False) -> float | None:
    val = _require_type(rec, key, (int, float), allow_none=allow_none)
    if val is None:
        return None
    if not math.isfinite(val):
        raise ValueError(f"{key}: must be finite (got {val!r})")
    if val <= 0:
        raise ValueError(f"{key}: must be > 0 (got {val!r})")
    return float(val)


def validate_record(rec: dict) -> None:
    """Raise ``ValueError`` naming the offending field, or return None.

    Every rule in implementation.md section 2.2 is enforced here. This is the only
    gate between a runner and the log.
    """
    if not isinstance(rec, dict):
        raise ValueError(f"record must be a dict, got {type(rec).__name__}")

    missing = [f for f in REQUIRED_FIELDS if f not in rec]
    if missing:
        raise ValueError(f"missing required field(s): {missing}")

    unknown = sorted(set(rec) - ALLOWED_FIELDS)
    if unknown:
        raise ValueError(
            f"unknown field(s): {unknown}. Extend REQUIRED_FIELDS/OPTIONAL_FIELDS in "
            "core/schema.py and implementation.md together, or drop the field."
        )

    # -- provenance ------------------------------------------------------------------
    prov = _require_type(rec, "provenance", (str,))
    if prov not in PROVENANCES:
        raise ValueError(f"provenance: {prov!r} not in {PROVENANCES}")

    # -- identity / environment ------------------------------------------------------
    for key in ("run_id", "condition_id", "timestamp", "hostname", "stack_version", "driver",
                "target_model", "target_dtype", "spec_method", "gamma_schedule",
                "prompt_id", "output_text"):
        _require_type(rec, key, (str,))
    if not rec["run_id"]:
        raise ValueError("run_id: must not be empty")
    if not rec["condition_id"]:
        raise ValueError("condition_id: must not be empty")

    _require_type(rec, "draft_model", (str,), allow_none=True)

    if rec["platform"] not in PLATFORMS:
        raise ValueError(f"platform: {rec['platform']!r} not in {PLATFORMS}")
    if rec["stack"] not in STACKS:
        raise ValueError(f"stack: {rec['stack']!r} not in {STACKS}")
    if rec["target_dtype"] not in DTYPES:
        raise ValueError(f"target_dtype: {rec['target_dtype']!r} not in {DTYPES}")
    if rec["spec_method"] not in SPEC_METHODS:
        raise ValueError(f"spec_method: {rec['spec_method']!r} not in {SPEC_METHODS}")
    if rec["gamma_schedule"] not in GAMMA_SCHEDULES:
        raise ValueError(f"gamma_schedule: {rec['gamma_schedule']!r} not in {GAMMA_SCHEDULES}")

    # -- flags -----------------------------------------------------------------------
    for key in ("nccl_p2p_disabled", "ignore_eos", "is_warmup"):
        _require_type(rec, key, (bool,))
    if "latency_valid" in rec:
        _require_type(rec, "latency_valid", (bool,))
    if "timing_method" in rec:
        _require_type(rec, "timing_method", (str,))
    if "acceptance_unavailable" in rec:
        _require_type(rec, "acceptance_unavailable", (bool,))

    # -- integers --------------------------------------------------------------------
    for key in ("tensor_parallel_size", "draft_tensor_parallel_size", "batch_size",
                "prompt_tokens", "max_tokens", "seed", "repeat_idx", "output_tokens"):
        val = _require_type(rec, key, (int,))
        if key in ("tensor_parallel_size", "draft_tensor_parallel_size", "batch_size",
                   "prompt_tokens", "max_tokens", "output_tokens") and val < 1:
            raise ValueError(f"{key}: must be >= 1 (got {val})")
        if key == "repeat_idx" and val < 0:
            raise ValueError(f"repeat_idx: must be >= 0 (got {val})")

    _require_type(rec, "temperature", (int, float))
    if rec["temperature"] < 0:
        raise ValueError(f"temperature: must be >= 0 (got {rec['temperature']})")

    # -- the ignore_eos guard --------------------------------------------------------
    # The single most important validation in the codebase. Without it, a "faster"
    # condition may simply be one that emitted fewer tokens, and the speedup column
    # silently becomes a measure of verbosity.
    if rec["ignore_eos"] and rec["output_tokens"] != rec["max_tokens"]:
        raise ValueError(
            f"output_tokens ({rec['output_tokens']}) != max_tokens ({rec['max_tokens']}) "
            "under ignore_eos=True -- the generation stopped early, so this timing is not "
            "comparable. Do not record it."
        )

    # -- timings ---------------------------------------------------------------------
    ttft = _require_positive(rec, "ttft_ms")
    total = _require_positive(rec, "total_ms")
    tpot = _require_positive(rec, "tpot_ms")
    if total < ttft:
        raise ValueError(f"total_ms ({total}) < ttft_ms ({ttft})")
    if rec["output_tokens"] < 2:
        raise ValueError(
            f"output_tokens ({rec['output_tokens']}) < 2: tpot_ms is undefined for a "
            "single token."
        )
    expected_tpot = (total - ttft) / (rec["output_tokens"] - 1)
    if abs(expected_tpot - tpot) > max(_TPOT_TOL_MS, abs(expected_tpot) * 1e-9):
        raise ValueError(
            f"tpot_ms ({tpot!r}) is not (total_ms - ttft_ms) / (output_tokens - 1) "
            f"= {expected_tpot!r}. Recompute it; do not record a hand-supplied value."
        )

    # -- clocks / power --------------------------------------------------------------
    # Null is legitimate on CPU where there is no SM clock; zero is not, anywhere.
    for key in ("clocks_sm_mhz", "power_draw_w"):
        _require_positive(rec, key, allow_none=True)

    # -- speculative fields ----------------------------------------------------------
    hist = rec["accept_length_histogram"]
    if not isinstance(hist, list):
        raise ValueError(f"accept_length_histogram: expected list, got {type(hist).__name__}")
    if any(not isinstance(x, int) or isinstance(x, bool) or x < 0 for x in hist):
        raise ValueError("accept_length_histogram: every entry must be a non-negative int")

    if rec["spec_method"] != "none" and rec.get("acceptance_unavailable"):
        # A speculative run on an engine that reports no per-request acceptance.
        #
        # The default rule -- speculation implies a non-empty histogram -- exists to catch
        # extraction silently failing, and it stays in force everywhere else. This is the
        # one way past it, and it is deliberately loud: the flag is stamped on the record,
        # so a missing distribution can never be mistaken for a measured one, and the
        # analysis layer refuses these rows for any acceptance-derived figure.
        #
        # Speed measurements on such a record remain fully valid. Acceptance ones do not
        # exist, which is different from being zero.
        for key in ("accepted_tokens", "draft_tokens_proposed", "acceptance_rate",
                    "mean_accept_length"):
            if rec[key] is not None:
                raise ValueError(
                    f"{key}: must be null when acceptance_unavailable is true (got "
                    f"{rec[key]!r}). Partial acceptance data is not an exemption."
                )
        if rec["accept_length_histogram"]:
            raise ValueError(
                "accept_length_histogram: must be [] when acceptance_unavailable is true"
            )
    elif rec["spec_method"] == "none":
        for key in _SPEC_ONLY_FIELDS:
            if rec[key] is not None:
                raise ValueError(
                    f"{key}: must be null when spec_method == 'none' (got {rec[key]!r}). "
                    "Non-applicable numeric fields are null, never 0."
                )
        if hist:
            raise ValueError("accept_length_histogram: must be [] when spec_method == 'none'")
    else:
        if rec["num_speculative_tokens"] is None:
            raise ValueError(f"num_speculative_tokens: required for spec_method={rec['spec_method']!r}")
        _require_type(rec, "num_speculative_tokens", (int,))
        accepted = _require_type(rec, "accepted_tokens", (int,))
        proposed = _require_type(rec, "draft_tokens_proposed", (int,))
        if accepted < 0 or proposed < 0:
            raise ValueError("accepted_tokens and draft_tokens_proposed must be >= 0")
        if accepted > proposed:
            raise ValueError(
                f"accepted_tokens ({accepted}) > draft_tokens_proposed ({proposed}): "
                "more tokens were accepted than were ever drafted."
            )
        rate = _require_type(rec, "acceptance_rate", (int, float))
        if not 0.0 <= rate <= 1.0:
            raise ValueError(f"acceptance_rate: must be in [0, 1] (got {rate})")
        if proposed > 0:
            expected_rate = accepted / proposed
            if abs(expected_rate - rate) > 1e-6:
                raise ValueError(
                    f"acceptance_rate ({rate}) != accepted_tokens / draft_tokens_proposed "
                    f"({expected_rate})"
                )
        mal = _require_type(rec, "mean_accept_length", (int, float))
        if mal < 0:
            raise ValueError(f"mean_accept_length: must be >= 0 (got {mal})")
        if not hist:
            raise ValueError(
                "accept_length_histogram: must be non-empty when speculation is active. "
                "An empty histogram means acceptance extraction silently failed."
            )


# --------------------------------------------------------------------------------------
# JSONL IO
# --------------------------------------------------------------------------------------


def append_record(path: str | os.PathLike[str], rec: dict) -> None:
    """Validate, then append exactly one JSON line.

    Append-only and flushed per record: a sweep killed mid-run must leave every
    completed measurement on disk and intact.
    """
    validate_record(rec)
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(rec, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    if "\n" in line:
        raise ValueError("serialized record contains a newline; JSONL invariant broken")
    with p.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def read_records(path: str | os.PathLike[str], *, validate: bool = True) -> list[dict]:
    """Read a JSONL log. Raises on the first malformed or invalid line, naming it."""
    p = Path(path)
    out: list[dict] = []
    with p.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{p}:{lineno}: not valid JSON: {exc}") from exc
            if validate:
                try:
                    validate_record(rec)
                except ValueError as exc:
                    raise ValueError(f"{p}:{lineno}: {exc}") from exc
            out.append(rec)
    return out


def new_run_id() -> str:
    """A fresh globally-unique run identifier.

    uuid4, not a counter: three cloud instances and a desktop append to separate logs
    with no coordination, and resumed sweeps must not reissue an id.
    """
    return uuid.uuid4().hex


def base_record(cfg: RunConfig, env: dict[str, Any]) -> dict[str, Any]:
    """The config- and environment-derived half of a record.

    Runners fill in the measured half. Keeping this in one place means a runner cannot
    disagree with the config about what it ran.
    """
    return {
        "run_id": new_run_id(),
        "condition_id": cfg.condition_id,
        "provenance": "measured",
        "hostname": env["hostname"],
        "platform": cfg.platform,
        "stack": cfg.stack,
        "stack_version": env["stack_version"],
        "driver": env["driver"],
        "target_model": cfg.target_model,
        "target_dtype": cfg.target_dtype,
        "draft_model": cfg.draft_model,
        "spec_method": cfg.spec_method,
        "num_speculative_tokens": cfg.num_speculative_tokens,
        "gamma_schedule": cfg.gamma_schedule,
        "tensor_parallel_size": cfg.tensor_parallel_size,
        "draft_tensor_parallel_size": cfg.draft_tensor_parallel_size,
        "nccl_p2p_disabled": cfg.nccl_p2p_disabled,
        "batch_size": cfg.batch_size,
        "max_tokens": cfg.max_tokens,
        "ignore_eos": cfg.ignore_eos,
        "temperature": cfg.temperature,
        "seed": cfg.seed,
    }
