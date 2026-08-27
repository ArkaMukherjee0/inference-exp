"""YAML sweep config -> conditions -> a fully materialized work queue.

The expansion happens once, up front, and the driver consumes an already-ordered
queue. That matters for two reasons: the interleaving is inspectable before anything
runs, and a resumed sweep continues in exactly the order it would have followed had it
never been interrupted.

Interleaving granularity
------------------------
implementation.md asks for two things that pull against each other: conditions must be
interleaved across prompts (section 2.3), and each condition must run in its own process
with a full teardown (WP2). Strict per-prompt interleaving would mean reloading a model
between every single prompt, which is not a sweep, it is a model-loading benchmark.

The resolution is to interleave at *round* granularity. One round visits every condition
once, running the full prompt set for that condition in a fresh process. There are
``repeats`` rounds, and the condition order is rotated each round so no condition sits
permanently at the start (where the machine is coldest) or the end (where it is hottest).
Session drift is therefore spread across every condition rather than concentrated in
whichever ran last, which is what the interleaving requirement is actually protecting.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Iterator

import yaml

from core.schema import RunConfig

# Keys allowed at the top level of a sweep YAML. Unknown keys raise: a typo'd axis name
# that silently does nothing would produce a sweep that is quietly missing a dimension.
_TOP_LEVEL_KEYS = frozenset({
    "instance", "platform", "stack", "log_path", "prompts", "defaults", "axes",
    "interleave", "model", "notes",
})
_DEFAULTS_KEYS = frozenset({
    "max_tokens", "ignore_eos", "temperature", "seed", "repeats", "warmup",
    "batch_size", "tensor_parallel_size", "draft_tensor_parallel_size",
    "nccl_p2p_disabled", "gamma_schedule", "gpu_memory_utilization", "num_threads",
    "gguf_quant", "target_dtype", "draft_model", "spec_method",
    "num_speculative_tokens",
})
_PROMPTS_KEYS = frozenset({"source", "split", "n", "seed", "stride", "path",
                           "allow_partial"})


@dataclass(frozen=True)
class WorkUnit:
    """One thing to measure: a condition, a prompt, a repeat index."""

    config: RunConfig
    prompt_id: str
    repeat_idx: int
    is_warmup: bool
    round_idx: int

    @property
    def key(self) -> tuple[str, str, int, bool]:
        """Checkpoint identity. Warmups are keyed separately from real measurements."""
        return (self.config.condition_id, self.prompt_id, self.repeat_idx, self.is_warmup)


@dataclass(frozen=True)
class PromptSpec:
    source: str
    split: str | None
    n: int
    seed: int
    stride: int | None = None
    path: str | None = None
    # Opt-in to running fewer prompts than the frozen exam holds.
    #
    # Off by default, because a real sweep whose n disagrees with the frozen subset is
    # almost always a typo -- and one that would quietly score a different exam than
    # every other instance. The smoke test genuinely wants twenty prompts, so it says so.
    # Conditions within a partial run still all see the identical prompt set, and
    # analysis.stats.align_pair refuses to compare a 20-prompt condition against a
    # 250-prompt one, so a partial run cannot leak into the real results.
    allow_partial: bool = False


@dataclass(frozen=True)
class SweepConfig:
    instance: str
    platform: str
    stack: str
    log_path: Path
    prompts: PromptSpec
    conditions: tuple[RunConfig, ...]
    interleave: bool
    raw: dict[str, Any]

    # -- queue construction -------------------------------------------------------

    def build_queue(self, prompt_ids: list[str]) -> list[WorkUnit]:
        """Materialize the full ordered work queue.

        ``prompt_ids`` comes from the evaluation loader, not from this module: config
        expansion must not depend on a dataset download.
        """
        if not prompt_ids:
            raise ValueError("build_queue: prompt_ids is empty")
        if len(set(prompt_ids)) != len(prompt_ids):
            raise ValueError("build_queue: prompt_ids contains duplicates")
        if len(prompt_ids) != self.prompts.n:
            raise ValueError(
                f"build_queue: got {len(prompt_ids)} prompt ids but config asks for "
                f"n={self.prompts.n}. Refusing to run a differently-sized prompt set."
            )

        # Fixed shuffle, seeded from the config: prompt order is arbitrary but must be
        # identical across conditions, or condition differences absorb ordering effects.
        ordered = list(prompt_ids)
        random.Random(self.prompts.seed).shuffle(ordered)

        repeats = max(c.repeats for c in self.conditions)
        queue: list[WorkUnit] = []
        for round_idx in range(repeats):
            conds = self._rotated(round_idx) if self.interleave else self.conditions
            for cfg in conds:
                if round_idx >= cfg.repeats:
                    continue
                # A fresh process per condition-visit means a fresh model load, so
                # warmup is paid on every visit, not just the first.
                for w in range(cfg.warmup):
                    queue.append(WorkUnit(cfg, ordered[w % len(ordered)], round_idx, True, round_idx))
                for pid in ordered:
                    queue.append(WorkUnit(cfg, pid, round_idx, False, round_idx))
        return queue

    def _rotated(self, round_idx: int) -> tuple[RunConfig, ...]:
        if not self.conditions:
            return ()
        k = round_idx % len(self.conditions)
        return self.conditions[k:] + self.conditions[:k]

    def condition_visits(self, queue: list[WorkUnit]) -> Iterator[tuple[int, RunConfig, list[WorkUnit]]]:
        """Group a queue into consecutive (round, condition) blocks.

        One block == one subprocess == one model load and teardown.
        """
        block: list[WorkUnit] = []
        for unit in queue:
            if block and (unit.config.condition_id != block[0].config.condition_id
                          or unit.round_idx != block[0].round_idx):
                yield block[0].round_idx, block[0].config, block
                block = []
            block.append(unit)
        if block:
            yield block[0].round_idx, block[0].config, block


# --------------------------------------------------------------------------------------
# Parsing
# --------------------------------------------------------------------------------------


def _reject_unknown(got: dict, allowed: frozenset[str], where: str) -> None:
    unknown = sorted(set(got) - allowed)
    if unknown:
        raise ValueError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


def load_sweep(path: str | Path) -> SweepConfig:
    """Parse a sweep YAML into a validated SweepConfig."""
    p = Path(path)
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{p}: top level must be a mapping")
    _reject_unknown(raw, _TOP_LEVEL_KEYS, str(p))

    for key in ("instance", "platform", "stack", "log_path", "prompts", "axes"):
        if key not in raw:
            raise ValueError(f"{p}: missing required key {key!r}")

    _reject_unknown(raw["prompts"], _PROMPTS_KEYS, f"{p}:prompts")
    prompts = PromptSpec(
        source=raw["prompts"]["source"],
        split=raw["prompts"].get("split"),
        n=int(raw["prompts"]["n"]),
        seed=int(raw["prompts"].get("seed", 0)),
        stride=raw["prompts"].get("stride"),
        path=raw["prompts"].get("path"),
        allow_partial=bool(raw["prompts"].get("allow_partial", False)),
    )

    defaults = dict(raw.get("defaults") or {})
    _reject_unknown(defaults, _DEFAULTS_KEYS, f"{p}:defaults")

    conditions = _expand_axes(
        axes=raw["axes"],
        defaults=defaults,
        platform=raw["platform"],
        stack=raw["stack"],
        model=raw.get("model"),
        where=str(p),
    )
    if not conditions:
        raise ValueError(f"{p}: axes expanded to zero conditions")

    ids = [c.condition_id for c in conditions]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(
            f"{p}: axes produced duplicate conditions {sorted(dupes)}. Two grid points "
            "describe the same measurement; fix the axes rather than deduplicating."
        )

    return SweepConfig(
        instance=raw["instance"],
        platform=raw["platform"],
        stack=raw["stack"],
        log_path=Path(raw["log_path"]),
        prompts=prompts,
        conditions=tuple(conditions),
        interleave=bool(raw.get("interleave", True)),
        raw=raw,
    )


def _expand_axes(
    *,
    axes: dict[str, Any],
    defaults: dict[str, Any],
    platform: str,
    stack: str,
    model: Any,
    where: str,
) -> list[RunConfig]:
    """Cartesian product over axes, where each axis value is either a scalar or a dict
    of co-varying fields (as ``spec`` is: method and gamma must move together)."""
    if not isinstance(axes, dict) or not axes:
        raise ValueError(f"{where}: axes must be a non-empty mapping")

    names = sorted(axes)
    value_lists: list[list[dict[str, Any]]] = []
    for name in names:
        values = axes[name]
        if not isinstance(values, list) or not values:
            raise ValueError(f"{where}:axes.{name}: must be a non-empty list")
        expanded: list[dict[str, Any]] = []
        for v in values:
            if isinstance(v, dict):
                # A grouped axis: the dict names the fields it sets.
                expanded.append(dict(v))
            else:
                expanded.append({name: v})
        value_lists.append(expanded)

    models = _model_axis(model, where)

    out: list[RunConfig] = []
    for model_fields in models:
        for combo in product(*value_lists):
            merged: dict[str, Any] = dict(defaults)
            merged.update(model_fields)
            # Two axes setting the same field would make the grid silently
            # order-dependent, so the second one to touch a field is an error.
            set_by: dict[str, str] = {}
            for axis_name, part in zip(names, combo):
                for fname in part:
                    if fname in set_by:
                        raise ValueError(
                            f"{where}: axes {set_by[fname]!r} and {axis_name!r} both set "
                            f"field {fname!r}; a field may only be set by one axis."
                        )
                    set_by[fname] = axis_name
                merged.update(part)
            merged.setdefault("spec_method", "none")
            # A 'none' spec point inside a spec axis carries no gamma and no drafter.
            if merged.get("spec_method") == "none":
                merged["num_speculative_tokens"] = None
                merged["draft_model"] = None
            merged["platform"] = platform
            merged["stack"] = stack
            unknown = sorted(set(merged) - {f for f in RunConfig.__dataclass_fields__} )
            if unknown:
                raise ValueError(f"{where}: axes/defaults set unknown RunConfig field(s) {unknown}")
            merged.pop("condition_id", None)
            out.append(RunConfig(**merged))
    return out


# Keys under `model:` that wire up a runner rather than describe a condition. They are
# read from cfg.raw by the driver and deliberately excluded from RunConfig: a GGUF path
# differs between machines, and putting it in the condition hash would give the same
# measurement two different condition_ids on two hosts. What the weights *are* is
# captured by target_model and gguf_quant, which do enter the hash.
_MODEL_WIRING_KEYS = frozenset({"binary", "target_gguf", "draft_gguf"})


def _model_axis(model: Any, where: str) -> list[dict[str, Any]]:
    """`model:` is either a single mapping or a list of them (target + its drafter)."""
    if model is None:
        return [{}]
    if isinstance(model, dict):
        model = [model]
    if not isinstance(model, list):
        raise ValueError(f"{where}:model: must be a mapping or a list of mappings")
    allowed = frozenset({"target_model", "draft_model"}) | _MODEL_WIRING_KEYS
    out = []
    for m in model:
        _reject_unknown(m, allowed, f"{where}:model")
        out.append({k: v for k, v in m.items() if k not in _MODEL_WIRING_KEYS})
    return out
