"""JSONL -> DataFrame, with the guards that keep bad data out of the report.

Three filters live here, and all three are on by default:

* **provenance** -- fixture records exist so the analysis and plotting code can be built
  before a GPU is free. ``require_measured`` is what structurally prevents them from
  reaching a figure. Every plot module calls it as its first statement.
* **warmup** -- excluded at the loader, so "forgot to filter warmups" is not a mistake
  anyone downstream can make.
* **latency_valid** -- HF records carry ``latency_valid: False`` and are dropped from
  anything that reads a timing column.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from core.schema import read_records

# Columns every analysis function may rely on being present and typed.
_NUMERIC = (
    "ttft_ms", "tpot_ms", "total_ms", "output_tokens", "prompt_tokens",
    "accepted_tokens", "draft_tokens_proposed", "acceptance_rate",
    "mean_accept_length", "clocks_sm_mhz", "power_draw_w", "batch_size",
    "num_speculative_tokens", "tensor_parallel_size", "draft_tensor_parallel_size",
    "max_tokens", "repeat_idx", "temperature", "seed",
)


class ProvenanceError(RuntimeError):
    """Raised when non-measured data reaches a function that reports results."""


def load_runs(
    paths: str | Path | Iterable[str | Path],
    *,
    drop_warmup: bool = True,
    validate: bool = True,
) -> pd.DataFrame:
    """Load one or more JSONL logs into a DataFrame.

    Every record is validated on the way in unless explicitly disabled, so a schema
    violation surfaces at load time -- naming the file and line -- rather than as a
    puzzling NaN three modules later.
    """
    if isinstance(paths, (str, Path)):
        paths = [paths]
    records: list[dict] = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"log not found: {path}")
        records.extend(read_records(path, validate=validate))

    if not records:
        raise ValueError(f"no records loaded from {list(paths)!r}")

    df = pd.DataFrame.from_records(records)
    for col in _NUMERIC:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "latency_valid" not in df.columns:
        df["latency_valid"] = True
    df["latency_valid"] = df["latency_valid"].fillna(True).astype(bool)

    if drop_warmup:
        df = df[~df["is_warmup"].astype(bool)].copy()
        if df.empty:
            raise ValueError("every loaded record was a warmup; nothing left to analyse")
    return df.reset_index(drop=True)


def require_measured(df: pd.DataFrame) -> pd.DataFrame:
    """Raise unless every row is a real measurement. Returns the frame unchanged.

    Called first by every plotting and reporting function. Returning the frame rather
    than None lets it be used inline, which makes the guard hard to forget and easy to
    spot in review.
    """
    if "provenance" not in df.columns:
        raise ProvenanceError("frame has no 'provenance' column; refusing to report on it")
    bad = df.loc[df["provenance"] != "measured", "provenance"]
    if not bad.empty:
        counts = bad.value_counts().to_dict()
        raise ProvenanceError(
            f"{len(bad)} of {len(df)} records are not measured ({counts}). Fixture data "
            "must never reach a figure or a table."
        )
    return df


def speed_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Rows whose timings may appear in a speed figure.

    Drops HF records. Kept as one named function so every speed figure filters
    identically, and so the test that asserts "no HF record reaches a speed figure" has
    a single place to check.
    """
    df = require_measured(df)
    out = df[df["latency_valid"]].copy()
    if out.empty:
        raise ValueError(
            "no latency-valid records remain. Every row was marked latency_valid=False "
            "(HF timings), which must not appear in a speed figure."
        )
    stacks = set(out["stack"].unique())
    if "hf" in stacks:
        raise ProvenanceError(
            "an HF record survived the latency_valid filter. HF timings understate "
            "speculative speedup and must never reach a speed figure."
        )
    return out


def assert_single_stack(df: pd.DataFrame, where: str = "figure") -> str:
    """Raise if a frame mixes stacks. Returns the single stack name.

    vLLM, HF and llama.cpp have entirely different overheads, so a number from one is
    not comparable with a number from another. Any figure placing them on shared axes
    must do it with within-stack *ratios*, never raw timings.
    """
    stacks = sorted(df["stack"].dropna().unique())
    if len(stacks) != 1:
        raise ValueError(
            f"{where}: frame spans stacks {stacks}. Cross-stack raw comparison is not "
            "meaningful; compare within-stack ratios instead."
        )
    return stacks[0]


def condition_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per condition, carrying the fields that define it.

    Used by the report's provenance appendix and by every figure that needs to label a
    point with what produced it.
    """
    keys = [
        "condition_id", "hostname", "platform", "stack", "stack_version", "driver",
        "target_model", "target_dtype", "draft_model", "spec_method",
        "num_speculative_tokens", "gamma_schedule", "tensor_parallel_size",
        "draft_tensor_parallel_size", "nccl_p2p_disabled", "batch_size", "max_tokens",
        "ignore_eos", "temperature", "seed",
    ]
    present = [k for k in keys if k in df.columns]
    grouped = df.groupby("condition_id", dropna=False)
    table = grouped[present].first().reset_index(drop=True)
    table["n_records"] = grouped.size().to_numpy()
    table["n_prompts"] = grouped["prompt_id"].nunique().to_numpy()
    table["run_ids"] = grouped["run_id"].apply(list).to_numpy()
    if "clocks_sm_mhz" in df.columns:
        table["clocks_sm_mhz_median"] = grouped["clocks_sm_mhz"].median().to_numpy()
    return table


def baseline_map(df: pd.DataFrame, axes: list[str]) -> dict[str, str]:
    """Map each speculative condition to its non-speculative twin.

    Two conditions are twins when they agree on every field in ``axes`` and differ only
    in speculation. Pairing by anything looser -- "the bf16 baseline", say -- would
    compare across batch size or TP without saying so.
    """
    missing = [a for a in axes if a not in df.columns]
    if missing:
        raise ValueError(f"baseline_map: frame lacks axis column(s) {missing}")

    conds = df.drop_duplicates("condition_id")
    baselines = conds[conds["spec_method"] == "none"]
    if baselines.empty:
        raise ValueError("no spec_method='none' condition present; nothing to pair against")

    key_to_baseline: dict[tuple, str] = {}
    for _, row in baselines.iterrows():
        key = tuple(row[a] for a in axes)
        if key in key_to_baseline:
            raise ValueError(f"two baseline conditions share axes {dict(zip(axes, key))}")
        key_to_baseline[key] = row["condition_id"]

    out: dict[str, str] = {}
    for _, row in conds[conds["spec_method"] != "none"].iterrows():
        key = tuple(row[a] for a in axes)
        if key not in key_to_baseline:
            raise ValueError(
                f"condition {row['condition_id']} has no baseline twin at "
                f"{dict(zip(axes, key))}. Refusing to pair it against a different cell."
            )
        out[row["condition_id"]] = key_to_baseline[key]
    return out
