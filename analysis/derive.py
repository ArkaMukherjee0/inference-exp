"""Derived tables shared by the figures and the report.

Not in the original module list, but the alternative was eight figure modules each
computing speedup their own way -- and eight chances for one of them to pair against the
wrong baseline or forget to collapse repeats. There is one implementation here and every
figure calls it.
"""

from __future__ import annotations

import pandas as pd

from analysis.load import baseline_map, require_measured, speed_frame
from analysis.stats import median_over_repeats, speedup_between

# Fields that must match between a speculative condition and its baseline twin. Anything
# omitted here becomes a hidden variable in every speedup number.
DEFAULT_PAIR_AXES = [
    "target_model", "target_dtype", "batch_size", "tensor_parallel_size",
    "nccl_p2p_disabled", "max_tokens", "stack", "platform", "hostname",
]


def speedup_table(
    df: pd.DataFrame,
    *,
    pair_axes: list[str] | None = None,
    value_col: str = "tpot_ms",
    n_boot: int = 10_000,
    seed: int = 0,
) -> pd.DataFrame:
    """One row per speculative condition: geometric-mean speedup vs its baseline twin.

    Speedup is measured on ``tpot_ms`` by default -- time per output token, excluding
    the first. Speculative decoding acts only on decode, so folding prefill in would
    dilute the effect by a prompt-length-dependent amount.
    """
    axes = list(pair_axes or DEFAULT_PAIR_AXES)
    axes = [a for a in axes if a in df.columns]

    speed = speed_frame(df)
    collapsed = median_over_repeats(speed)
    pairs = baseline_map(speed, axes)

    meta = speed.drop_duplicates("condition_id").set_index("condition_id")
    rows = []
    for opt_id, base_id in pairs.items():
        interval = speedup_between(
            collapsed, base_condition=base_id, opt_condition=opt_id,
            value_col=value_col, n_boot=n_boot, seed=seed,
        )
        row = {
            "condition_id": opt_id,
            "baseline_condition_id": base_id,
            "speedup": interval.point,
            "speedup_lo95": interval.lo95,
            "speedup_hi95": interval.hi95,
            "n_prompts": interval.n,
        }
        for col in set(axes) | {"spec_method", "num_speculative_tokens", "draft_model",
                                "gamma_schedule", "draft_tensor_parallel_size"}:
            if col in meta.columns:
                row[col] = meta.loc[opt_id, col]
        rows.append(row)

    if not rows:
        raise ValueError(
            "no speculative conditions could be paired with a baseline. Check that the "
            "sweep actually contains spec_method='none' twins."
        )
    return pd.DataFrame(rows)


def acceptance_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-condition acceptance statistics, including the summed run-length histogram.

    Histograms are summed across prompts and repeats rather than averaged: they are
    counts of verification steps, and adding counts is the only operation that keeps
    them counts.
    """
    df = require_measured(df)
    spec = df[df["spec_method"] != "none"]
    if spec.empty:
        raise ValueError("no speculative conditions present")

    if "acceptance_unavailable" in spec.columns:
        exempt = spec["acceptance_unavailable"].fillna(False).astype(bool)
        if exempt.all():
            raise ValueError(
                "every speculative record is marked acceptance_unavailable: this engine "
                "reported no per-request acceptance counts, so the accepted-run-length "
                "distribution was never measured. Figures 04 and 06 cannot be built from "
                "these runs, and a histogram inferred from speedup alone would be "
                "fabricated. Speed figures (01, 02, 03, 05) are unaffected."
            )
        spec = spec[~exempt]

    rows = []
    for cid, group in spec.groupby("condition_id"):
        gamma = int(group["num_speculative_tokens"].iloc[0])
        summed = [0] * (gamma + 1)
        for hist in group["accept_length_histogram"]:
            if len(hist) != gamma + 1:
                raise ValueError(
                    f"condition {cid}: histogram length {len(hist)} does not match "
                    f"gamma+1 ({gamma + 1}); histograms from different gammas cannot be "
                    "summed."
                )
            for k, count in enumerate(hist):
                summed[k] += int(count)
        total_steps = sum(summed)
        rows.append({
            "condition_id": cid,
            "spec_method": group["spec_method"].iloc[0],
            "num_speculative_tokens": gamma,
            "target_dtype": group["target_dtype"].iloc[0],
            "target_model": group["target_model"].iloc[0],
            "draft_model": group["draft_model"].iloc[0],
            "batch_size": int(group["batch_size"].iloc[0]),
            "acceptance_rate": float(group["acceptance_rate"].mean()),
            "mean_accept_length": float(group["mean_accept_length"].mean()),
            "accept_length_histogram": summed,
            "n_steps": total_steps,
            "n_records": len(group),
        })
    return pd.DataFrame(rows)


def throughput_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per-condition decode throughput in tokens/second, from median tpot.

    Throughput is derived from the collapsed per-prompt median rather than from total
    tokens over total wall time, so a single slow prompt cannot drag a condition's
    headline number.
    """
    speed = speed_frame(df)
    collapsed = median_over_repeats(speed)
    grouped = collapsed.groupby("condition_id")
    out = grouped.agg(
        tpot_ms_median=("tpot_ms", "median"),
        n_prompts=("prompt_id", "nunique"),
    ).reset_index()
    out["throughput_tok_s"] = 1000.0 / out["tpot_ms_median"]

    meta = speed.drop_duplicates("condition_id").set_index("condition_id")
    for col in ("target_dtype", "spec_method", "num_speculative_tokens", "batch_size",
                "target_model", "stack", "platform", "tensor_parallel_size"):
        if col in meta.columns:
            out[col] = out["condition_id"].map(meta[col])
    # Per-request throughput times batch size is the served rate; at batch 1 they agree.
    if "batch_size" in out.columns:
        out["throughput_tok_s_batch"] = out["throughput_tok_s"] * out["batch_size"]
    return out


def condition_label(row: pd.Series, *, include_model: bool = False) -> str:
    """A short human label for a condition, used on figure points and in tables.

    ``include_model`` is required wherever two models can appear in one figure: without
    it, the same precision and gamma on two different targets produce identical labels,
    and a facet grid would silently merge two distinct conditions into one panel.
    """
    from plots.style import DTYPE_LABEL

    dtype = DTYPE_LABEL.get(row.get("target_dtype"), str(row.get("target_dtype")))
    prefix = ""
    if include_model:
        model = str(row.get("target_model", ""))
        prefix = f"{model.split('/')[-1]}\n"
    method = row.get("spec_method", "none")
    if method == "none":
        return f"{prefix}{dtype}, no spec"
    gamma = row.get("num_speculative_tokens")
    gamma_txt = f", γ={int(gamma)}" if pd.notna(gamma) else ""
    return f"{prefix}{dtype}, {method}{gamma_txt}"


def select_model(table: pd.DataFrame, target_model: str | None = None) -> pd.DataFrame:
    """Restrict a table to one target model.

    Figures that put precision or batch size on the x axis assume a single model; two
    models sharing an x position would either collide or be silently averaged, and
    averaging two different models' speedups is not a quantity that means anything. So
    the ambiguity is raised rather than resolved.
    """
    if "target_model" not in table.columns:
        return table
    models = sorted(table["target_model"].dropna().unique())
    if target_model is not None:
        if target_model not in models:
            raise ValueError(f"target_model {target_model!r} not present; have {models}")
        return table[table["target_model"] == target_model].copy()
    if len(models) > 1:
        raise ValueError(
            f"this figure covers one target model but the frame has {len(models)}: "
            f"{models}. Pass target_model=... to choose; combining them would average "
            "quantities that are not comparable."
        )
    return table.copy()
