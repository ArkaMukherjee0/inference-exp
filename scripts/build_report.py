"""Assemble the report: primary table, all eight figures, provenance appendix.

The appendix is the point of this script as much as the figures are. Every number in the
primary table resolves to a list of ``run_id``s in the appendix, along with the resolved
config, host, stack version, driver and clock state that produced it. A number that
cannot be traced that way does not get written.

Run:
    python -m scripts.build_report --logs logs/*.jsonl --outdir report
"""

from __future__ import annotations

import argparse
import sys
import json
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.derive import acceptance_table, condition_label, speedup_table, throughput_table
from analysis.load import condition_table, load_runs, require_measured
from core.env import utc_now

REPO_ROOT = Path(__file__).resolve().parent.parent


# --------------------------------------------------------------------------------------
# Primary table
# --------------------------------------------------------------------------------------


def primary_table(df: pd.DataFrame, scored: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per condition: throughput, speedup with CI, acceptance, quality."""
    require_measured(df)
    tput = throughput_table(df).set_index("condition_id")

    try:
        speed = speedup_table(df).set_index("condition_id")
    except ValueError:
        speed = pd.DataFrame()
    try:
        accept = acceptance_table(df).set_index("condition_id")
    except ValueError:
        accept = pd.DataFrame()

    rows = []
    for cid, t in tput.iterrows():
        row: dict[str, Any] = {
            "condition_id": cid,
            "condition": condition_label(t),
            "target_model": t.get("target_model"),
            "precision": t.get("target_dtype"),
            "spec_method": t.get("spec_method"),
            "gamma": t.get("num_speculative_tokens"),
            "batch_size": t.get("batch_size"),
            "tp": t.get("tensor_parallel_size"),
            "tpot_ms_median": round(float(t["tpot_ms_median"]), 4),
            "throughput_tok_s": round(float(t["throughput_tok_s"]), 3),
            "n_prompts": int(t["n_prompts"]),
        }
        if cid in speed.index:
            s = speed.loc[cid]
            row["speedup"] = round(float(s["speedup"]), 4)
            row["speedup_ci95"] = f"[{s['speedup_lo95']:.3f}, {s['speedup_hi95']:.3f}]"
            row["baseline_condition_id"] = s["baseline_condition_id"]
        else:
            row["speedup"] = None
            row["speedup_ci95"] = None
            row["baseline_condition_id"] = None
        if cid in accept.index:
            a = accept.loc[cid]
            row["acceptance_rate"] = round(float(a["acceptance_rate"]), 4)
            row["mean_accept_length"] = round(float(a["mean_accept_length"]), 4)
        else:
            row["acceptance_rate"] = None
            row["mean_accept_length"] = None
        if scored is not None:
            sub = scored[scored["condition_id"] == cid]
            row["gsm8k_em"] = round(float(sub["em"].mean()), 4) if not sub.empty else None
        rows.append(row)

    table = pd.DataFrame(rows)
    return table.sort_values(["precision", "spec_method", "gamma"], na_position="first")


def to_markdown(table: pd.DataFrame) -> str:
    cols = [c for c in table.columns if c != "baseline_condition_id"]
    return table[cols].to_markdown(index=False, floatfmt=".4g")


# --------------------------------------------------------------------------------------
# Provenance appendix
# --------------------------------------------------------------------------------------


def provenance_appendix(df: pd.DataFrame) -> str:
    """Every condition, its resolved config, and the run_ids behind it."""
    table = condition_table(df)
    lines = [
        "# Provenance appendix",
        "",
        f"Generated {utc_now()}.",
        "",
        "Every number in the primary table resolves to the run ids listed here.",
        "",
    ]
    for _, row in table.iterrows():
        lines.append(f"## `{row['condition_id']}`")
        lines.append("")
        lines.append(f"- **model**: `{row.get('target_model')}`"
                     f" (dtype `{row.get('target_dtype')}`)")
        if row.get("draft_model"):
            lines.append(f"- **draft**: `{row.get('draft_model')}`"
                         f" via `{row.get('spec_method')}`, γ={row.get('num_speculative_tokens')}")
        lines.append(f"- **host**: `{row.get('hostname')}` ({row.get('platform')})")
        lines.append(f"- **stack**: `{row.get('stack')}` `{row.get('stack_version')}`")
        lines.append(f"- **driver**: `{row.get('driver')}`")
        lines.append(f"- **batch** {row.get('batch_size')}, **TP** {row.get('tensor_parallel_size')}"
                     f" (draft TP {row.get('draft_tensor_parallel_size')}),"
                     f" **NCCL P2P disabled**: {row.get('nccl_p2p_disabled')}")
        lines.append(f"- **generation**: max_tokens={row.get('max_tokens')}, "
                     f"ignore_eos={row.get('ignore_eos')}, temperature={row.get('temperature')}, "
                     f"seed={row.get('seed')}")
        if "clocks_sm_mhz_median" in row and pd.notna(row["clocks_sm_mhz_median"]):
            lines.append(f"- **median SM clock**: {row['clocks_sm_mhz_median']:.0f} MHz")
        lines.append(f"- **records**: {row['n_records']} over {row['n_prompts']} prompts")
        run_ids = row["run_ids"]
        lines.append(f"- **run_ids** ({len(run_ids)}): "
                     + ", ".join(f"`{r}`" for r in run_ids[:8])
                     + (f" … (+{len(run_ids) - 8} more)" if len(run_ids) > 8 else ""))
        lines.append("")
    return "\n".join(lines)


def run_id_index(df: pd.DataFrame) -> dict[str, list[str]]:
    """condition_id -> every run_id, written as JSON so traceability is machine-checkable."""
    return {
        cid: sorted(group["run_id"].tolist())
        for cid, group in df.groupby("condition_id")
    }


def identity_appendix(df: pd.DataFrame, *, min_rate: float = 0.95) -> str:
    """The §7.4 side-by-side: every speculative condition against its own baseline.

    Speculative decoding is supposed to be distribution-preserving, and at temperature 0
    that means byte-identical output. This is the check that says whether it was, and it
    belongs in the report rather than in a test: a speedup measured against a baseline
    the speculative arm does not reproduce is not a speedup, it is a different model.

    A condition is matched to its baseline on every field except the speculative ones, so
    a spec run is only ever compared against the non-speculative run of the *same* model,
    precision, batch size and parallelism.
    """
    from evals.identity import compare_conditions, side_by_side

    conds = condition_table(df)
    match_on = [c for c in ("target_model", "target_dtype", "batch_size", "stack",
                            "platform", "tensor_parallel_size", "draft_tensor_parallel_size",
                            "nccl_p2p_disabled", "max_tokens", "temperature")
                if c in conds.columns]

    baselines = conds[conds["spec_method"] == "none"]
    blocks: list[str] = ["# Byte-identity against baseline (§7.4)", ""]
    if baselines.empty:
        return "\n".join(blocks + ["No non-speculative baseline condition present."])

    rates: list[str] = []
    for _, cand in conds[conds["spec_method"] != "none"].iterrows():
        peers = baselines
        for col in match_on:
            peers = peers[peers[col] == cand[col]]
        if peers.empty:
            blocks.append(f"- `{cand['condition_id']}`: no matching baseline; not compared.")
            continue
        base_id = peers.iloc[0]["condition_id"]
        try:
            report = compare_conditions(
                df, condition_id=cand["condition_id"], baseline_condition_id=base_id
            )
        except ValueError as exc:
            blocks.append(f"- `{cand['condition_id']}`: not compared ({exc}).")
            continue
        flag = "" if report.identity_rate >= min_rate else "  **BELOW THRESHOLD**"
        rates.append(
            f"| `{cand['condition_id']}` | {condition_label(cand)} | "
            f"{report.identity_rate:.2%} | {report.n_identical}/{report.n_compared} |{flag}"
        )
        blocks.append("")
        blocks.append(side_by_side(report))

    summary = ["| condition | label | identity rate | identical/compared |",
               "|---|---|---|---|", *rates, ""] if rates else []
    return "\n".join(blocks[:2] + summary + blocks[2:])


# --------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------


def _fig04_jobs(df: pd.DataFrame, outdir: Path, fig04, *, c: float | None,
                c_by_pair: dict[tuple[str, str], float] | None) -> list[tuple[str, Any]]:
    """One figure 04 per target checkpoint, each with its own measured c.

    c is a ratio against a *specific* target: the same 1B drafter is relatively cheaper
    against a bf16 8B than against the same model quantized to w4a16, because quantizing
    shortens the target's step and leaves the draft's alone. Drawing all three precisions
    against a single c would put two predicted curves on the wrong cost ratio -- and they
    would look entirely reasonable, which is what makes it worth splitting.

    With no c_by_pair this falls back to the single-c call, so behaviour is unchanged for
    a single-precision log.
    """
    if not c_by_pair:
        return [("fig04", lambda: fig04.render(df, outdir, c=c))]

    models = sorted(df["target_model"].dropna().unique())
    if len(models) <= 1:
        return [("fig04", lambda: fig04.render(df, outdir, c=c, c_by_pair=c_by_pair))]

    jobs: list[tuple[str, Any]] = []
    for model in models:
        dtypes = sorted(df[df["target_model"] == model]["target_dtype"].dropna().unique())
        tag = dtypes[0] if len(dtypes) == 1 else model.split("/")[-1]
        jobs.append((
            f"fig04:{tag}",
            # Bind both loop variables; a bare closure would render the last model N times.
            lambda m=model, s=tag: fig04.render(
                df, outdir, c=c, c_by_pair=c_by_pair, target_model=m, name_suffix=f"_{s}"
            ),
        ))
    return jobs


def render_figures(
    df: pd.DataFrame,
    outdir: Path,
    *,
    scored: pd.DataFrame | None = None,
    c: float | None = None,
    c_by_pair: dict[tuple[str, str], float] | None = None,
    ridge_points: dict[str, float] | None = None,
    perplexity: dict[str, Any] | None = None,
    architecture: dict[str, str] | None = None,
    skip_on_missing_inputs: bool = True,
) -> dict[str, Path | str]:
    """Render every figure the loaded data supports.

    A figure whose inputs are absent is *skipped and reported as skipped*, never drawn
    from a stand-in value. The report says which figures are missing and why.
    """
    from plots import fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08

    outdir.mkdir(parents=True, exist_ok=True)
    results: dict[str, Path | str] = {}

    jobs = [
        ("fig01", lambda: fig01.render(df, outdir, scored=scored)),
        ("fig02", lambda: fig02.render(df, outdir)),
        ("fig03", lambda: fig03.render(df, outdir)),
        *_fig04_jobs(df, outdir, fig04, c=c, c_by_pair=c_by_pair),
        ("fig05", lambda: fig05.render(df, outdir, ridge_points=ridge_points or {})),
        ("fig06", lambda: fig06.render(df, outdir)),
        ("fig07", lambda: fig07.render(df, outdir, perplexity=perplexity or {}, scored=scored)),
        ("fig08", lambda: fig08.render(df, outdir, architecture=architecture or {})),
    ]
    for name, job in jobs:
        try:
            results[name] = job()
        except Exception as exc:  # noqa: BLE001 -- a missing figure is reported, not faked
            if not skip_on_missing_inputs:
                raise
            results[name] = f"SKIPPED: {exc}"
    return results


# --------------------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------------------


def build(
    log_paths: list[str | Path],
    outdir: Path,
    *,
    c: float | None = None,
    c_by_pair: dict[tuple[str, str], float] | None = None,
    ridge_points: dict[str, float] | None = None,
    perplexity: dict[str, Any] | None = None,
    architecture: dict[str, str] | None = None,
    score_quality: bool = True,
) -> dict[str, Any]:
    df = load_runs(log_paths)
    require_measured(df)
    outdir.mkdir(parents=True, exist_ok=True)

    scored = None
    if score_quality:
        try:
            from evals.gsm8k import score_frame

            scored = score_frame(df)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            print(f"quality scoring unavailable ({exc}); tables will omit GSM8K columns.")

    table = primary_table(df, scored)
    table.to_csv(outdir / "primary_table.csv", index=False)
    (outdir / "primary_table.md").write_text(to_markdown(table), encoding="utf-8")

    (outdir / "provenance_appendix.md").write_text(provenance_appendix(df), encoding="utf-8")
    try:
        (outdir / "identity_appendix.md").write_text(identity_appendix(df), encoding="utf-8")
    except (ImportError, KeyError, ValueError) as exc:
        # Reported as absent, never as passing. An identity check that did not run is
        # not an identity check that succeeded.
        (outdir / "identity_appendix.md").write_text(
            f"# Byte-identity against baseline (§7.4)\n\nNOT RUN: {exc}\n", encoding="utf-8"
        )
        print(f"identity appendix unavailable ({exc}).")
    (outdir / "run_id_index.json").write_text(
        json.dumps(run_id_index(df), indent=2), encoding="utf-8"
    )

    figures = render_figures(
        df, outdir, scored=scored, c=c, c_by_pair=c_by_pair, ridge_points=ridge_points,
        perplexity=perplexity, architecture=architecture
    )
    (outdir / "figures.json").write_text(
        json.dumps({k: str(v) for k, v in figures.items()}, indent=2), encoding="utf-8"
    )

    _verify_traceability(table, df)
    return {"table": table, "figures": figures, "n_records": len(df)}


def _verify_traceability(table: pd.DataFrame, df: pd.DataFrame) -> None:
    """Every table row must map to at least one run_id. The acceptance criterion, checked."""
    index = run_id_index(df)
    orphans = [cid for cid in table["condition_id"] if not index.get(cid)]
    if orphans:
        raise ValueError(
            f"{len(orphans)} table row(s) have no run_ids behind them ({orphans[:3]}). "
            "Every number in the report must be traceable to a log line."
        )


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
    ap.add_argument("--logs", nargs="+", required=True)
    ap.add_argument("--outdir", default="report")
    ap.add_argument("--c", type=float, default=None,
                    help="measured draft/target cost ratio for figure 04")
    ap.add_argument("--micro", nargs="*", default=None,
                    help="platform_key=path/to/host_micro.json, for figure 05")
    ap.add_argument("--architecture", nargs="*", default=None,
                    help="model=dense|moe, for figure 08")
    ap.add_argument("--c-by-pair", default=None,
                    help="path to a scripts.measure_c result; gives figure 04 a separate "
                         "measured c per target checkpoint")
    ap.add_argument("--perplexity", nargs="*", default=None,
                    help="target_dtype=path/to/ppl.json, for figure 07 "
                         "(written by scripts.run_perplexity)")
    args = ap.parse_args(argv)

    ridge = None
    if args.micro:
        from plots.fig05 import ridge_points_from_micro

        ridge = ridge_points_from_micro(dict(kv.split("=", 1) for kv in args.micro))

    c_pairs = None
    if args.c_by_pair:
        from scripts.measure_c import load_c_by_pair

        c_pairs = load_c_by_pair(args.c_by_pair)

    ppl = None
    if args.perplexity:
        from evals.perplexity import PPLResult

        ppl = {}
        for kv in args.perplexity:
            dtype, path = kv.split("=", 1)
            ppl[dtype] = PPLResult(**json.loads(Path(path).read_text(encoding="utf-8")))

    arch = dict(kv.split("=", 1) for kv in args.architecture) if args.architecture else None

    out = build(args.logs, Path(args.outdir), c=args.c, c_by_pair=c_pairs,
                ridge_points=ridge, perplexity=ppl, architecture=arch)
    print(f"\n{out['n_records']} records -> {args.outdir}")
    for name, path in out["figures"].items():
        print(f"  {name}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
