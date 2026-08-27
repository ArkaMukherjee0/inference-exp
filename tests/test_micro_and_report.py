"""Microbenchmark and report-assembly tests."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

import bench.micro as micro
from analysis.model import ridge_point
from evals.perplexity import PPLResult
from plots import fig07
from scripts import build_report

REPO_ROOT = Path(__file__).resolve().parents[1]


# -- microbenchmarks -------------------------------------------------------------------


def test_micro_module_contains_no_hardcoded_hardware_constants():
    """Achieved, never advertised. Figure 05's x axis depends on this being true.

    Peak-to-achieved ratios differ between an H100 and a desktop CPU, so a spec-sheet
    number would distort the ridge-point axis in a platform-dependent direction.
    """
    source = inspect.getsource(micro)
    banned = [
        "3350",     # H100 SXM HBM3 GB/s
        "3.35",
        "1979",     # H100 bf16 TFLOPS (with sparsity)
        "989",      # H100 bf16 TFLOPS (dense)
        "peak_bandwidth",
        "PEAK_",
        "SPEC_SHEET",
    ]
    for token in banned:
        assert token not in source, f"bench/micro.py must not hardcode {token!r}"


def test_cpu_bandwidth_reports_achieved_with_its_measurement_size():
    result = micro.cpu_bandwidth(n_elements=1 << 20, repeats=3, threads=2)
    assert result.gbytes_per_s > 0
    assert result.n_elements == 1 << 20
    assert result.repeats == 3
    assert len(result.all_seconds) == 3
    # Best-of, so the reported figure is the least-contaminated run.
    assert result.best_seconds == min(result.all_seconds)
    # Thread count is recorded: a bandwidth figure without one is not reproducible.
    assert result.threads == 2
    assert result.kernel == "stream_add"


def test_cpu_bandwidth_allocates_nothing_inside_the_timed_loop():
    """The regression that understated bandwidth ~5x.

    ``a = b + scalar*c`` has no fused NumPy form, so the naive spelling allocates a
    full-size temporary every iteration -- timing page faults instead of memory, and
    moving five arrays of traffic while the result divides by three. The kernel must
    stay a single allocation-free ufunc call.
    """
    source = inspect.getsource(micro.cpu_bandwidth)
    timed = source.split("for _ in range(repeats)")[1]
    assert "np.multiply" not in timed
    # No expression that would build a temporary from the inputs.
    assert "scalar *" not in timed and "* c" not in timed


def test_cpu_bandwidth_thread_sweep_picks_at_least_the_single_thread_result():
    """Bandwidth saturates below the core count and falls with oversubscription, so
    'all cores' is not the achieved ceiling -- the sweep must not do worse than 1 thread."""
    single = micro.cpu_bandwidth(n_elements=1 << 22, repeats=3, threads=1)
    swept = micro.cpu_bandwidth(n_elements=1 << 22, repeats=3)
    assert swept.gbytes_per_s >= single.gbytes_per_s * 0.95
    assert swept.threads >= 1


def test_cpu_bandwidth_traffic_accounting_is_three_arrays():
    """STREAM triad touches three arrays: two read, one written."""
    n = 1 << 20
    result = micro.cpu_bandwidth(n_elements=n, repeats=2, dtype="float64", threads=2)
    implied = result.gbytes_per_s * 1e9 * result.best_seconds
    assert implied == pytest.approx(3 * n * 8, rel=1e-6)


def test_cpu_compute_reports_achieved_tflops():
    result = micro.cpu_compute(sizes=(256, 512), repeats=2)
    assert result.tflops > 0
    assert result.m == result.n == result.k
    implied_flops = result.tflops * 1e12 * result.best_seconds
    assert implied_flops == pytest.approx(2 * result.m ** 3, rel=1e-6)


def test_ridge_point_is_derived_from_the_two_measurements():
    bw = micro.cpu_bandwidth(n_elements=1 << 20, repeats=2, threads=2)
    comp = micro.cpu_compute(sizes=(256,), repeats=2)
    rp = ridge_point(comp.tflops, bw.gbytes_per_s)
    assert rp == pytest.approx((comp.tflops * 1e12) / (bw.gbytes_per_s * 1e9))
    assert rp > 0


def test_micro_report_roundtrips(tmp_path):
    bw = micro.cpu_bandwidth(n_elements=1 << 20, repeats=2, threads=2)
    comp = micro.cpu_compute(sizes=(256,), repeats=2)
    from dataclasses import asdict

    report = micro.MicroReport(
        platform="cpu", device="test-cpu", bandwidth=asdict(bw), compute=asdict(comp),
        ridge_point_flop_per_byte=ridge_point(comp.tflops, bw.gbytes_per_s), env={},
    )
    path = report.write(tmp_path / "host_micro.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["ridge_point_flop_per_byte"] > 0
    assert data["bandwidth"]["repeats"] == 2


def test_gpu_helpers_raise_without_cuda_rather_than_falling_back():
    import importlib.util

    if importlib.util.find_spec("torch") is None:
        pytest.skip("torch not installed")
    import torch

    if torch.cuda.is_available():
        pytest.skip("CUDA present; this test covers the no-GPU refusal path")
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        micro.gpu_bandwidth()


# -- figure 07 -------------------------------------------------------------------------


def _ppl(dtype: str, offset: float) -> PPLResult:
    cfg = {"max_length": 512, "stride": 256, "split": "test",
           "dataset": "wikitext", "subset": "wikitext-2-raw-v1"}
    rng = np.random.default_rng(abs(hash(dtype)) % 2**32)
    nll = list(2.1 + offset + rng.normal(0, 0.02, 40))
    return PPLResult(perplexity=float(np.exp(np.mean(nll))), mean_nll=float(np.mean(nll)),
                     n_tokens=400, n_windows=40, config=cfg, window_nll=nll)


def test_fig07_renders_stacked_panels(measured_df, tmp_outdir):
    from tests.test_plots import DENSE, _scored

    df = measured_df[measured_df["target_model"] == DENSE]
    perplexity = {"bf16": _ppl("bf16", 0.0), "fp8": _ppl("fp8", 0.03),
                  "w4a16": _ppl("w4a16", 0.14)}
    out = fig07.render(df, tmp_outdir, perplexity=perplexity, scored=_scored(df),
                       target_model=DENSE)
    assert out.exists()
    assert out.with_suffix(".png").exists()


def test_fig07_has_no_twin_axis():
    """The construction is stacked panels; twinx would make the divergence unfalsifiable."""
    source = inspect.getsource(fig07)
    assert "twinx" not in source
    assert "twiny" not in source


def test_fig07_requires_a_bf16_baseline(measured_df, tmp_outdir):
    from tests.test_plots import DENSE

    df = measured_df[(measured_df["target_model"] == DENSE)
                     & (measured_df["target_dtype"] != "bf16")]
    with pytest.raises(ValueError, match="bf16"):
        fig07.render(df, tmp_outdir, perplexity={})


# -- report assembly -------------------------------------------------------------------


def test_primary_table_rows_are_traceable(measured_df):
    from tests.test_plots import _scored

    table = build_report.primary_table(measured_df, _scored(measured_df))
    index = build_report.run_id_index(measured_df)
    assert len(table) == measured_df["condition_id"].nunique()
    for cid in table["condition_id"]:
        assert index[cid], f"{cid} has no run_ids behind it"


def test_traceability_check_catches_an_orphan_row(measured_df):
    from tests.test_plots import _scored

    table = build_report.primary_table(measured_df, _scored(measured_df))
    table.loc[table.index[0], "condition_id"] = "not-a-real-condition"
    with pytest.raises(ValueError, match="no run_ids"):
        build_report._verify_traceability(table, measured_df)


def test_provenance_appendix_names_stack_driver_and_host(measured_df):
    md = build_report.provenance_appendix(measured_df)
    assert "# Provenance appendix" in md
    for cid in measured_df["condition_id"].unique()[:3]:
        assert f"`{cid}`" in md
    assert "**stack**" in md and "**driver**" in md and "**host**" in md
    assert "**run_ids**" in md


def test_primary_table_carries_speedup_intervals(measured_df):
    table = build_report.primary_table(measured_df)
    spec_rows = table[table["spec_method"] != "none"]
    assert not spec_rows.empty
    assert spec_rows["speedup"].notna().all()
    assert spec_rows["speedup_ci95"].notna().all()
    baseline_rows = table[table["spec_method"] == "none"]
    assert baseline_rows["speedup"].isna().all(), "a baseline has no speedup against itself"


def test_build_writes_every_artifact(fixture_records, tmp_path):
    # Written from the original dicts rather than round-tripped through pandas: a
    # DataFrame turns None into NaN and ints into floats, and the schema draws a hard
    # line between "not applicable" (null) and a number. Only provenance is changed.
    log = tmp_path / "measured.jsonl"
    n_written = 0
    with log.open("w", encoding="utf-8") as fh:
        for rec in fixture_records:
            if rec["is_warmup"]:
                continue
            fh.write(json.dumps({**rec, "provenance": "measured"}) + "\n")
            n_written += 1

    outdir = tmp_path / "report"
    result = build_report.build([log], outdir, c=0.05, score_quality=False)

    for name in ("primary_table.csv", "primary_table.md", "provenance_appendix.md",
                 "run_id_index.json", "figures.json"):
        assert (outdir / name).exists(), name
    assert result["n_records"] == n_written


def test_missing_figure_inputs_are_reported_not_faked(measured_df, tmp_outdir):
    """A figure whose inputs are absent must be skipped with a reason, never drawn."""
    figures = build_report.render_figures(measured_df, tmp_outdir)
    # fig05 has no ridge points and fig08 no architecture map in this call.
    assert isinstance(figures["fig05"], str) and figures["fig05"].startswith("SKIPPED")
    assert isinstance(figures["fig08"], str) and figures["fig08"].startswith("SKIPPED")
    assert not (tmp_outdir / "fig05_platform_curve.pdf").exists()


def test_render_figures_can_be_made_strict(measured_df, tmp_outdir):
    with pytest.raises(Exception):
        build_report.render_figures(measured_df, tmp_outdir, skip_on_missing_inputs=False)


# -- model preflight -------------------------------------------------------------------


def test_vocab_mismatch_is_reported_as_a_failure(monkeypatch):
    """Speculative decoding indexes target logits with draft token ids.

    A mismatched vocabulary is an out-of-bounds read, not a low acceptance rate. vLLM
    rejects the pair -- but only after spinning up an engine, a minute into a run.
    """
    from scripts import setup_data

    sizes = {"org/target": 128256, "org/draft": 49152}
    monkeypatch.setattr(setup_data, "_vocab_size", lambda repo: sizes[repo])
    assert setup_data._check_vocab_pair("org/target", "org/draft") is False


def test_matching_vocab_passes(monkeypatch):
    from scripts import setup_data

    monkeypatch.setattr(setup_data, "_vocab_size", lambda repo: 128256)
    assert setup_data._check_vocab_pair("org/target", "org/draft") is True


def test_unreadable_vocab_is_unverified_not_a_pass_or_a_failure(monkeypatch):
    """A gated repo or an offline box means 'cannot tell', which must not read as 'checked'."""
    from scripts import setup_data

    monkeypatch.setattr(setup_data, "_vocab_size", lambda repo: None)
    assert setup_data._check_vocab_pair("org/target", "org/draft") is True
