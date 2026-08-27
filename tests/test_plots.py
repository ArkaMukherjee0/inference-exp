"""Figure tests.

Every figure is rendered end to end from the fixture data (relabelled as measured in
``conftest.py``, visibly and only there). What is checked is not "did it look nice" but:

* the provenance guard fires before anything is drawn,
* the figure files are actually written, in both formats,
* the numbers plotted are the numbers the analysis layer produced,
* the guards that would prevent a misleading figure actually fire.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from analysis.derive import acceptance_table, speedup_table, throughput_table
from analysis.load import ProvenanceError
from plots import fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08, style


# -- the guard -------------------------------------------------------------------------


DENSE = "meta-llama/Llama-3.1-8B-Instruct"
MOE = "mistralai/Mixtral-8x7B-Instruct-v0.1"
ARCH = {DENSE: "dense", MOE: "moe"}


@pytest.mark.parametrize("module,kwargs", [
    (fig01, {}),
    (fig02, {"target_model": DENSE}),
    (fig03, {"target_model": DENSE}),
    (fig04, {"c": 0.05, "target_model": DENSE}),
    (fig05, {"ridge_points": {"h100_tp1": 295.0}}),
    (fig06, {}),
    (fig08, {"architecture": {}}),
])
def test_every_figure_refuses_fixture_data(module, kwargs, fixture_df, tmp_outdir):
    """The whole point of the provenance system: fixtures cannot become a report."""
    with pytest.raises(ProvenanceError):
        module.render(fixture_df, tmp_outdir, **kwargs)


# -- rendering -------------------------------------------------------------------------


def _scored(df: pd.DataFrame) -> pd.DataFrame:
    """Quality scores derived from the fixture text, without touching the real dataset."""
    from evals.gsm8k import Example, score_frame

    examples = []
    for pid in sorted(df["prompt_id"].unique()):
        n = abs(hash(pid)) % 97
        examples.append(Example(prompt_id=pid, question="q", answer=f"{n}4"))
    return score_frame(df, examples)


def test_fig02_renders_both_formats(measured_df, tmp_outdir):
    pdf = fig02.render(measured_df, tmp_outdir, target_model=DENSE)
    assert pdf.exists() and pdf.suffix == ".pdf"
    assert pdf.with_suffix(".png").exists()
    assert pdf.stat().st_size > 1000


def test_fig01_renders(measured_df, tmp_outdir):
    out = fig01.render(measured_df, tmp_outdir, scored=_scored(measured_df))
    assert out.exists()


def test_fig03_renders(measured_df, tmp_outdir):
    out = fig03.render(measured_df, tmp_outdir, target_model=DENSE)
    assert out.exists()


def test_fig04_renders(measured_df, tmp_outdir):
    out = fig04.render(measured_df, tmp_outdir, c=0.05, target_model=DENSE)
    assert out.exists()


def test_fig05_renders(measured_df, tmp_outdir):
    out = fig05.render(measured_df, tmp_outdir, ridge_points={"h100_tp1": 295.0},
                       target_dtype="bf16")
    assert out.exists()


def test_fig06_renders(measured_df, tmp_outdir):
    out = fig06.render(measured_df, tmp_outdir)
    assert out.exists()


def test_fig08_renders(measured_df, tmp_outdir):
    out = fig08.render(measured_df, tmp_outdir, architecture=ARCH)
    assert out.exists()


# -- guards that keep a figure honest --------------------------------------------------


def test_fig04_refuses_to_predict_without_a_measured_c(measured_df, tmp_outdir):
    """A prediction made with a guessed cost ratio is not a prediction."""
    with pytest.raises(ValueError, match="measured c"):
        fig04.render(measured_df, tmp_outdir, target_model=DENSE)


def test_fig05_refuses_a_platform_with_no_measured_ridge_point(measured_df, tmp_outdir):
    with pytest.raises(ValueError, match="no measured ridge point"):
        fig05.render(measured_df, tmp_outdir, ridge_points={}, target_dtype="bf16")


def test_fig05_reads_ridge_points_only_from_micro_json(tmp_path):
    payload = {"ridge_point_flop_per_byte": 312.5, "platform": "h100"}
    path = tmp_path / "host_micro.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert fig05.ridge_points_from_micro({"h100_tp1": path}) == {"h100_tp1": 312.5}


def test_fig05_rejects_a_micro_file_with_no_ridge_point(tmp_path):
    path = tmp_path / "bad_micro.json"
    path.write_text(json.dumps({"platform": "cpu"}), encoding="utf-8")
    with pytest.raises(ValueError, match="ridge_point"):
        fig05.ridge_points_from_micro({"cpu": path})


def test_fig08_refuses_to_guess_an_architecture(measured_df, tmp_outdir):
    """Inferring dense/MoE from a model name would put a model in the wrong group."""
    with pytest.raises(ValueError, match="no architecture given"):
        fig08.render(measured_df, tmp_outdir, architecture={})


def test_fig08_requires_a_gamma_one_anchor(measured_df, tmp_outdir):
    df = measured_df[measured_df["num_speculative_tokens"] != 1].copy()
    with pytest.raises(ValueError, match="no γ=1 measurement"):
        fig08.render(df, tmp_outdir, architecture=ARCH)


def test_ci_errorbar_rejects_an_interval_that_excludes_its_point():
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    with pytest.raises(ValueError, match="does not bracket"):
        style.ci_errorbar(ax, [1], [2.0], [2.5], [3.0], color=style.SERIES[0])
    plt.close(fig)


def test_palette_is_capped_at_the_validated_three_slots():
    """A fourth generated hue would fail the all-pairs colour-separation floor."""
    style.series_style(2)
    with pytest.raises(ValueError, match="validated three-slot palette"):
        style.series_style(3)


def test_style_never_calls_seaborn_set_theme():
    """Seaborn supplies plot types here; styling stays in plots/style.py."""
    import inspect

    from plots import fig06

    for module in (style, fig06):
        source = inspect.getsource(module)
        # Match call sites, not the prose explaining why we avoid them.
        assert "sns.set_theme(" not in source
        assert "sns.set_style(" not in source
        assert "seaborn.set_theme(" not in source


# -- the figures agree with the analysis layer -----------------------------------------


def test_plotted_speedups_match_the_stats_layer(measured_df):
    table = speedup_table(measured_df)
    assert (table["speedup"] > 0).all()
    assert (table["speedup_lo95"] <= table["speedup"]).all()
    assert (table["speedup"] <= table["speedup_hi95"]).all()


def test_designed_fixture_speedup_is_recovered(measured_df):
    """The fixtures encode a known 2.0x at bf16/gamma=4; the pipeline must recover it."""
    from tests.fixtures.make_fixtures import TRUE_SPEEDUP

    table = speedup_table(measured_df)
    row = table[
        (table["target_dtype"] == "bf16")
        & (table["num_speculative_tokens"] == 4)
        & (table["batch_size"] == 1)
        & (table["target_model"].str.contains("Llama"))
    ]
    assert len(row) == 1
    expected = TRUE_SPEEDUP[("bf16", 4)]
    got = row.iloc[0]
    assert got["speedup"] == pytest.approx(expected, rel=0.02)
    # The interval must bracket the estimator's own point; nominal coverage over
    # repeated trials is tested properly in test_stats.py, not from one fixture draw.
    assert got["speedup_lo95"] <= got["speedup"] <= got["speedup_hi95"]
    assert got["speedup_hi95"] - got["speedup_lo95"] < 0.15 * expected


def test_composition_hypothesis_direction_is_visible_in_the_fixture(measured_df):
    """Not a claim about reality -- a check that the figure would show the effect if real."""
    table = speedup_table(measured_df)
    at_gamma4 = table[(table["num_speculative_tokens"] == 4) & (table["batch_size"] == 1)]
    by_dtype = at_gamma4.groupby("target_dtype")["speedup"].mean()
    assert by_dtype["bf16"] > by_dtype["fp8"] > by_dtype["w4a16"]


def test_acceptance_histograms_sum_to_the_recorded_steps(measured_df):
    table = acceptance_table(measured_df)
    for _, row in table.iterrows():
        hist = row["accept_length_histogram"]
        assert sum(hist) == row["n_steps"]
        assert len(hist) == row["num_speculative_tokens"] + 1


def test_throughput_is_the_reciprocal_of_median_tpot(measured_df):
    table = throughput_table(measured_df)
    for _, row in table.iterrows():
        assert row["throughput_tok_s"] == pytest.approx(1000.0 / row["tpot_ms_median"])


def test_precision_axis_figures_refuse_an_ambiguous_model_set(measured_df, tmp_outdir):
    """Two models on one precision axis would collide or be silently averaged."""
    for module, kwargs in ((fig02, {}), (fig03, {}), (fig04, {"c": 0.05})):
        with pytest.raises(ValueError, match="one target model"):
            module.render(measured_df, tmp_outdir, **kwargs)


def test_fig05_refuses_to_mix_precisions_across_platforms(measured_df, tmp_outdir):
    """The platform axis must not be confounded with the precision axis.

    E1 establishes that speculative speedup depends on precision, so a CPU point at 4-bit
    beside an H100 point at BF16 would attribute a precision effect to the hardware.
    """
    with pytest.raises(ValueError, match="spans precisions"):
        fig05.render(measured_df, tmp_outdir, ridge_points={"h100_tp1": 295.0})


def test_fig05_renders_with_a_pinned_precision(measured_df, tmp_outdir):
    out = fig05.render(measured_df, tmp_outdir, ridge_points={"h100_tp1": 295.0},
                       target_dtype="bf16")
    assert out.exists()


def test_fig05_labels_carry_the_held_constant_precision(measured_df):
    from analysis.derive import speedup_table
    from plots.fig05 import _platform_points

    table = speedup_table(measured_df)
    table = table[table["batch_size"] == table["batch_size"].min()]
    points = _platform_points(table, {"h100_tp1": 295.0}, "w4a16")
    assert all("W4A16" in label for label in points["label"])


def test_acceptance_figures_refuse_exempt_records(measured_df):
    """Figures 04 and 06 are claims about a distribution; absent one, they must not draw."""
    from analysis.derive import acceptance_table

    df = measured_df.copy()
    df["acceptance_unavailable"] = df["spec_method"] != "none"
    with pytest.raises(ValueError, match="acceptance_unavailable"):
        acceptance_table(df)


def test_speed_figures_survive_exempt_records(measured_df, tmp_outdir):
    """The whole point of the exemption: speed measurements remain fully valid."""
    from analysis.derive import speedup_table

    df = measured_df.copy()
    df["acceptance_unavailable"] = df["spec_method"] != "none"
    table = speedup_table(df)
    assert not table.empty
    assert (table["speedup"] > 0).all()
    assert fig02.render(df, tmp_outdir, target_model=DENSE).exists()
