"""Loader guards and sweep-config expansion."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from analysis.load import (
    ProvenanceError,
    assert_single_stack,
    baseline_map,
    condition_table,
    load_runs,
    require_measured,
    speed_frame,
)
from core.config import load_sweep

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


# -- the provenance guard --------------------------------------------------------------


def test_require_measured_rejects_fixture_data(fixture_df):
    """The structural reason fixture data cannot reach a figure."""
    with pytest.raises(ProvenanceError, match="not measured"):
        require_measured(fixture_df)


def test_require_measured_rejects_a_single_contaminating_row(measured_df):
    df = measured_df.copy()
    df.loc[df.index[0], "provenance"] = "fixture"
    with pytest.raises(ProvenanceError):
        require_measured(df)


def test_require_measured_passes_and_returns_frame(measured_df):
    assert require_measured(measured_df) is measured_df


def test_every_figure_module_guards_provenance():
    """A figure module that forgot the guard is a figure that can plot fixtures."""
    import inspect

    from plots import fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08

    for module in (fig01, fig02, fig03, fig04, fig05, fig06, fig07, fig08):
        source = inspect.getsource(module.render)
        assert "require_measured" in source, f"{module.__name__}.render lacks the guard"


# -- warmup exclusion ------------------------------------------------------------------


def test_warmups_are_dropped_by_default(fixture_records, tmp_path):
    log = _write(tmp_path, fixture_records)
    df = load_runs(log)
    assert not df["is_warmup"].any()
    assert len(df) < len(fixture_records), "the fixture set should contain warmups to drop"


def test_warmups_are_available_when_explicitly_requested(fixture_records, tmp_path):
    log = _write(tmp_path, fixture_records)
    df = load_runs(log, drop_warmup=False)
    assert df["is_warmup"].any()


def test_warmup_rows_are_genuinely_slower(fixture_records, tmp_path):
    """Confirms the fixture models a real cold-start effect, so the guard has teeth."""
    log = _write(tmp_path, fixture_records)
    df = load_runs(log, drop_warmup=False)
    warm = df[df["is_warmup"]]["tpot_ms"].median()
    steady = df[~df["is_warmup"]]["tpot_ms"].median()
    assert warm > steady * 1.5


# -- HF exclusion ----------------------------------------------------------------------


def test_hf_records_never_reach_a_speed_figure(measured_df):
    df = measured_df.copy()
    hf = df.iloc[[0]].copy()
    hf["stack"] = "hf"
    hf["latency_valid"] = False
    hf["run_id"] = "hf-contaminant"
    df = pd.concat([df, hf], ignore_index=True)

    out = speed_frame(df)
    assert "hf" not in set(out["stack"])
    assert "hf-contaminant" not in set(out["run_id"])


def test_hf_record_marked_latency_valid_true_is_caught(measured_df):
    """Defence in depth: if the flag is wrong, the stack name still stops it."""
    df = measured_df.copy()
    hf = df.iloc[[0]].copy()
    hf["stack"] = "hf"
    hf["latency_valid"] = True
    df = pd.concat([df, hf], ignore_index=True)
    with pytest.raises(ProvenanceError, match="HF"):
        speed_frame(df)


def test_assert_single_stack_rejects_mixed_frames(measured_df):
    df = measured_df.copy()
    df.loc[df.index[0], "stack"] = "llamacpp"
    with pytest.raises(ValueError, match="spans stacks"):
        assert_single_stack(df)


# -- baseline pairing ------------------------------------------------------------------


def test_baseline_map_pairs_within_precision(measured_df):
    axes = ["target_model", "target_dtype", "batch_size", "tensor_parallel_size"]
    pairs = baseline_map(measured_df, axes)
    meta = measured_df.drop_duplicates("condition_id").set_index("condition_id")
    assert pairs
    for opt_id, base_id in pairs.items():
        for axis in axes:
            assert meta.loc[opt_id, axis] == meta.loc[base_id, axis], axis
        assert meta.loc[base_id, "spec_method"] == "none"


def test_baseline_map_raises_when_a_twin_is_missing(measured_df):
    """A speculative condition with no same-precision baseline must not borrow another's."""
    df = measured_df[~(
        (measured_df["target_dtype"] == "fp8") & (measured_df["spec_method"] == "none")
    )].copy()
    with pytest.raises(ValueError, match="no baseline twin"):
        baseline_map(df, ["target_model", "target_dtype", "batch_size"])


def test_condition_table_carries_run_ids(measured_df):
    table = condition_table(measured_df)
    assert table["n_records"].sum() == len(measured_df)
    assert all(len(ids) > 0 for ids in table["run_ids"])


# -- sweep configs ---------------------------------------------------------------------


@pytest.mark.parametrize("name", [
    "instance1_precision_spec.yaml",
    "instance2_batch_drafter.yaml",
    "instance3_scale_tp.yaml",
    "local_cpu.yaml",
    "smoke.yaml",
])
def test_shipped_configs_expand(name):
    cfg = load_sweep(CONFIG_DIR / name)
    assert cfg.conditions
    ids = [c.condition_id for c in cfg.conditions]
    assert len(set(ids)) == len(ids)


def test_instance1_expands_to_the_expected_grid():
    cfg = load_sweep(CONFIG_DIR / "instance1_precision_spec.yaml")
    assert len(cfg.conditions) == 3 * 7           # 3 precisions x 7 spec points
    assert {c.target_dtype for c in cfg.conditions} == {"bf16", "fp8", "w4a16"}
    baselines = [c for c in cfg.conditions if c.spec_method == "none"]
    assert len(baselines) == 3
    assert all(c.num_speculative_tokens is None and c.draft_model is None for c in baselines)


def test_queue_is_interleaved_across_conditions():
    """Consecutive blocks must not all belong to one condition (the drift guard)."""
    cfg = load_sweep(CONFIG_DIR / "instance1_precision_spec.yaml")
    ids = [f"gsm8k-test-{i}" for i in range(cfg.prompts.n)]
    queue = cfg.build_queue(ids)

    blocks = [(r, c.condition_id) for r, c, _ in cfg.condition_visits(queue)]
    rounds = {}
    for round_idx, cid in blocks:
        rounds.setdefault(round_idx, []).append(cid)

    # Every condition appears once per round, and the order rotates between rounds.
    for round_idx, cids in rounds.items():
        assert len(cids) == len(set(cids)) == len(cfg.conditions)
    assert rounds[0] != rounds[1], "condition order must rotate between rounds"


def test_queue_covers_every_cell_exactly_once():
    cfg = load_sweep(CONFIG_DIR / "smoke.yaml")
    ids = [f"gsm8k-test-{i}" for i in range(cfg.prompts.n)]
    queue = cfg.build_queue(ids)
    real = [u for u in queue if not u.is_warmup]
    keys = [u.key for u in real]
    assert len(keys) == len(set(keys))
    expected = len(cfg.conditions) * cfg.prompts.n * max(c.repeats for c in cfg.conditions)
    assert len(real) == expected


def test_warmups_are_scheduled_per_condition_visit():
    cfg = load_sweep(CONFIG_DIR / "smoke.yaml")
    ids = [f"gsm8k-test-{i}" for i in range(cfg.prompts.n)]
    queue = cfg.build_queue(ids)
    for _, condition, block in cfg.condition_visits(queue):
        warm = [u for u in block if u.is_warmup]
        assert len(warm) == condition.warmup
        # Warmups come first inside the block: a fresh process needs a fresh warmup.
        assert all(u.is_warmup for u in block[:condition.warmup])


def test_build_queue_rejects_a_differently_sized_prompt_set():
    cfg = load_sweep(CONFIG_DIR / "smoke.yaml")
    with pytest.raises(ValueError, match="differently-sized"):
        cfg.build_queue([f"p{i}" for i in range(cfg.prompts.n - 1)])


def test_unknown_config_key_raises(tmp_path):
    cfg = yaml.safe_load((CONFIG_DIR / "smoke.yaml").read_text(encoding="utf-8"))
    cfg["typo_axis"] = [1, 2]
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown key"):
        load_sweep(path)


def test_two_axes_setting_the_same_field_raises(tmp_path):
    cfg = yaml.safe_load((CONFIG_DIR / "smoke.yaml").read_text(encoding="utf-8"))
    cfg["axes"]["extra"] = [{"num_speculative_tokens": 2}]
    path = tmp_path / "clash.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    with pytest.raises(ValueError, match="both set"):
        load_sweep(path)


def _write(tmp_path: Path, records: list[dict]) -> Path:
    import json

    path = tmp_path / "log.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return path
