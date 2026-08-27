"""Schema tests.

These are the tests that fail when a measurement is subtly wrong, rather than when the
code fails to run. Each one corresponds to a documented way a study of this kind gets
destroyed.
"""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from core.schema import (
    REQUIRED_FIELDS,
    RunConfig,
    append_record,
    read_records,
    validate_record,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def _valid_record(**overrides) -> dict:
    rec = {
        "run_id": "abc123", "condition_id": "deadbeef1234", "provenance": "measured",
        "timestamp": "2026-08-27T10:00:00.000+00:00", "hostname": "h100-a",
        "platform": "h100", "stack": "vllm", "stack_version": "vllm==0.8.0",
        "driver": "driver=550.54", "target_model": "org/model-8b", "target_dtype": "bf16",
        "draft_model": None, "spec_method": "none", "num_speculative_tokens": None,
        "gamma_schedule": "constant", "tensor_parallel_size": 1,
        "draft_tensor_parallel_size": 1, "nccl_p2p_disabled": False, "batch_size": 1,
        "prompt_id": "gsm8k-test-1", "prompt_tokens": 88, "max_tokens": 256,
        "ignore_eos": True, "temperature": 0.0, "seed": 42, "repeat_idx": 0,
        "is_warmup": False, "ttft_ms": 40.0, "total_ms": 40.0 + 20.0 * 255,
        "tpot_ms": 20.0, "output_tokens": 256, "accepted_tokens": None,
        "draft_tokens_proposed": None, "acceptance_rate": None,
        "mean_accept_length": None, "accept_length_histogram": [],
        "clocks_sm_mhz": 1755.0, "power_draw_w": 300.0, "output_text": "the answer is 72",
    }
    rec.update(overrides)
    return rec


def test_valid_record_passes():
    validate_record(_valid_record())


@pytest.mark.parametrize("field", REQUIRED_FIELDS)
def test_missing_required_field_raises(field):
    rec = _valid_record()
    del rec[field]
    with pytest.raises(ValueError, match=field):
        validate_record(rec)


def test_ignore_eos_short_output_raises():
    """The single most important validation: a run that stopped early is not comparable.

    Without this, a 'faster' condition can simply be one that emitted fewer tokens, and
    the speedup column silently becomes a measure of verbosity.
    """
    rec = _valid_record(output_tokens=200)
    rec["tpot_ms"] = (rec["total_ms"] - rec["ttft_ms"]) / 199
    with pytest.raises(ValueError, match="output_tokens"):
        validate_record(rec)


def test_ignore_eos_false_permits_short_output():
    rec = _valid_record(ignore_eos=False, output_tokens=200)
    rec["tpot_ms"] = (rec["total_ms"] - rec["ttft_ms"]) / 199
    validate_record(rec)


def test_tpot_must_be_rederivable():
    """A hand-supplied tpot that disagrees with the endpoints is rejected.

    Catches a runner that computed time-per-token including prefill, which would track
    prompt length instead of decoding speed.
    """
    rec = _valid_record(tpot_ms=18.0)
    with pytest.raises(ValueError, match="tpot_ms"):
        validate_record(rec)


def test_tpot_including_prefill_is_rejected():
    rec = _valid_record()
    rec["tpot_ms"] = rec["total_ms"] / rec["output_tokens"]  # the classic wrong formula
    with pytest.raises(ValueError, match="tpot_ms"):
        validate_record(rec)


def test_single_token_output_raises():
    rec = _valid_record(max_tokens=1, output_tokens=1, tpot_ms=1.0)
    with pytest.raises(ValueError):
        validate_record(rec)


def test_non_applicable_fields_must_be_null_not_zero():
    """`0` and `null` are different claims: one says 'measured zero', the other 'n/a'."""
    rec = _valid_record(accepted_tokens=0)
    with pytest.raises(ValueError, match="accepted_tokens"):
        validate_record(rec)


def test_speculative_record_valid():
    rec = _spec_record()
    validate_record(rec)


def _spec_record(**overrides) -> dict:
    hist = [4, 6, 10, 12, 18]           # gamma = 4
    steps = sum(hist)
    accepted = sum(k * n for k, n in enumerate(hist))
    proposed = 4 * steps
    rec = _valid_record(
        spec_method="draft_model", num_speculative_tokens=4,
        draft_model="org/model-1b", accepted_tokens=accepted,
        draft_tokens_proposed=proposed, acceptance_rate=accepted / proposed,
        mean_accept_length=accepted / steps + 1.0, accept_length_histogram=hist,
    )
    rec.update(overrides)
    return rec


def test_acceptance_rate_must_match_counts():
    rec = _spec_record(acceptance_rate=0.99)
    with pytest.raises(ValueError, match="acceptance_rate"):
        validate_record(rec)


def test_accepted_cannot_exceed_proposed():
    rec = _spec_record()
    rec["accepted_tokens"] = rec["draft_tokens_proposed"] + 1
    rec["acceptance_rate"] = rec["accepted_tokens"] / rec["draft_tokens_proposed"]
    with pytest.raises(ValueError):
        validate_record(rec)


def test_empty_histogram_on_speculative_condition_raises():
    """An empty histogram means acceptance extraction failed silently."""
    rec = _spec_record(accept_length_histogram=[])
    with pytest.raises(ValueError, match="accept_length_histogram"):
        validate_record(rec)


def test_histogram_must_be_empty_when_not_speculative():
    rec = _valid_record(accept_length_histogram=[1, 2])
    with pytest.raises(ValueError, match="accept_length_histogram"):
        validate_record(rec)


def test_unknown_field_raises():
    """Schema drift between parallel agents starts with one unrecognized key."""
    rec = _valid_record(my_extra_metric=1.0)
    with pytest.raises(ValueError, match="unknown field"):
        validate_record(rec)


def test_zero_clock_rejected_but_null_allowed():
    validate_record(_valid_record(clocks_sm_mhz=None))
    with pytest.raises(ValueError, match="clocks_sm_mhz"):
        validate_record(_valid_record(clocks_sm_mhz=0))


def test_total_before_ttft_raises():
    rec = _valid_record(total_ms=10.0)
    rec["tpot_ms"] = (rec["total_ms"] - rec["ttft_ms"]) / (rec["output_tokens"] - 1)
    with pytest.raises(ValueError):
        validate_record(rec)


# -- condition_id ----------------------------------------------------------------------


def test_condition_id_is_deterministic_within_process():
    a = RunConfig(target_model="m", target_dtype="bf16", stack="vllm", platform="h100")
    b = RunConfig(target_model="m", target_dtype="bf16", stack="vllm", platform="h100")
    assert a.condition_id == b.condition_id


def test_condition_id_ignores_repeats_and_warmup():
    """Resuming a sweep with a different repeat count must not fork the condition."""
    a = RunConfig(target_model="m", target_dtype="bf16", stack="vllm", platform="h100",
                  repeats=5, warmup=3)
    b = RunConfig(target_model="m", target_dtype="bf16", stack="vllm", platform="h100",
                  repeats=9, warmup=1)
    assert a.condition_id == b.condition_id


def test_condition_id_changes_with_every_measurement_field():
    base = RunConfig(target_model="m", target_dtype="bf16", stack="vllm", platform="h100")
    variants = [
        {"target_dtype": "fp8"},
        {"batch_size": 4},
        {"tensor_parallel_size": 2},
        {"max_tokens": 128},
        {"seed": 43},
        {"nccl_p2p_disabled": True},
        {"spec_method": "ngram", "num_speculative_tokens": 4},
    ]
    for override in variants:
        fields = {"target_model": "m", "target_dtype": "bf16", "stack": "vllm",
                  "platform": "h100", **override}
        assert RunConfig(**fields).condition_id != base.condition_id, override


def test_condition_id_stable_across_processes(tmp_path):
    """hash() is salted per process; this proves we are not using it."""
    script = (
        "import sys; sys.path.insert(0, r'%s');"
        "from core.schema import RunConfig;"
        "print(RunConfig(target_model='m', target_dtype='fp8', stack='vllm',"
        " platform='h100', batch_size=8, spec_method='eagle',"
        " num_speculative_tokens=3, draft_model='d').condition_id)" % REPO_ROOT
    )
    outs = set()
    for _ in range(2):
        proc = subprocess.run([sys.executable, "-c", script], capture_output=True,
                              text=True, check=True, cwd=REPO_ROOT)
        outs.add(proc.stdout.strip())
    assert len(outs) == 1
    assert len(outs.pop()) == 12


def test_runconfig_rejects_gamma_without_method():
    with pytest.raises(ValueError, match="num_speculative_tokens"):
        RunConfig(target_model="m", target_dtype="bf16", stack="vllm", platform="h100",
                  spec_method="none", num_speculative_tokens=4)


def test_runconfig_requires_draft_model_for_draft_method():
    with pytest.raises(ValueError, match="draft_model"):
        RunConfig(target_model="m", target_dtype="bf16", stack="vllm", platform="h100",
                  spec_method="draft_model", num_speculative_tokens=4)


# -- IO --------------------------------------------------------------------------------


def test_append_record_validates_before_writing(tmp_path):
    path = tmp_path / "log.jsonl"
    with pytest.raises(ValueError):
        append_record(path, _valid_record(output_tokens=10))
    assert not path.exists(), "an invalid record must not create or touch the log"


def test_append_and_read_roundtrip(tmp_path):
    path = tmp_path / "log.jsonl"
    recs = [_valid_record(run_id=f"r{i}", prompt_id=f"p{i}") for i in range(3)]
    for rec in recs:
        append_record(path, rec)
    loaded = read_records(path)
    assert [r["run_id"] for r in loaded] == ["r0", "r1", "r2"]


def test_read_records_reports_offending_line(tmp_path):
    path = tmp_path / "log.jsonl"
    append_record(path, _valid_record())
    with path.open("a", encoding="utf-8") as fh:
        bad = _valid_record(run_id="bad")
        bad["output_tokens"] = 9
        fh.write(json.dumps(bad) + "\n")
    with pytest.raises(ValueError, match=r":2:"):
        read_records(path)


def test_multiline_output_text_stays_one_jsonl_line(tmp_path):
    path = tmp_path / "log.jsonl"
    append_record(path, _valid_record(output_text="line one\nline two\n"))
    assert len(path.read_text(encoding="utf-8").strip().splitlines()) == 1
