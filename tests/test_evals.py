"""Evaluation tests: exact-match extraction, identity checking, perplexity config."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evals.gsm8k import (
    Example,
    extract_gold,
    extract_prediction,
    score_example,
    score_frame,
)
from evals.identity import (
    IdentityReport,
    assert_identity,
    compare_conditions,
    first_divergence,
    side_by_side,
)
from evals.perplexity import PPLConfig, PPLResult, paired_window_nll


# -- GSM8K scoring ---------------------------------------------------------------------


def test_extract_gold_from_reference():
    assert extract_gold("She sold 48 + 24 = 72 clips.\n#### 72") == "72"
    assert extract_gold("#### 1,234") == "1234"


def test_extract_gold_raises_without_marker():
    with pytest.raises(ValueError, match="####"):
        extract_gold("no marker here")


@pytest.mark.parametrize("completion,expected", [
    ("The answer is 72.", "72"),
    ("... so 48 + 24 = 72", "72"),
    ("The answer is $1,234.00", "1234"),
    ("Answer: -5", "-5"),
    ("The result is 3.50", "3.5"),
    ("no numbers at all", None),
])
def test_extract_prediction(completion, expected):
    assert extract_prediction(completion) == expected


def test_scoring_is_the_last_number():
    """Under ignore_eos the text runs past the answer; the rule must be stated and stable."""
    assert score_example("first 12, then the answer is 72", "72") == 1
    assert score_example("the answer is 72, and then 99", "72") == 0


def test_comma_and_currency_normalization():
    assert score_example("The answer is $1,234", "1234") == 1


def test_negative_zero_matches_zero():
    assert score_example("the answer is -0", "0") == 1


def test_score_frame_emits_per_example_vectors():
    """Aggregate accuracy cannot be paired; the per-example vector is the requirement."""
    examples = [Example(f"p{i}", "q", "72") for i in range(4)]
    df = pd.DataFrame([
        {"condition_id": "c1", "prompt_id": f"p{i}", "repeat_idx": r,
         "output_text": "the answer is 72" if i % 2 == 0 else "the answer is 99"}
        for i in range(4) for r in range(2)
    ])
    scored = score_frame(df, examples)
    assert len(scored) == 4                       # one row per (condition, prompt)
    assert set(scored.columns) >= {"condition_id", "prompt_id", "em"}
    assert scored["em"].tolist() == [1, 0, 1, 0]


def test_score_frame_flags_nondeterministic_repeats():
    """At temperature 0 two repeats of one prompt must score alike; disagreement is a bug."""
    examples = [Example("p0", "q", "72")]
    df = pd.DataFrame([
        {"condition_id": "c1", "prompt_id": "p0", "repeat_idx": 0,
         "output_text": "the answer is 72"},
        {"condition_id": "c1", "prompt_id": "p0", "repeat_idx": 1,
         "output_text": "the answer is 99"},
    ])
    with pytest.raises(ValueError, match="scored differently across"):
        score_frame(df, examples)


def test_score_frame_rejects_prompts_outside_the_frozen_subset():
    examples = [Example("p0", "q", "72")]
    df = pd.DataFrame([{"condition_id": "c1", "prompt_id": "rogue", "repeat_idx": 0,
                        "output_text": "72"}])
    with pytest.raises(ValueError, match="not in the frozen subset"):
        score_frame(df, examples)


# -- identity --------------------------------------------------------------------------


def test_first_divergence_index():
    assert first_divergence("abcdef", "abcdef") is None
    assert first_divergence("abcdef", "abcXef") == 3
    assert first_divergence("abc", "abcdef") == 3


def _identity_frame(candidate_text: dict[str, str]) -> pd.DataFrame:
    rows = []
    for pid in ["p0", "p1", "p2", "p3"]:
        for repeat in (0, 1):
            rows.append({"condition_id": "base", "prompt_id": pid, "repeat_idx": repeat,
                         "output_text": "the answer is 72"})
            rows.append({"condition_id": "spec", "prompt_id": pid, "repeat_idx": repeat,
                         "output_text": candidate_text.get(pid, "the answer is 72")})
    return pd.DataFrame(rows)


def test_identical_outputs_report_full_identity():
    report = compare_conditions(_identity_frame({}), condition_id="spec",
                                baseline_condition_id="base")
    assert report.n_compared == 8
    assert report.identity_rate == 1.0
    assert not report.divergences


def test_corrupted_output_gives_the_right_divergence_index():
    """A deliberately corrupted output must point at the right position, not just 'differs'."""
    corrupted = "the answer is 79"           # differs at index 15
    report = compare_conditions(_identity_frame({"p2": corrupted}),
                                condition_id="spec", baseline_condition_id="base")
    assert report.n_compared == 8
    assert len(report.divergences) == 2       # both repeats of p2
    for d in report.divergences:
        assert d.prompt_id == "p2"
        assert d.first_divergent_char == 15


def test_identity_is_keyed_not_positional():
    """Interleaved logs are out of order; a positional zip would compare the wrong pairs."""
    df = _identity_frame({})
    shuffled = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
    report = compare_conditions(shuffled, condition_id="spec", baseline_condition_id="base")
    assert report.identity_rate == 1.0


def test_partial_coverage_is_refused():
    df = _identity_frame({})
    df = df.drop(df[(df["condition_id"] == "spec") & (df["prompt_id"] == "p3")].index)
    with pytest.raises(ValueError, match="coverage differs"):
        compare_conditions(df, condition_id="spec", baseline_condition_id="base")


def test_assert_identity_gates_on_the_threshold():
    good = IdentityReport("spec", "base", n_compared=100, n_identical=97, divergences=[])
    assert_identity(good, min_rate=0.95)

    bad = IdentityReport("spec", "base", n_compared=100, n_identical=80, divergences=[])
    with pytest.raises(AssertionError, match="below the 95% threshold"):
        assert_identity(bad, min_rate=0.95)


def test_side_by_side_marks_the_divergence():
    report = compare_conditions(_identity_frame({"p1": "the answer is 79"}),
                                condition_id="spec", baseline_condition_id="base")
    md = side_by_side(report)
    assert "DIVERGES HERE" in md
    assert "Identity rate" in md


# -- perplexity ------------------------------------------------------------------------


def test_ppl_config_rejects_stride_larger_than_window():
    """Windows would skip tokens, so the reported perplexity would cover part of the text."""
    with pytest.raises(ValueError, match="skip tokens"):
        PPLConfig(max_length=512, stride=1024)


def test_ppl_config_rejects_nonsense():
    with pytest.raises(ValueError):
        PPLConfig(max_length=512, stride=0)


def _ppl(config: dict, nll: list[float]) -> PPLResult:
    return PPLResult(perplexity=float(np.exp(np.mean(nll))), mean_nll=float(np.mean(nll)),
                     n_tokens=len(nll) * 10, n_windows=len(nll), config=config,
                     window_nll=nll)


def test_paired_window_nll_requires_identical_configs():
    """Perplexity is not comparable across strides; pairing across two is meaningless."""
    a = _ppl({"max_length": 512, "stride": 256}, [2.0, 2.1, 2.2])
    b = _ppl({"max_length": 512, "stride": 128}, [2.0, 2.1, 2.2])
    with pytest.raises(ValueError, match="not comparable"):
        paired_window_nll(a, b)


def test_paired_window_nll_aligns_equal_configs():
    cfg = {"max_length": 512, "stride": 256}
    a = _ppl(cfg, [2.0, 2.1, 2.2])
    b = _ppl(cfg, [2.05, 2.2, 2.25])
    av, bv = paired_window_nll(a, b)
    assert av.shape == bv.shape == (3,)
    assert np.all(bv > av)


# -- dataset identifiers ---------------------------------------------------------------


def test_dataset_ids_are_namespace_qualified():
    """Bare 'canonical' dataset aliases no longer resolve.

    The legacy ids (``gsm8k``, ``wikitext``) were moved under real namespaces, and
    current huggingface_hub rejects an unqualified id: it builds
    ``hf://datasets/gsm8k@.../...`` and raises HfUriError because the repo id is not
    ``namespace/name``. This failed at the first line of a sweep, after the model had
    already loaded.
    """
    import inspect

    from evals import gsm8k, perplexity

    assert "/" in gsm8k.GSM8K_DATASET, gsm8k.GSM8K_DATASET
    assert "/" in perplexity.PPLConfig(max_length=512, stride=256).dataset

    # And no bare literal survives at a call site.
    for module in (gsm8k, perplexity):
        source = inspect.getsource(module)
        assert 'load_dataset("gsm8k"' not in source
        assert 'load_dataset("wikitext"' not in source


def test_frozen_subset_records_the_dataset_it_indexes():
    """The ids are positional, so they only mean anything against one source repo."""
    import json

    from evals.gsm8k import SUBSET_PATH

    if not SUBSET_PATH.exists():
        pytest.skip("frozen subset not built on this machine")
    meta = json.loads(SUBSET_PATH.read_text(encoding="utf-8"))
    assert "/" in meta["dataset"]
    assert meta["n"] == len(meta["ids"])
