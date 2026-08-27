"""Byte-identity checking against a matched baseline.

Correctly implemented speculative decoding is distribution-preserving: the verification
step accepts or rejects each drafted token in a way that leaves the target model's output
distribution untouched. At temperature 0 that means the text should be identical, byte
for byte, to the non-speculative baseline.

So this is not a quality metric, it is a correctness check on the acceptance logic, and
it runs *before* the real sweeps.

Why the first-divergence index matters
--------------------------------------
An identity *rate* cannot answer the question on its own. Some divergence is expected:
floating-point addition is not associative, the verification path sums logits in a
different order than plain decoding, and at a near-tie between two candidate tokens that
is enough to flip the argmax. That is benign.

What is not benign is a divergence at a position where the target model was confident,
which means tokens are being accepted that the target would have rejected. Both show up
identically in a boolean, and only the position tells them apart -- so this module
reports where, not just whether.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import pandas as pd


@dataclass(frozen=True)
class Divergence:
    prompt_id: str
    repeat_idx: int
    identical: bool
    first_divergent_char: int | None
    first_divergent_token: int | None
    baseline_excerpt: str
    candidate_excerpt: str


@dataclass(frozen=True)
class IdentityReport:
    condition_id: str
    baseline_condition_id: str
    n_compared: int
    n_identical: int
    divergences: list[Divergence]

    @property
    def identity_rate(self) -> float:
        if self.n_compared == 0:
            raise ValueError("no pairs were compared")
        return self.n_identical / self.n_compared

    def summary(self) -> str:
        return (
            f"{self.condition_id} vs {self.baseline_condition_id}: "
            f"{self.n_identical}/{self.n_compared} identical "
            f"({self.identity_rate:.1%}), {len(self.divergences)} divergent"
        )


def first_divergence(a: str, b: str) -> int | None:
    """Index of the first differing character, or None if identical."""
    if a == b:
        return None
    limit = min(len(a), len(b))
    for i in range(limit):
        if a[i] != b[i]:
            return i
    return limit  # one is a prefix of the other


def _token_index_of_char(text: str, char_idx: int, tokenizer: Any) -> int | None:
    """Which token contains a given character offset.

    Uses offset mapping when the tokenizer is a fast one; falls back to incremental
    decoding otherwise. Returns None when neither is possible -- a missing token index
    is reported as missing, not approximated.
    """
    if tokenizer is None:
        return None
    try:
        enc = tokenizer(text, return_offsets_mapping=True, add_special_tokens=False)
        offsets = enc["offset_mapping"]
    except (TypeError, KeyError, NotImplementedError):
        offsets = None

    if offsets:
        for i, (start, end) in enumerate(offsets):
            if start <= char_idx < end:
                return i
        return len(offsets)

    try:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    except Exception:  # noqa: BLE001 -- tokenizer APIs vary; a failure here is non-fatal
        return None
    consumed = 0
    for i in range(len(ids)):
        piece = tokenizer.decode(ids[: i + 1], skip_special_tokens=True)
        consumed = len(piece)
        if consumed > char_idx:
            return i
    return len(ids)


def compare_conditions(
    df: pd.DataFrame,
    *,
    condition_id: str,
    baseline_condition_id: str,
    tokenizer: Any = None,
    excerpt_chars: int = 60,
) -> IdentityReport:
    """Compare one condition's outputs against its baseline, keyed on (prompt_id, repeat_idx).

    Keyed, never positional. Interleaved execution means log order is not condition
    order, so zipping two lists would compare unrelated pairs and report a low identity
    rate that means nothing at all.
    """
    for cid in (condition_id, baseline_condition_id):
        if not (df["condition_id"] == cid).any():
            raise ValueError(f"no rows for condition {cid!r}")

    key = ["prompt_id", "repeat_idx"]
    cand = df[df["condition_id"] == condition_id].set_index(key)["output_text"]
    base = df[df["condition_id"] == baseline_condition_id].set_index(key)["output_text"]

    for name, series in (("candidate", cand), ("baseline", base)):
        if series.index.duplicated().any():
            raise ValueError(
                f"{name} condition has duplicate (prompt_id, repeat_idx) keys; the log "
                "contains a repeated measurement that must be resolved first."
            )

    shared = sorted(set(cand.index) & set(base.index))
    if not shared:
        raise ValueError(
            "the two conditions share no (prompt_id, repeat_idx) keys, so nothing can be "
            "compared."
        )
    only_cand = set(cand.index) - set(base.index)
    only_base = set(base.index) - set(cand.index)
    if only_cand or only_base:
        raise ValueError(
            f"condition coverage differs: {len(only_cand)} keys only in the candidate, "
            f"{len(only_base)} only in the baseline. Refusing to compare a subset -- a "
            "missing cell is a gap to explain, not to drop."
        )

    divergences: list[Divergence] = []
    n_identical = 0
    for k in shared:
        a, b = base.loc[k], cand.loc[k]
        idx = first_divergence(a, b)
        if idx is None:
            n_identical += 1
            continue
        divergences.append(
            Divergence(
                prompt_id=k[0],
                repeat_idx=int(k[1]),
                identical=False,
                first_divergent_char=idx,
                first_divergent_token=_token_index_of_char(a, idx, tokenizer),
                baseline_excerpt=a[max(0, idx - excerpt_chars // 2): idx + excerpt_chars],
                candidate_excerpt=b[max(0, idx - excerpt_chars // 2): idx + excerpt_chars],
            )
        )

    return IdentityReport(
        condition_id=condition_id,
        baseline_condition_id=baseline_condition_id,
        n_compared=len(shared),
        n_identical=n_identical,
        divergences=divergences,
    )


def assert_identity(report: IdentityReport, *, min_rate: float = 0.95) -> None:
    """Gate the speculative arm on the identity rate.

    Below the threshold the acceptance logic is suspect, and every speedup measured with
    it is measuring something other than what the study claims.
    """
    if report.identity_rate < min_rate:
        raise AssertionError(
            f"{report.summary()} -- below the {min_rate:.0%} threshold. Speculative "
            "decoding is not reproducing the target model's output, so the acceptance "
            "logic is wrong. Fix it before measuring speed with it."
        )


def side_by_side(
    report: IdentityReport,
    *,
    limit: int = 3,
    marker: str = " <<<DIVERGES HERE>>> ",
) -> str:
    """Render divergent cases as a Markdown block for the report.

    The first divergence position is marked inline so a reader can judge for themselves
    whether it sits at a genuine close call.
    """
    if not report.divergences:
        return f"All {report.n_compared} pairs identical for {report.condition_id}."

    lines = [
        f"### {report.condition_id} vs baseline {report.baseline_condition_id}",
        "",
        f"Identity rate: **{report.identity_rate:.2%}** "
        f"({report.n_identical}/{report.n_compared})",
        "",
        "| prompt_id | repeat | first divergent token | baseline | speculative |",
        "|---|---|---|---|---|",
    ]
    for d in report.divergences[:limit]:
        tok = d.first_divergent_token if d.first_divergent_token is not None else "n/a"
        lines.append(
            f"| {d.prompt_id} | {d.repeat_idx} | {tok} | "
            f"`{_mark(d.baseline_excerpt, marker)}` | `{_mark(d.candidate_excerpt, marker)}` |"
        )
    if len(report.divergences) > limit:
        lines.append("")
        lines.append(f"_{len(report.divergences) - limit} further divergent pairs not shown._")
    return "\n".join(lines)


def _mark(excerpt: str, marker: str) -> str:
    """Excerpts are centred on the divergence, so the marker goes mid-string."""
    mid = len(excerpt) // 3
    return (excerpt[:mid] + marker + excerpt[mid:]).replace("|", "\\|").replace("\n", " ")
