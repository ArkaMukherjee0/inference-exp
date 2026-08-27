"""GSM8K loader and exact-match scorer.

Two properties matter more than anything else here.

**The subset is frozen and committed.** Every condition must see byte-identical
examples, or a quality difference between conditions is partly a difference in which
questions were asked. The 250 ids are selected once by seed and written to
``evals/gsm8k_subset_ids.json``; after that the file is the authority and the seed is
only provenance. If the file is missing, loading raises rather than re-deriving it --
a silently regenerated subset is how two instances end up scoring different exams.

**Scores are per-example, not aggregate.** ``analysis.stats.paired_bootstrap_delta``
needs the per-example binary vector to pair by ``prompt_id``. An accuracy percentage
cannot be paired, and comparing two percentages throws away the fact that the same
questions were asked both times -- which is most of the statistical power available.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SUBSET_PATH = Path(__file__).with_name("gsm8k_subset_ids.json")

# Fully-qualified ``namespace/name``, never the bare legacy alias.
#
# The old "canonical" datasets (``gsm8k``, ``wikitext``, ...) were moved under real
# namespaces, and current huggingface_hub rejects an unqualified id outright: it builds
# ``hf://datasets/gsm8k@.../...`` and raises HfUriError because the repo id is not
# ``namespace/name``. Pinning the qualified id resolves on both old and new clients, so
# there is no version sniffing here.
GSM8K_DATASET = "openai/gsm8k"
GSM8K_CONFIG = "main"

# The gold answer sits after '####' in GSM8K's reference solutions.
_GOLD_RE = re.compile(r"####\s*(-?[\d,]*\.?\d+)")
# Standard final-number extraction: the last number appearing in the completion.
_NUMBER_RE = re.compile(r"-?\d[\d,]*\.?\d*")


@dataclass(frozen=True)
class Example:
    prompt_id: str
    question: str
    answer: str  # the gold final number, normalized

    @property
    def prompt(self) -> str:
        return self.question.strip()


def _normalize_number(raw: str) -> str:
    """Canonical form for comparison: no commas, no trailing zeros, no '+'.

    Done as string normalization rather than float comparison because GSM8K answers are
    exact integers and rationals; float round-tripping would make 1/3 cases compare
    unequal for reasons that have nothing to do with the model.
    """
    s = raw.replace(",", "").replace("$", "").strip().lstrip("+")
    if s.endswith("."):
        s = s[:-1]
    if "." in s:
        s = s.rstrip("0").rstrip(".")
    if s in ("", "-"):
        return ""
    # -0 and 0 are the same answer.
    if s.lstrip("-") == "0":
        return "0"
    return s


def extract_gold(reference: str) -> str:
    m = _GOLD_RE.search(reference)
    if not m:
        raise ValueError(f"no '#### <answer>' found in GSM8K reference: {reference[-120:]!r}")
    return _normalize_number(m.group(1))


def extract_prediction(completion: str) -> str | None:
    """The model's final answer: the last number it produced.

    Returns None when the completion contains no number at all, which scores as wrong
    rather than raising -- a model that never produced a number genuinely got it wrong.
    Note that under ``ignore_eos`` completions run past their natural end, so the last
    number in the text may follow the real answer; that affects every condition
    identically, which is why the identical-token-budget rule matters for quality too.
    """
    matches = _NUMBER_RE.findall(completion)
    if not matches:
        return None
    return _normalize_number(matches[-1])


def score_example(completion: str, gold: str) -> int:
    """1 if the extracted final number matches the gold answer, else 0."""
    pred = extract_prediction(completion)
    return int(pred is not None and pred == gold)


# --------------------------------------------------------------------------------------
# Subset management
# --------------------------------------------------------------------------------------


def build_subset(*, n: int, seed: int, split: str = "test", force: bool = False) -> list[str]:
    """Select and commit the frozen id list. Run once, then never again.

    Refuses to overwrite an existing subset without ``force``: silently reselecting
    would invalidate every quality number already measured against the old list.
    """
    if SUBSET_PATH.exists() and not force:
        raise FileExistsError(
            f"{SUBSET_PATH} already exists. The subset is frozen once chosen -- "
            "reselecting it would make already-measured quality scores incomparable. "
            "Pass force=True only if you intend to invalidate them."
        )
    rows = _load_hf_split(split)
    if n > len(rows):
        raise ValueError(f"asked for {n} examples but split {split!r} has {len(rows)}")
    rng = np.random.default_rng(seed)
    idx = sorted(rng.choice(len(rows), size=n, replace=False).tolist())
    ids = [f"gsm8k-{split}-{i}" for i in idx]
    SUBSET_PATH.write_text(
        json.dumps(
            {
                "dataset": GSM8K_DATASET,
                "config": GSM8K_CONFIG,
                "split": split,
                "seed": seed,
                "n": n,
                "ids": ids,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return ids


def load_subset() -> list[Example]:
    """The frozen 250 examples, in committed order."""
    if not SUBSET_PATH.exists():
        raise FileNotFoundError(
            f"{SUBSET_PATH} not found. Create it once with build_subset(n=250, seed=0); "
            "it must then be committed so every instance scores the same exam."
        )
    meta = json.loads(SUBSET_PATH.read_text(encoding="utf-8"))
    # The ids are positional indices into a split, so they are only meaningful against
    # the dataset they were drawn from. Files written before this field existed are
    # assumed to be the current source, which they were.
    recorded = meta.get("dataset", GSM8K_DATASET)
    if recorded != GSM8K_DATASET:
        raise ValueError(
            f"{SUBSET_PATH} was frozen against {recorded!r} but this code loads "
            f"{GSM8K_DATASET!r}. The ids are positional, so they would select different "
            "questions. Re-freeze the subset deliberately rather than silently rescoring."
        )
    rows = _load_hf_split(meta["split"])
    examples: list[Example] = []
    for pid in meta["ids"]:
        idx = int(pid.rsplit("-", 1)[1])
        if idx >= len(rows):
            raise ValueError(
                f"{pid} is out of range for the loaded split ({len(rows)} rows). The "
                "dataset version has changed under the frozen id list."
            )
        row = rows[idx]
        examples.append(
            Example(prompt_id=pid, question=row["question"], answer=extract_gold(row["answer"]))
        )
    if len(examples) != meta["n"]:
        raise ValueError(f"expected {meta['n']} examples, built {len(examples)}")
    return examples


def _load_hf_split(split: str) -> list[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "the 'datasets' package is required to load GSM8K. Install it; do not "
            "substitute a local copy of unknown provenance."
        ) from exc
    ds = load_dataset(GSM8K_DATASET, GSM8K_CONFIG, split=split)
    return [{"question": r["question"], "answer": r["answer"]} for r in ds]


# --------------------------------------------------------------------------------------
# Scoring a run log
# --------------------------------------------------------------------------------------


def score_frame(df: pd.DataFrame, examples: list[Example] | None = None) -> pd.DataFrame:
    """Attach a per-example binary ``em`` column to a run frame.

    Returns one row per (condition_id, prompt_id) so it lines up with the output of
    ``median_over_repeats``. Repeats of the same prompt at temperature 0 should score
    identically; if they do not, that is reported rather than averaged away.
    """
    examples = examples or load_subset()
    gold = {e.prompt_id: e.answer for e in examples}

    missing = sorted(set(df["prompt_id"]) - set(gold))
    if missing:
        raise ValueError(
            f"{len(missing)} prompt_id(s) in the log are not in the frozen subset "
            f"(e.g. {missing[:3]}). The log and the exam disagree."
        )

    scored = df.copy()
    scored["em"] = [
        score_example(text, gold[pid])
        for text, pid in zip(scored["output_text"], scored["prompt_id"])
    ]

    grouped = scored.groupby(["condition_id", "prompt_id"], dropna=False)["em"]
    agg = grouped.agg(["mean", "nunique", "size"]).reset_index()
    inconsistent = agg[agg["nunique"] > 1]
    if not inconsistent.empty:
        raise ValueError(
            f"{len(inconsistent)} (condition, prompt) cells scored differently across "
            "repeats at temperature 0, which should be deterministic. Investigate before "
            "aggregating: e.g. "
            f"{inconsistent.head(3)[['condition_id', 'prompt_id']].to_dict('records')}"
        )
    out = agg.rename(columns={"mean": "em", "size": "n_repeats"})[
        ["condition_id", "prompt_id", "em", "n_repeats"]
    ]
    out["em"] = out["em"].astype(int)
    return out


def accuracy(scored: pd.DataFrame, condition_id: str) -> float:
    """Aggregate exact-match for one condition. Per-example vectors stay available."""
    rows = scored[scored["condition_id"] == condition_id]
    if rows.empty:
        raise ValueError(f"no scored rows for condition {condition_id!r}")
    return float(rows["em"].mean())
