"""Shared test fixtures.

Note the ``measured_df`` fixture below. Plot and report modules refuse fixture-provenance
data by design, so exercising them requires data labelled ``measured``. That relabelling
happens **here, visibly, inside the test suite** -- never in library code. There is no
code path in the package itself that can turn a fixture into a measurement.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

FIXTURE_DIR = Path(__file__).parent / "fixtures"
FIXTURE_LOG = FIXTURE_DIR / "fixture_runs.jsonl"


@pytest.fixture(scope="session")
def fixture_records() -> list[dict]:
    if not FIXTURE_LOG.exists():
        pytest.skip("fixture log not generated; run python -m tests.fixtures.make_fixtures")
    with FIXTURE_LOG.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


@pytest.fixture()
def fixture_df(fixture_records) -> pd.DataFrame:
    """The fixture log as a frame, warmups dropped, provenance still 'fixture'."""
    from analysis.load import load_runs

    return load_runs(FIXTURE_LOG)


@pytest.fixture()
def measured_df(fixture_df) -> pd.DataFrame:
    """Fixture data relabelled as measured, so plotting code can be exercised.

    Deliberately explicit. If this line ever appears outside the test suite, that is a
    bug worth failing the build over.
    """
    df = fixture_df.copy()
    df["provenance"] = "measured"
    return df


@pytest.fixture()
def llamacpp_output() -> str:
    path = FIXTURE_DIR / "llamacpp_speculative.txt"
    if not path.exists():
        pytest.skip("llama.cpp sample output fixture missing")
    return path.read_text(encoding="utf-8")


@pytest.fixture()
def tmp_outdir(tmp_path: Path) -> Path:
    out = tmp_path / "figures"
    out.mkdir()
    return out
