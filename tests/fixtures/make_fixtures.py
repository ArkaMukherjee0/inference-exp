"""Generate the synthetic fixture log.

Synthetic data exists **only** here, and every record it writes carries
``provenance: "fixture"``. That label is what the provenance guard in
``analysis.load.require_measured`` keys on, so nothing produced by this file can reach a
figure or a table without a test explicitly and visibly relabelling it.

The fixtures are built with a *known* ground truth -- a designed speedup per condition
and a designed acceptance rate -- so the statistics tests can check that the estimators
recover the number that was put in.

Regenerate with:
    python -m tests.fixtures.make_fixtures
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

FIXTURE_PATH = Path(__file__).with_name("fixture_runs.jsonl")

# Ground truth the tests assert against.
TRUE_SPEEDUP = {
    # (dtype, gamma) -> designed speedup vs the same-dtype baseline.
    # Falls with precision, which is the hypothesis the study tests; the fixtures encode
    # it so the *plots* can be exercised, never so the finding can be assumed.
    ("bf16", 1): 1.35, ("bf16", 2): 1.72, ("bf16", 4): 2.00, ("bf16", 7): 1.78,
    ("fp8", 1): 1.22, ("fp8", 2): 1.44, ("fp8", 4): 1.55, ("fp8", 7): 1.38,
    ("w4a16", 1): 1.10, ("w4a16", 2): 1.20, ("w4a16", 4): 1.24, ("w4a16", 7): 1.12,
}
TRUE_ALPHA = {"bf16": 0.78, "fp8": 0.76, "w4a16": 0.74}
BASE_TPOT_MS = {"bf16": 22.0, "fp8": 13.0, "w4a16": 9.0}
MAX_TOKENS = 64
N_PROMPTS = 24
N_REPEATS = 3
BATCH_SIZES = (1, 4, 16, 64)


def condition_id_for(**fields: Any) -> str:
    """Reuse the real hashing so fixtures group exactly like real records."""
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from core.schema import RunConfig

    cfg = RunConfig(**fields)
    return cfg.condition_id


def _simulate_histogram(gamma: int, alpha: float, rng: random.Random) -> list[int]:
    """Accepted-run lengths for one generation, ending exactly at MAX_TOKENS.

    Each step emits (k accepted drafts + 1 bonus token), so the token budget closes
    exactly -- the same invariant the real runners are held to.
    """
    hist = [0] * (gamma + 1)
    tokens = 0
    while tokens < MAX_TOKENS:
        k = 0
        while k < gamma and rng.random() < alpha:
            k += 1
        if tokens + k + 1 > MAX_TOKENS:
            k = MAX_TOKENS - tokens - 1
        hist[k] += 1
        tokens += k + 1
    return hist


def _record(
    *,
    cfg_fields: dict[str, Any],
    prompt_id: str,
    repeat_idx: int,
    is_warmup: bool,
    tpot_ms: float,
    hist: list[int] | None,
    rng: random.Random,
) -> dict[str, Any]:
    gamma = cfg_fields.get("num_speculative_tokens")
    ttft = 40.0 + rng.uniform(-3, 3)
    total = ttft + tpot_ms * (MAX_TOKENS - 1)

    rec: dict[str, Any] = {
        "run_id": f"fix{rng.getrandbits(48):012x}",
        "condition_id": condition_id_for(**cfg_fields),
        "provenance": "fixture",
        "timestamp": "2026-08-01T00:00:00.000+00:00",
        "hostname": "fixture-host",
        "platform": cfg_fields["platform"],
        "stack": cfg_fields["stack"],
        "stack_version": "vllm==0.0.0-fixture",
        "driver": "driver=000.00-fixture",
        "target_model": cfg_fields["target_model"],
        "target_dtype": cfg_fields["target_dtype"],
        "draft_model": cfg_fields.get("draft_model"),
        "spec_method": cfg_fields["spec_method"],
        "num_speculative_tokens": gamma,
        "gamma_schedule": cfg_fields.get("gamma_schedule", "constant"),
        "tensor_parallel_size": cfg_fields.get("tensor_parallel_size", 1),
        "draft_tensor_parallel_size": cfg_fields.get("draft_tensor_parallel_size", 1),
        "nccl_p2p_disabled": cfg_fields.get("nccl_p2p_disabled", False),
        "batch_size": cfg_fields.get("batch_size", 1),
        "prompt_id": prompt_id,
        "prompt_tokens": 64 + (hash(prompt_id) % 40),
        "max_tokens": MAX_TOKENS,
        "ignore_eos": True,
        "temperature": 0.0,
        "seed": 42,
        "repeat_idx": repeat_idx,
        "is_warmup": is_warmup,
        "ttft_ms": ttft,
        "tpot_ms": tpot_ms,
        "total_ms": total,
        "output_tokens": MAX_TOKENS,
        "clocks_sm_mhz": 1755.0,
        "power_draw_w": 310.0 + rng.uniform(-15, 15),
        "output_text": _fixture_text(prompt_id, cfg_fields),
    }
    # Recompute tpot from the two endpoints so the record satisfies the validator's
    # re-derivation exactly, the same way build_record does.
    rec["tpot_ms"] = (rec["total_ms"] - rec["ttft_ms"]) / (rec["output_tokens"] - 1)

    if hist is None:
        rec.update({
            "accepted_tokens": None, "draft_tokens_proposed": None,
            "acceptance_rate": None, "mean_accept_length": None,
            "accept_length_histogram": [],
        })
    else:
        steps = sum(hist)
        accepted = sum(k * n for k, n in enumerate(hist))
        proposed = gamma * steps
        rec.update({
            "accepted_tokens": accepted,
            "draft_tokens_proposed": proposed,
            "acceptance_rate": accepted / proposed,
            "mean_accept_length": accepted / steps + 1.0,
            "accept_length_histogram": hist,
        })
    return rec


def _fixture_text(prompt_id: str, cfg_fields: dict[str, Any]) -> str:
    """Deterministic text so identity checks have something meaningful to compare.

    Speculative and non-speculative conditions at the same precision produce identical
    text (as they should at temperature 0); a lower precision changes the last digit,
    which gives the identity checker a real divergence to locate.
    """
    n = abs(hash(prompt_id)) % 97
    tail = "7" if cfg_fields["target_dtype"] == "w4a16" else "4"
    return f"Let us work through it. Step one gives {n}. The answer is {n}{tail}."


def build() -> list[dict[str, Any]]:
    rng = random.Random(20260827)
    prompt_ids = [f"gsm8k-test-{i}" for i in range(N_PROMPTS)]
    records: list[dict[str, Any]] = []

    models = [
        ("meta-llama/Llama-3.1-8B-Instruct", "meta-llama/Llama-3.2-1B-Instruct"),
        ("mistralai/Mixtral-8x7B-Instruct-v0.1", "mistralai/Mistral-7B-Instruct-v0.2"),
    ]

    for target, draft in models:
        is_moe = "Mixtral" in target
        for dtype in ("bf16", "fp8", "w4a16"):
            # The MoE model is measured at bf16 only, mirroring the real sweep.
            if is_moe and dtype != "bf16":
                continue
            for batch in BATCH_SIZES:
                if is_moe and batch != 1:
                    continue
                base_tpot = BASE_TPOT_MS[dtype] * (1.0 + 0.055 * math.log2(batch))
                for gamma in (None, 1, 2, 4, 7):
                    if batch != 1 and gamma not in (None, 4):
                        continue
                    spec = "none" if gamma is None else "draft_model"
                    cfg_fields = {
                        "target_model": target,
                        "draft_model": None if gamma is None else draft,
                        "target_dtype": dtype,
                        "spec_method": spec,
                        "num_speculative_tokens": gamma,
                        "stack": "vllm",
                        "platform": "h100",
                        "batch_size": batch,
                        "max_tokens": MAX_TOKENS,
                    }
                    speedup = 1.0 if gamma is None else _batch_decayed(
                        TRUE_SPEEDUP.get((dtype, gamma), 1.0), batch, is_moe, gamma
                    )
                    alpha = TRUE_ALPHA[dtype]
                    for repeat in range(N_REPEATS):
                        for pid in prompt_ids:
                            # Per-prompt variation, constant across conditions, so the
                            # paired ratio recovers the designed speedup exactly.
                            prompt_factor = 1.0 + 0.18 * ((abs(hash(pid)) % 100) / 100 - 0.5)
                            noise = 1.0 + rng.gauss(0, 0.012)
                            tpot = base_tpot * prompt_factor * noise / speedup
                            hist = None if gamma is None else _simulate_histogram(gamma, alpha, rng)
                            records.append(_record(
                                cfg_fields=cfg_fields, prompt_id=pid, repeat_idx=repeat,
                                is_warmup=False, tpot_ms=tpot, hist=hist, rng=rng,
                            ))

    # A handful of warmup rows, so the "warmups are excluded" tests have something to
    # exclude. They are deliberately much slower, as real cold iterations are.
    warm_cfg = {
        "target_model": models[0][0], "draft_model": None, "target_dtype": "bf16",
        "spec_method": "none", "num_speculative_tokens": None, "stack": "vllm",
        "platform": "h100", "batch_size": 1, "max_tokens": MAX_TOKENS,
    }
    for pid in prompt_ids[:3]:
        records.append(_record(
            cfg_fields=warm_cfg, prompt_id=pid, repeat_idx=0, is_warmup=True,
            tpot_ms=BASE_TPOT_MS["bf16"] * 2.4, hist=None, rng=rng,
        ))
    return records


def _batch_decayed(speedup: float, batch: int, is_moe: bool, gamma: int) -> float:
    """Speculative advantage decays toward (and past) 1.0 as the batch grows."""
    if batch == 1:
        return speedup * (1.0 + 0.04 * gamma if is_moe else 1.0)
    decay = 1.0 / (1.0 + 0.42 * math.log2(batch))
    return max(0.82, 1.0 + (speedup - 1.0) * decay)


def main() -> None:
    records = build()
    with FIXTURE_PATH.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, sort_keys=True, separators=(",", ":")) + "\n")
    print(f"wrote {len(records)} fixture records -> {FIXTURE_PATH}")


if __name__ == "__main__":
    main()
