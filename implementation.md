# Implementation Specification

Build spec for the inference-optimization study. Companion to
`inference_optimization_manual.md` — the manual holds the argument and the protocol; this
holds the interfaces, work packages and acceptance criteria.

**Audience:** coding agents working in parallel, plus you as integrator.

Section references like §5.1 point at the manual.

---

## 0. Rules that override everything else

Give these to every agent verbatim. They are the failure modes that quietly destroy this
kind of study.

1. **Never fabricate, simulate, mock, estimate or interpolate a measurement.** If a model
   fails to load, a run OOMs, or a dependency is missing, **raise and stop**. Do not
   substitute a smaller model, do not fall back to random numbers, do not fill gaps.
   A missing cell in the results is correct; an invented one is fatal.
2. **Synthetic data exists only under `tests/fixtures/`.** Every record carries
   `provenance: "measured" | "fixture"`. Plotting and reporting functions must raise on any
   record where `provenance != "measured"`. Tests may use fixtures; the report may not.
3. **Do not modify `core/schema.py` without changing this document first.** It is the
   contract between every module. An agent that "improves" the schema breaks every other
   agent's work.
4. **Do not change measurement semantics for convenience.** Specifically: `ignore_eos` stays
   `True`, `max_tokens` stays fixed, warmup iterations stay discarded, conditions stay
   interleaved. If a constraint makes something slow, it is still the constraint.
5. **Ratios use geometric means.** Arithmetic means of speedups are wrong. Confidence
   intervals are paired. No exceptions.
6. **No network at plot time.** No style downloads, no font fetching, no CDN. Matplotlib
   defaults only, no seaborn.
7. **Every number that appears in a figure or table must be traceable to a `run_id`** in a
   JSONL log. If you cannot trace it, it does not go in.

---

## 1. Repository layout

```
inference-study/
├── core/
│   ├── schema.py          # RunRecord, RunConfig, validation, JSONL IO
│   ├── config.py          # YAML sweep config -> list[RunConfig]
│   └── env.py             # provenance capture: versions, driver, clocks, host
├── runners/
│   ├── base.py            # Runner protocol + GenResult
│   ├── vllm_runner.py     # GPU arm (all speculative + precision configs)
│   ├── hf_runner.py       # MTP drafters, acceptance experiments, CPU quality
│   └── llamacpp_runner.py # CPU wall-clock arm
├── evals/
│   ├── gsm8k.py           # loader + exact-match scorer
│   ├── perplexity.py      # WikiText-2 sliding-window PPL
│   └── identity.py        # byte-identity checker vs baseline
├── analysis/
│   ├── stats.py           # paired bootstraps, aggregation
│   ├── model.py           # Leviathan speedup formula, optimal gamma, roofline
│   └── load.py            # JSONL -> DataFrame, validation, provenance guard
├── bench/
│   └── micro.py           # STREAM-style bandwidth, GEMM throughput, ridge point
├── plots/
│   ├── style.py           # shared rcParams, colours, CI rendering
│   └── fig01..fig08.py    # one module per figure
├── scripts/
│   ├── run_sweep.py       # orchestration driver
│   ├── smoke.py           # end-to-end tiny run
│   └── build_report.py    # tables + figures -> report/
├── configs/
│   ├── instance1_precision_spec.yaml
│   ├── instance2_batch_drafter.yaml
│   ├── instance3_scale_tp.yaml
│   └── local_cpu.yaml
├── tests/
│   ├── fixtures/
│   └── test_*.py
├── logs/                  # JSONL, one file per instance
└── report/                # generated figures + tables
```

---

## 2. Shared contracts

### 2.1 `RunConfig`

One immutable object fully describing a single measurement condition.

```python
@dataclass(frozen=True)
class RunConfig:
    # identity
    condition_id: str            # stable hash of all fields below; used for grouping
    # model
    target_model: str
    target_dtype: Literal["bf16", "fp8", "w4a16"]
    draft_model: str | None
    spec_method: Literal["none", "draft_model", "mtp", "eagle", "eagle3", "ngram"]
    num_speculative_tokens: int | None   # gamma
    gamma_schedule: Literal["constant", "heuristic"] = "constant"
    # parallelism
    tensor_parallel_size: int = 1
    draft_tensor_parallel_size: int = 1
    nccl_p2p_disabled: bool = False
    # generation
    batch_size: int = 1
    max_tokens: int = 256
    ignore_eos: bool = True
    temperature: float = 0.0
    seed: int = 42
    # execution
    stack: Literal["vllm", "hf", "llamacpp"]
    platform: Literal["h100", "cpu"]
    repeats: int = 5
    warmup: int = 3
```

`condition_id` = first 12 hex chars of sha256 over the sorted field dict excluding
`repeats` and `warmup`. Deterministic across machines. Grouping key for all analysis.

### 2.2 `RunRecord`

Exactly one JSON object per prompt per repeat, appended to `logs/{hostname}.jsonl`. Schema
is fixed (this is the §5.3 schema, extended with provenance and condition linkage):

```python
REQUIRED_FIELDS = [
    "run_id", "condition_id", "provenance", "timestamp",
    "hostname", "platform", "stack", "stack_version", "driver",
    "target_model", "target_dtype", "draft_model", "spec_method",
    "num_speculative_tokens", "gamma_schedule",
    "tensor_parallel_size", "draft_tensor_parallel_size", "nccl_p2p_disabled",
    "batch_size", "prompt_id", "prompt_tokens", "max_tokens", "ignore_eos",
    "temperature", "seed", "repeat_idx", "is_warmup",
    "ttft_ms", "tpot_ms", "total_ms", "output_tokens",
    "accepted_tokens", "draft_tokens_proposed", "acceptance_rate",
    "mean_accept_length", "accept_length_histogram",
    "clocks_sm_mhz", "power_draw_w", "output_text",
]
```

Semantics, fixed and non-negotiable:

- `ttft_ms` — submit to first token emitted.
- `tpot_ms` — `(total_ms - ttft_ms) / (output_tokens - 1)`. Raise if `output_tokens < 2`.
- `total_ms` — submit to final token.
- `output_tokens` — **must equal `max_tokens`** when `ignore_eos` is true. If it does not,
  raise; do not record. This is the §5.1 guard and it is the single most important
  validation in the codebase.
- `accept_length_histogram` — list of counts indexed by accepted-run length, `[]` when
  `spec_method == "none"`.
- `is_warmup` — warmup records are written (useful for diagnosing thermal ramp) but every
  analysis function filters them out by default.
- Non-applicable numeric fields are `null`, never `0`.

`core/schema.py` exposes:

```python
def validate_record(rec: dict) -> None      # raises ValueError with the offending field
def append_record(path: Path, rec: dict) -> None   # validates, then appends one line
```

### 2.3 Sweep config (YAML)

```yaml
instance: instance-1
platform: h100
stack: vllm
log_path: logs/instance-1.jsonl
prompts:
  source: gsm8k
  split: test
  n: 250
  seed: 0
defaults:
  max_tokens: 256
  ignore_eos: true
  temperature: 0.0
  seed: 42
  repeats: 5
  warmup: 3
axes:
  target_dtype: [bf16, fp8, w4a16]
  spec:
    - {spec_method: none, num_speculative_tokens: null}
    - {spec_method: draft_model, num_speculative_tokens: 1}
    - {spec_method: draft_model, num_speculative_tokens: 2}
    - {spec_method: draft_model, num_speculative_tokens: 3}
    - {spec_method: draft_model, num_speculative_tokens: 4}
    - {spec_method: draft_model, num_speculative_tokens: 5}
    - {spec_method: draft_model, num_speculative_tokens: 7}
interleave: true
```

`interleave: true` means the driver emits conditions in round-robin order across prompts
rather than completing one condition at a time (§5.1 rule 3). This is a hard requirement,
not an optimization.

---

## 3. Work packages

Dependency order: **WP1 blocks everything.** WP2–WP5 are mutually independent. WP6–WP9 can
be built against fixtures in parallel with the runners.

### WP1 — Core contracts *(blocking; do this alone, first)*

**Deliver:** `core/schema.py`, `core/config.py`, `core/env.py`, tests.

- `validate_record` enforcing every rule in §2.2, with a specific error per violation.
- `condition_id` hashing, proven stable across processes by a test.
- YAML → `list[RunConfig]` expansion with the interleaving order materialized.
- `env.py` captures: hostname, stack version, driver version, CUDA version, GPU name,
  `clocks.sm`, `power.draw`, CPU model, thread count, and whether other processes are
  resident on the GPU. Called once per run, merged into every record.

**Acceptance:** a record missing any required field raises; a record with
`output_tokens != max_tokens` under `ignore_eos` raises; identical configs produce identical
`condition_id` in two separate interpreter processes.

### WP2 — vLLM runner

**Deliver:** `runners/vllm_runner.py`.

- Implements the `Runner` protocol against real vLLM. Speculative config passed as the
  `speculative_config` dict (§8.1), never as legacy flat flags.
- **Log the resolved `SpeculativeConfig` from vLLM startup into the first record of each
  condition.** If the requested method does not survive resolution (e.g. an MTP checkpoint
  resolving to `draft_model`), raise — do not proceed silently.
- Extract per-request acceptance counts and build `accept_length_histogram`.
- Disable logprobs for latency conditions.
- Set `gpu_memory_utilization` from config; do not tune it per-model without recording it.
- One process per condition. Tear down fully between conditions — no state reuse.

**Acceptance:** with `spec_method: none` and `spec_method: draft_model, gamma=4` at
temperature 0, both produce `output_tokens == max_tokens`, and the byte-identity checker
(WP5) reports ≥95% identity between them.

### WP3 — llama.cpp runner (CPU)

**Deliver:** `runners/llamacpp_runner.py`.

- Subprocess wrapper around the speculative binary. **Detect flag names at runtime by
  parsing `--help`** — draft-length flag names have changed across versions. Raise a clear
  error if expected flags are absent rather than guessing.
- Parse accepted-token counts from output into the same schema fields.
- Explicit thread pinning and fixed thread count from config.
- Same GGUF quantization for draft and target; record it.

**Acceptance:** produces schema-valid records; accepted-token parsing verified against a
captured sample of real output committed under `tests/fixtures/`.

### WP4 — HF runner

**Deliver:** `runners/hf_runner.py`.

- `generate(..., assistant_model=...)` path for MTP drafters (§8.4).
- Supports `num_assistant_tokens` and `num_assistant_tokens_schedule`.
- **Emit records with `stack: "hf"` and set `tpot_ms`/`total_ms` as measured, but mark the
  condition as quality-only** via a `latency_valid: false` flag in the record. Analysis must
  exclude HF timings from all speed figures. HF Python overhead understates speedup and its
  timings must never appear in a speed plot.

**Acceptance:** acceptance-rate extraction matches vLLM's within a reasonable band on the
same model pair; a test asserts no HF record reaches any speed figure.

### WP5 — Quality evaluation

**Deliver:** `evals/gsm8k.py`, `evals/perplexity.py`, `evals/identity.py`.

- GSM8K: fixed 250-example subset selected by seed, committed as an ID list so every
  condition sees identical examples. Exact-match scorer with the standard final-number
  extraction; emit **per-example binary scores**, not just an aggregate — the stats module
  needs per-example vectors for pairing.
- Perplexity: sliding window over WikiText-2, fixed stride, recorded in the config.
- Identity: byte-compare `output_text` against the matched baseline condition, keyed on
  `(prompt_id, repeat_idx)`. Report identity rate plus, for divergent cases, the token index
  of first divergence.

**Acceptance:** per-example score vectors align by `prompt_id` across all conditions; a
deliberately corrupted output produces a divergence index pointing at the right token.

### WP6 — Statistics

**Deliver:** `analysis/stats.py`.

```python
def median_over_repeats(df) -> DataFrame
    # collapse repeats within (condition_id, prompt_id) by median; warmup excluded

def paired_bootstrap_speedup(t_base, t_opt, n_boot=10_000, seed=0) -> tuple[float, float, float]
    # geometric mean of per-prompt ratios; returns (point, lo95, hi95)

def paired_bootstrap_delta(score_base, score_opt, n_boot=10_000, seed=0) -> tuple[float, float, float]
    # mean of per-example deltas; returns (point, lo95, hi95)
```

- Inputs must be aligned by `prompt_id`; raise on misalignment rather than truncating.
- Collapse repeats **before** bootstrapping across prompts (§5.4). A test must assert that
  bootstrapping over un-collapsed repeats is not possible through the public API.

**Acceptance:** on synthetic data with a known 2.0× ratio and known noise, the CI covers 2.0
at approximately the nominal rate over repeated trials.

### WP7 — Analytical model

**Deliver:** `analysis/model.py`.

```python
def expected_speedup(alpha: float, gamma: int, c: float) -> float
    # (1 - alpha**(gamma+1)) / ((1 - alpha) * (gamma*c + 1))

def optimal_gamma(alpha: float, c: float, gamma_max: int = 16) -> int
def measure_c(draft_step_ms: float, target_step_ms: float) -> float
def ridge_point(tflops: float, bandwidth_gbs: float) -> float
def predicted_vs_measured(df) -> DataFrame   # adds predicted, residual columns
```

- `expected_speedup` handles `alpha == 1.0` by limit rather than dividing by zero.
- `measure_c` takes timings from real isolated batch-1 runs, never from parameter counts.

**Acceptance:** unit tests against hand-computed values at α ∈ {0.5, 0.8}, γ ∈ {1, 4},
c ∈ {0.05, 0.2}; monotonicity assertions (speedup decreasing in `c`, increasing in `α`).

### WP8 — Microbenchmarks

**Deliver:** `bench/micro.py`.

- STREAM-style triad for achieved memory bandwidth, per platform.
- Large GEMM sweep for achieved compute throughput at the relevant dtype.
- `ridge_point()` from measured values, written to `logs/{hostname}_micro.json`.

**Acceptance:** reports achieved not peak, with the measurement size and repeat count
recorded. A test asserts the module never reads a hardcoded spec-sheet constant.

### WP9 — Plots

**Deliver:** `plots/style.py` and `plots/fig01.py` … `fig08.py`.

Every module exposes `def render(df: DataFrame, outdir: Path) -> Path`. Every module calls
the provenance guard first and raises on fixture data. Figures are PDF plus PNG at 200 dpi.
Full spec in §4.

### WP10 — Orchestration

**Deliver:** `scripts/run_sweep.py`, `scripts/smoke.py`.

- Expands the YAML into an interleaved condition×prompt×repeat queue.
- Checkpoints progress so an interrupted sweep resumes without repeating completed cells.
- Re-checks GPU exclusivity before each condition block; aborts with a clear message if
  another process appears.
- Logs `clocks.sm` and `power.draw` per record.
- `smoke.py`: 1 target model, 2 conditions, 20 prompts, 1 repeat, `max_tokens=32`, and it
  must run **the full pipeline through figure generation**. Smoke is not a runner test; it
  is a pipeline test.

**Acceptance:** `python scripts/smoke.py` produces every figure file from real (tiny) runs.
Killing it mid-sweep and restarting produces no duplicate `run_id`s and no gaps.

### WP11 — Report assembly

**Deliver:** `scripts/build_report.py`.

- Primary table (§10.1) as CSV and Markdown.
- All figures.
- A provenance appendix: every `condition_id`, its resolved config, run count, host,
  stack version, driver, and clock state.
- The §7.4 side-by-side rendered as a three-column Markdown block from real `output_text`
  fields, with the first divergence position marked.

**Acceptance:** every number in the primary table resolves to a set of `run_id`s listed in
the appendix.

---

## 4. Figure specifications

All: 95% CIs as error bars or bands, geometric-mean point estimates for ratios, sample size
annotated, no title inside the figure (captions live in the report).

| Fig | Content | x | y | Series | Notes |
|---|---|---|---|---|---|
| 01 | Quality–throughput Pareto | throughput (tok/s) | GSM8K EM | one point per condition | Frontier traced; CIs on both axes; label precision + γ |
| 02 | Composition (§3.3) | target precision | spec speedup (geo mean) | grouped bars | The headline test — annotate the predicted direction |
| 03 | Batch-size collapse | batch size (log) | spec speedup | one line per precision | Horizontal line at 1.0; mark predicted crossover |
| 04 | Model vs measured | γ | speedup | measured points + predicted curve | Residual panel beneath |
| 05 | Platform curve | measured ridge point (log) | spec speedup | CPU, H100 TP1, H100 TP2 | Extrapolated points visually distinct (open markers, dashed) |
| 06 | Acceptance distribution | accepted run length | frequency | one histogram per condition | Small multiples, not overlaid |
| 07 | Metric sensitivity | precision level | Δ from BF16 baseline | PPL delta and EM delta | Twin axes; the divergence is the point |
| 08 | γ-slope, dense vs MoE (axis K) | γ | speedup normalized to γ=1 | dense vs MoE | Slope is the measurement; annotate fitted slopes |

Figure 05 must visually distinguish measured points from extrapolation. An extrapolated
point drawn identically to a measured one is a misrepresentation.

---

## 5. Parallelization plan

| Agent | Packages | Starts |
|---|---|---|
| A | WP1 | immediately, alone |
| B | WP2, WP10 | after WP1 |
| C | WP3, WP4 | after WP1 |
| D | WP5 | after WP1 |
| E | WP6, WP7, WP8 | after WP1, against fixtures |
| F | WP9, WP11 | after WP1, against fixtures |

Agents E and F work entirely against `tests/fixtures/` and never need a GPU. They should be
started at the same time as B/C/D — plotting and statistics are usually the bottleneck at
the end, and they have no hardware dependency.

---

## 6. Delegation prompt template

Prefix every agent task with this:

> You are implementing one module of a controlled inference-benchmarking study. Correctness
> of measurement matters more than performance, elegance, or feature coverage.
>
> Hard rules, in priority order:
> 1. Never fabricate, mock, simulate, or estimate a measurement. If something fails, raise
>    with a clear message and stop. Do not substitute a smaller model or synthetic data.
> 2. Do not modify `core/schema.py` or any shared interface. If you believe an interface is
>    wrong, stop and report it instead of changing it.
> 3. Do not "helpfully" add defaults, fallbacks, retries, or convenience behaviour that
>    changes measurement semantics.
> 4. Every function you write that consumes results must reject records where
>    `provenance != "measured"`.
> 5. Write tests that would fail if the measurement were subtly wrong, not tests that assert
>    the code runs.
>
> Read `IMPLEMENTATION_SPEC.md` §<n> for your package. Implement exactly that scope. Report
> anything ambiguous rather than resolving it yourself.

Then append the specific work package.

---

## 7. Known agent failure modes

Watch for these in review. Each has destroyed a study of this kind before.

| Failure | How it shows up | Guard |
|---|---|---|
| Silent model substitution | A 7B config that quietly loaded a 1.5B after an OOM | Log resolved model path per record; assert it matches config |
| Fabricated fallback data | Plots render beautifully before any GPU run finished | Provenance guard in every plot module |
| `ignore_eos` dropped | Speedups track output length | Hard validation in `validate_record` |
| Arithmetic mean of ratios | Speedup inflated a few percent | Geometric mean only in the public API |
| Repeats treated as independent | CIs implausibly tight | Collapse-before-bootstrap enforced by API shape |
| Warmup included | First condition of each block looks slow | `is_warmup` filtered by default in `load.py` |
| Cross-stack comparison | CPU vs GPU gap looks enormous | `stack` in the grouping key; assertion in figure modules |
| HF timings in speed plots | Speculative speedup mysteriously flat | `latency_valid: false` enforced |
| Spec config silently downgraded | MTP resolved to generic draft model | Assert resolved config matches requested |
| Schema drift between agents | Records fail to load at analysis time | WP1 lands first; schema frozen |
| Tests that assert nothing | Green suite, wrong numbers | Require at least one test per module that fails on a plausible measurement bug |

---

## 8. Integration checklist

Before launching the real sweeps:

- [ ] `pytest` green
- [ ] `python scripts/smoke.py` produces all eight figures from real tiny runs
- [ ] Byte-identity check (§7.2) run on ~200 prompts; identity rate recorded; divergences
      inspected and confirmed to sit at low-margin positions
- [ ] `bench/micro.py` run on every platform; measured ridge points recorded
- [ ] `measure_c` run for every draft/target pair actually used
- [ ] Predicted optimal γ written down, with timestamp, **before** the γ sweep runs
- [ ] Calibration config (§5.2) run on all three instances; drift recorded
- [ ] GPU exclusivity confirmed on each instance
- [ ] Clocks locked; `clocks.sm` logging verified non-null in sample records
- [ ] Resume-after-kill tested on a real sweep config