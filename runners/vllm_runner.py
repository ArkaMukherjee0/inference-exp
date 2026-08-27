"""vLLM runner -- the GPU arm.

Carries every GPU speed measurement in the study: precision x speculation, the gamma
sweep, batch-size collapse, and tensor parallelism.

Three things in here exist purely to stop a plausible-looking wrong measurement:

1. Speculation is configured through the structured ``speculative_config`` dict, never
   the legacy flat flags, because the structured form is what the resolver consumes and
   therefore the form whose resolution can be checked.
2. The resolved config is read back after startup and compared against the request. An
   MTP checkpoint that quietly resolves to a generic draft model produces a perfectly
   valid run at a perfectly plausible speed -- and it is not the experiment.
3. The resolved model path is compared against the requested one, which is how a silent
   substitution after an OOM gets caught.

There is no fallback path in this file. Every failure raises.
"""

from __future__ import annotations

import os
import time
from typing import Any

from core.schema import RunConfig
from runners.base import GenResult, RunnerError


def _cuda_sync() -> None:
    """Block until queued GPU work has actually finished.

    perf_counter around an async submission measures the submission. Every wall-clock
    boundary in this runner is fenced with this.
    """
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


# Field names a vLLM V1 build uses for cumulative speculative-decoding counters. The
# per-position list is the one that matters: it is what a run-length histogram can be
# reconstructed from. Totals alone cannot make a distribution.
_SPEC_PER_POS_FIELDS = ("num_accepted_tokens_per_pos", "accepted_tokens_per_pos",
                        "num_accepted_tokens_per_position",
                        # vLLM V1's SpecDecodingLogging keeps a list *of lists*, one per
                        # logged interval, under this name.
                        "accepted_tokens_per_pos_lists")
_SPEC_TOTAL_FIELDS = ("num_accepted_tokens", "num_draft_tokens", "num_drafts")


def _find_spec_stats(root: Any, depth: int = 0, seen: set[int] | None = None) -> Any:
    """Locate an object carrying cumulative per-position acceptance counts."""
    seen = seen if seen is not None else set()
    if root is None or id(root) in seen or depth > 6:
        return None
    seen.add(id(root))

    for name in _SPEC_PER_POS_FIELDS:
        value = getattr(root, name, None)
        if isinstance(value, (list, tuple)) and value:
            return root

    for name in dir(root):
        if name.startswith("_"):
            continue
        try:
            child = getattr(root, name)
        except Exception:  # noqa: BLE001 -- probing engine internals
            continue
        if callable(child) or isinstance(child, (str, bytes, int, float, bool)):
            continue
        # Descend into containers as well as objects. vLLM V1 keeps its stat loggers in
        # a dict on the manager, so refusing to look inside dicts and lists means never
        # reaching the only object in the parent process that has the counts at all.
        children = []
        if isinstance(child, dict):
            children = list(child.values())
        elif isinstance(child, (list, tuple, set)):
            children = list(child)
        else:
            children = [child]
        for item in children:
            if isinstance(item, (str, bytes, int, float, bool)):
                continue
            found = _find_spec_stats(item, depth + 1, seen)
            if found is not None:
                return found
    return None


def _spec_stats_snapshot(llm: Any) -> dict[str, Any] | None:
    """Cumulative acceptance counters, or None if this build exposes none.

    At batch 1 with one request in flight, the difference between two snapshots is
    exactly that request's acceptance -- which is why this is usable at all. It would
    not be under concurrency, and the reconciliation against output_tokens in
    _hist_from_per_pos is what catches it if that assumption ever breaks.
    """
    engine = getattr(llm, "llm_engine", None)
    if engine is None:
        return None
    stats = _find_spec_stats(engine)
    if stats is None:
        return None

    snap: dict[str, Any] = {}
    for name in _SPEC_PER_POS_FIELDS:
        value = getattr(stats, name, None)
        if not isinstance(value, (list, tuple)) or not value:
            continue
        if isinstance(value[0], (list, tuple)):
            # A list of per-interval lists: sum them column-wise into one cumulative
            # per-position vector.
            width = max(len(row) for row in value)
            totals = [0] * width
            for row in value:
                for i, count in enumerate(row):
                    totals[i] += int(count)
            snap["per_pos"] = totals
        else:
            snap["per_pos"] = [int(x) for x in value]
        break
    for name in _SPEC_TOTAL_FIELDS:
        value = getattr(stats, name, None)
        if isinstance(value, int):
            snap[name] = value
    return snap or None


class _AcceptanceCollector:
    """Accumulates per-iteration SpecDecodingStats for one request.

    vLLM V1 runs its scheduler in a separate process and surfaces speculative statistics
    only as ``SchedulerStats.spec_decoding_stats``, handed to the parent's stat loggers
    once per engine iteration. Two consequences shaped this class:

    * ``last_scheduler_stats`` holds the **most recent iteration only** and is overwritten
      every step, so it cannot be read after a request and cannot be differenced. It has
      to be accumulated as it arrives.
    * The built-in logging logger resets its own accumulation whenever it logs (roughly
      every ten seconds), so borrowing that would silently lose part of a request.

    So we register as an additional stat logger and keep our own totals, reset explicitly
    at the start of each measured generation. At batch 1 with one request in flight, what
    accumulates between reset and read is exactly that request's acceptance -- and the
    reconciliation against ``output_tokens`` in ``_hist_from_per_pos`` is what catches it
    if that ever stops being true.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.per_pos: list[int] = []
        self.num_drafts = 0
        self.num_draft_tokens = 0
        self.num_accepted_tokens = 0
        self.iterations = 0

    # vLLM calls this once per engine iteration. The signature has varied across
    # versions, so it accepts anything and reads what it recognizes.
    def record(self, scheduler_stats: Any = None, iteration_stats: Any = None,
               *args: Any, **kwargs: Any) -> None:
        stats = getattr(scheduler_stats, "spec_decoding_stats", None)
        if stats is None:
            return
        per_pos = getattr(stats, "num_accepted_tokens_per_pos", None)
        if not per_pos:
            return

        self.iterations += 1
        if len(per_pos) > len(self.per_pos):
            self.per_pos.extend([0] * (len(per_pos) - len(self.per_pos)))
        for i, count in enumerate(per_pos):
            self.per_pos[i] += int(count)

        for attr, field in (("num_drafts", "num_drafts"),
                            ("num_draft_tokens", "num_draft_tokens"),
                            ("num_accepted_tokens", "num_accepted_tokens")):
            value = getattr(stats, field, None)
            if isinstance(value, int):
                setattr(self, attr, getattr(self, attr) + value)

    # Everything below is the StatLoggerBase surface. We are a collector, not a logger,
    # so these do nothing -- but they must exist or vLLM's manager will fail on us.
    def log(self, *args: Any, **kwargs: Any) -> None:
        return

    def log_engine_initialized(self, *args: Any, **kwargs: Any) -> None:
        return

    def record_sleep_state(self, *args: Any, **kwargs: Any) -> None:
        return


def _attach_acceptance_collector(llm: Any) -> _AcceptanceCollector | None:
    """Register a collector with the engine's stat-logger manager.

    Returns None when no manager is reachable, in which case the caller falls back to
    the other extraction strategies and ultimately raises rather than guessing.
    """
    engine = getattr(llm, "llm_engine", None)
    manager = getattr(engine, "logger_manager", None)
    if manager is None:
        return None
    loggers = getattr(manager, "stat_loggers", None)
    if not isinstance(loggers, list):
        return None
    collector = _AcceptanceCollector()
    loggers.append(collector)
    return collector


# Quantization methods vLLM may report for a 4-bit weight / 16-bit activation checkpoint.
_W4A16_METHODS = frozenset({"awq", "awq_marlin", "gptq", "gptq_marlin", "compressed-tensors"})
_FP8_METHODS = frozenset({"fp8", "compressed-tensors", "modelopt"})


class VLLMRunner:
    """One vLLM engine for exactly one condition. Never reused across conditions."""

    def __init__(self, config: RunConfig, *, log_extra: dict[str, Any] | None = None,
                 allow_missing_acceptance: bool = False) -> None:
        if config.stack != "vllm":
            raise RunnerError(f"VLLMRunner got stack={config.stack!r}")
        self.config = config
        # Off by default. When on, an engine that exposes no per-request acceptance
        # yields records that are speed-valid and acceptance-null, stamped so that the
        # gap is visible in every downstream table. It never fabricates a distribution.
        self.allow_missing_acceptance = allow_missing_acceptance
        self.acceptance_unavailable = False
        self.resolved: dict[str, Any] = {}
        self._llm: Any = None
        self._sampling: Any = None
        self._spec_before: dict[str, Any] | None = None
        self._collector: Any = None
        self._accept_capture: dict[str, Any] | None = None
        self._log_extra = log_extra or {}

    # -- lifecycle ------------------------------------------------------------------

    def setup(self) -> None:
        try:
            from vllm import LLM, SamplingParams
        except ImportError as exc:
            raise RunnerError(
                "vLLM is not installed in this interpreter. Install it or run this "
                "condition on an instance that has it -- do not substitute another stack."
            ) from exc

        cfg = self.config
        if cfg.nccl_p2p_disabled:
            # Must be set before the engine initializes NCCL.
            os.environ["NCCL_P2P_DISABLE"] = "1"

        # FlashInfer's sampling kernel is JIT-compiled at engine warmup and needs nvcc,
        # i.e. the full CUDA toolkit rather than just the driver. On a box with only the
        # driver the engine dies during warmup with "Could not find nvcc".
        #
        # Disabling it is also the better default for this study, for a reason beyond
        # convenience: whether a box happens to have a CUDA toolkit installed would
        # otherwise silently change which sampler runs, and therefore the per-step
        # overhead, between instances that are supposed to be comparable. Pinning it off
        # everywhere makes the sampler one less thing that varies by machine.
        #
        # An operator who has nvcc and explicitly wants FlashInfer can set the variable
        # themselves; we only supply the default. Either way the effective value is
        # recorded in the environment blob of every record.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

        kwargs: dict[str, Any] = {
            "model": cfg.target_model,
            "tensor_parallel_size": cfg.tensor_parallel_size,
            "seed": cfg.seed,
            "enforce_eager": False,
            # Prefix caching would turn a batch condition into a cache-hit measurement,
            # and would let one repeat contaminate the next.
            "enable_prefix_caching": False,
            "disable_log_stats": False,
        }
        if cfg.gpu_memory_utilization is not None:
            kwargs["gpu_memory_utilization"] = cfg.gpu_memory_utilization

        kwargs.update(self._dtype_kwargs())
        spec = self._speculative_config()
        if spec is not None:
            kwargs["speculative_config"] = spec

        self._llm = LLM(**kwargs)
        self._sampling = SamplingParams(
            temperature=cfg.temperature,
            max_tokens=cfg.max_tokens,
            ignore_eos=cfg.ignore_eos,
            seed=cfg.seed,
            # Logprobs force an extra device-to-host copy per step and would inflate
            # every latency number in the study.
            logprobs=None,
            detokenize=True,
        )
        self._verify_resolution(requested_spec=spec)
        if cfg.spec_method != "none":
            self._collector = _attach_acceptance_collector(self._llm)

    def close(self) -> None:
        if self._llm is None:
            return
        try:
            import contextlib
            import gc

            with contextlib.suppress(Exception):
                self._llm.llm_engine.engine_core.shutdown()
            del self._llm
            self._llm = None
            gc.collect()
            with contextlib.suppress(ImportError):
                import torch

                torch.cuda.empty_cache()
        finally:
            self._llm = None

    # -- engine configuration --------------------------------------------------------

    def _dtype_kwargs(self) -> dict[str, Any]:
        """Map our dtype axis onto vLLM's dtype/quantization knobs.

        w4a16 and fp8 are properties of the *checkpoint*, so we let vLLM detect them and
        verify afterwards rather than asserting a method name that may not match how the
        checkpoint was produced.
        """
        dt = self.config.target_dtype
        if dt == "bf16":
            return {"dtype": "bfloat16", "quantization": None}
        if dt in ("fp8", "w4a16"):
            return {"dtype": "auto", "quantization": None}
        raise RunnerError(f"unhandled target_dtype {dt!r}")

    def _speculative_config(self) -> dict[str, Any] | None:
        """The structured speculative_config dict (never the legacy flat flags)."""
        cfg = self.config
        if cfg.spec_method == "none":
            return None

        spec: dict[str, Any] = {"num_speculative_tokens": cfg.num_speculative_tokens}
        if cfg.spec_method == "draft_model":
            spec["model"] = cfg.draft_model
            spec["draft_tensor_parallel_size"] = cfg.draft_tensor_parallel_size
        elif cfg.spec_method == "ngram":
            spec["method"] = "ngram"
            spec["prompt_lookup_max"] = max(4, int(cfg.num_speculative_tokens or 1))
            spec["prompt_lookup_min"] = 1
        elif cfg.spec_method in ("eagle", "eagle3", "mtp"):
            spec["method"] = cfg.spec_method
            if cfg.draft_model:
                spec["model"] = cfg.draft_model
            spec["draft_tensor_parallel_size"] = cfg.draft_tensor_parallel_size
        else:
            raise RunnerError(f"unhandled spec_method {cfg.spec_method!r}")

        if cfg.gamma_schedule != "constant":
            raise RunnerError(
                "vLLM's speculative_config takes a fixed num_speculative_tokens; a "
                "heuristic gamma schedule must be run on the HF stack (quality only)."
            )
        return spec

    # -- the resolution guards -------------------------------------------------------

    def _verify_resolution(self, *, requested_spec: dict[str, Any] | None) -> None:
        """Compare what the engine actually built against what we asked for.

        This is the guard against the single most damaging failure mode in the study:
        a request that resolves to something else and runs happily.
        """
        vllm_cfg = self._engine_config()
        model_cfg = getattr(vllm_cfg, "model_config", None)
        if model_cfg is None:
            raise RunnerError(
                "could not read vLLM's resolved ModelConfig; cannot verify that the "
                "engine loaded what we asked for. Refusing to measure."
            )

        # 1. model identity -- catches a silent substitution.
        resolved_model = str(getattr(model_cfg, "model", ""))
        if resolved_model and os.path.basename(resolved_model.rstrip("/")) != os.path.basename(
            self.config.target_model.rstrip("/")
        ):
            raise RunnerError(
                f"resolved model {resolved_model!r} does not match requested "
                f"{self.config.target_model!r}."
            )

        # 2. precision -- catches a bf16 run mislabelled as fp8 (or vice versa).
        quant = getattr(model_cfg, "quantization", None)
        quant = str(quant).lower() if quant else None
        dt = self.config.target_dtype
        if dt == "bf16":
            if quant is not None:
                raise RunnerError(
                    f"target_dtype=bf16 but the checkpoint resolved to quantization={quant!r}."
                )
            resolved_dtype = str(getattr(model_cfg, "dtype", "")).lower()
            if "bfloat16" not in resolved_dtype:
                raise RunnerError(f"target_dtype=bf16 but engine dtype resolved to {resolved_dtype!r}")
        elif dt == "fp8":
            if quant is None or not any(m in quant for m in _FP8_METHODS):
                raise RunnerError(
                    f"target_dtype=fp8 but resolved quantization is {quant!r}. Point "
                    "target_model at an FP8 checkpoint; do not measure bf16 as fp8."
                )
        elif dt == "w4a16":
            if quant is None or not any(m in quant for m in _W4A16_METHODS):
                raise RunnerError(
                    f"target_dtype=w4a16 but resolved quantization is {quant!r}. Point "
                    "target_model at a 4-bit checkpoint."
                )

        # 3. speculative method -- catches MTP silently becoming a draft model.
        spec_cfg = getattr(vllm_cfg, "speculative_config", None)
        if requested_spec is None:
            if spec_cfg is not None:
                raise RunnerError(
                    "speculation was not requested but the engine resolved a "
                    f"speculative_config: {spec_cfg!r}"
                )
        else:
            if spec_cfg is None:
                raise RunnerError(
                    f"requested speculative_config {requested_spec!r} but the engine "
                    "resolved none -- speculation is silently disabled."
                )
            resolved_method = str(getattr(spec_cfg, "method", "") or "").lower()
            requested_method = self.config.spec_method
            if requested_method == "draft_model":
                if resolved_method and resolved_method not in ("draft_model", "draft"):
                    raise RunnerError(
                        f"requested a draft-model condition; engine resolved method="
                        f"{resolved_method!r}."
                    )
            elif resolved_method != requested_method:
                raise RunnerError(
                    f"requested spec_method={requested_method!r}; engine resolved "
                    f"method={resolved_method!r}. Refusing to record this as "
                    f"{requested_method!r}."
                )
            resolved_gamma = getattr(spec_cfg, "num_speculative_tokens", None)
            if resolved_gamma is not None and int(resolved_gamma) != int(self.config.num_speculative_tokens):
                raise RunnerError(
                    f"requested gamma={self.config.num_speculative_tokens}; engine "
                    f"resolved {resolved_gamma}."
                )

        self.resolved = {
            "model_path": resolved_model,
            "quantization": quant,
            "dtype": str(getattr(model_cfg, "dtype", "")),
            "speculative_config": repr(spec_cfg) if spec_cfg is not None else None,
        }

    def _engine_config(self) -> Any:
        engine = getattr(self._llm, "llm_engine", None)
        if engine is None:
            raise RunnerError("vLLM LLM object exposes no llm_engine; cannot verify resolution.")
        for attr in ("vllm_config", "engine_config", "model_config"):
            cfg = getattr(engine, attr, None)
            if cfg is not None:
                # model_config is the leaf, not the container; wrap it uniformly.
                if attr == "model_config":
                    class _Shim:
                        model_config = cfg
                        speculative_config = getattr(engine, "speculative_config", None)

                    return _Shim()
                return cfg
        raise RunnerError("could not locate vLLM's resolved config on the engine object.")

    # -- measurement -----------------------------------------------------------------

    def generate(self, prompt: str, fillers: list[str] | None = None) -> GenResult:
        if self._llm is None:
            raise RunnerError("generate() called before setup()")
        fillers = fillers or []
        expected_batch = self.config.batch_size
        batch = [prompt, *fillers[: expected_batch - 1]]
        if len(batch) != expected_batch:
            raise RunnerError(
                f"batch_size={expected_batch} needs {expected_batch - 1} filler prompts, "
                f"got {len(fillers)}. Refusing to measure a smaller batch than configured."
            )

        # Order matters, and it is not the obvious one.
        #
        # vLLM delivers SchedulerStats to the parent's stat loggers with a one-iteration
        # lag, so a request's final iteration arrives during the *next* engine activity.
        # With the TTFT probe running first, its tail landed after our reset and was
        # attributed to the measurement -- inflating accepted tokens by roughly one gamma
        # and producing counts that could not be reconciled with the tokens emitted.
        #
        # So the measured generation runs first against a freshly reset collector, its
        # acceptance is captured immediately, and only then does the probe run. The
        # probe's stats land in a collector nobody reads again this call.
        if self._collector is not None:
            self._collector.reset()
        self._spec_before = _spec_stats_snapshot(self._llm)

        _cuda_sync()
        t0 = time.perf_counter()
        outputs = self._llm.generate(batch, self._sampling, use_tqdm=False)
        _cuda_sync()
        total_ms = (time.perf_counter() - t0) * 1000.0

        self._accept_capture = self._capture_acceptance()
        target = self._match_target(outputs, prompt)

        # TTFT last, on a matched single-token request. See _measure_ttft_ms.
        ttft_ms = self._measure_ttft_ms(batch)
        return self._to_result(target, wall_ttft_ms=ttft_ms, wall_total_ms=total_ms)

    def _capture_acceptance(self) -> dict[str, Any] | None:
        """Freeze the collector's totals the instant the measured request finishes."""
        if self._collector is None or not self._collector.per_pos:
            return None
        return {
            "per_pos": list(self._collector.per_pos),
            "num_draft_tokens": self._collector.num_draft_tokens,
            "num_accepted_tokens": self._collector.num_accepted_tokens,
            "iterations": self._collector.iterations,
        }

    def _measure_ttft_ms(self, batch: list[str]) -> float:
        """Time-to-first-token, measured as a separate one-token request.

        vLLM's V1 offline ``LLM`` API does not populate per-token timestamps --
        ``RequestOutput.metrics`` comes back with ``first_token_time=None`` and
        ``finished_time=None`` -- so submit-to-first-token cannot be read off the full
        generation. The alternatives were:

        * the async streaming engine, which does expose real per-token timing but would
          make every measurement depend on an event loop and a different code path from
          the one the study is about;
        * dropping TTFT, which is not available: ``tpot_ms`` is *defined* as
          ``(total_ms - ttft_ms) / (output_tokens - 1)``, so without TTFT the primary
          metric of the whole study cannot be computed.

        So TTFT is measured by submitting the identical batch with ``max_tokens=1`` and
        timing it end to end. That request performs exactly the work TTFT names: prefill
        plus one decode step. Prefix caching is disabled, so the full generation that
        follows redoes prefill identically rather than reusing this one's KV cache.

        Two honest caveats, recorded with every record via ``timing_method``:
        it is a *matched* request rather than the same request, and both measurements
        include the same fixed Python-side submission overhead -- which largely cancels
        in the subtraction that produces ``tpot_ms``.
        """
        from vllm import SamplingParams

        one_token = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=1,
            ignore_eos=self.config.ignore_eos,
            seed=self.config.seed,
            logprobs=None,
            detokenize=True,
        )
        _cuda_sync()
        t0 = time.perf_counter()
        self._llm.generate(batch, one_token, use_tqdm=False)
        _cuda_sync()
        return (time.perf_counter() - t0) * 1000.0

    @staticmethod
    def _match_target(outputs: list[Any], prompt: str) -> Any:
        """Find the target request in the returned batch by prompt text.

        vLLM does not guarantee output order matches submission order, and a positional
        assumption would silently attribute a filler's timings to the target.
        """
        for out in outputs:
            if getattr(out, "prompt", None) == prompt:
                return out
        raise RunnerError("target prompt not found in vLLM's returned outputs.")

    def _to_result(self, out: Any, *, wall_ttft_ms: float, wall_total_ms: float) -> GenResult:
        # Prefer the engine's own per-token timestamps when a build populates them: they
        # exclude our submission overhead and are the better measurement. vLLM V1 leaves
        # them None, in which case we fall back to the wall-clock pair from generate().
        # Which one was used is recorded per record, because two records timed different
        # ways are not interchangeable.
        timing_method = "engine_metrics"
        metrics = getattr(out, "metrics", None)
        arrival = getattr(metrics, "arrival_time", None)
        first_token = getattr(metrics, "first_token_time", None)
        finished = getattr(metrics, "finished_time", None)

        if arrival is not None and first_token is not None and finished is not None:
            ttft_ms = (first_token - arrival) * 1000.0
            total_ms = (finished - arrival) * 1000.0
        else:
            timing_method = "wall_clock_matched_ttft"
            ttft_ms = wall_ttft_ms
            total_ms = wall_total_ms

        if total_ms <= ttft_ms:
            raise RunnerError(
                f"total_ms ({total_ms:.3f}) <= ttft_ms ({ttft_ms:.3f}) via {timing_method}. "
                "The full generation was not slower than a single token, which means the "
                "two measurements are not describing the work they claim to."
            )

        completion = out.outputs[0]
        token_ids = getattr(completion, "token_ids", None)
        if token_ids is None:
            raise RunnerError("completion carries no token_ids; cannot count output tokens.")

        try:
            hist, proposed = self._acceptance(out, output_tokens=len(token_ids))
        except RunnerError:
            if not self.allow_missing_acceptance or self.config.spec_method == "none":
                raise
            # Recorded as absent, not as zero. Speed figures still use this row; every
            # acceptance-derived figure refuses it.
            self.acceptance_unavailable = True
            hist, proposed = [], None
        return GenResult(
            ttft_ms=ttft_ms,
            total_ms=total_ms,
            output_tokens=len(token_ids),
            output_text=completion.text,
            prompt_tokens=len(getattr(out, "prompt_token_ids", []) or []),
            accept_length_histogram=hist,
            draft_tokens_proposed=proposed,
            extra={
                "finish_reason": getattr(completion, "finish_reason", None),
                "timing_method": timing_method,
            },
        )

    # -- acceptance extraction -------------------------------------------------------

    def _acceptance(self, out: Any, *, output_tokens: int) -> tuple[list[int], int | None]:
        """Per-request accepted-run-length histogram.

        vLLM has moved this between engine versions, so we look in each known place and
        raise a specific, actionable error if none of them is populated. Guessing here
        would produce an acceptance rate that is merely plausible, and every gamma
        conclusion in the study rests on it.
        """
        if self.config.spec_method == "none":
            return [], None

        gamma = int(self.config.num_speculative_tokens)
        tried: list[str] = []

        # Preferred: our own accumulation of the per-iteration scheduler stats. This is
        # the only path that sees every iteration of the request rather than the last.
        tried.append("registered stat-logger collector (SpecDecodingStats)")
        capture = self._accept_capture
        if capture and capture["per_pos"]:
            hist, derived = self._hist_from_per_pos(
                list(capture["per_pos"]), gamma, output_tokens
            )
            # Cross-check the engine's own step count against the one implied by the
            # tokens emitted. A mismatch means stats from outside this request leaked in
            # (or this request's tail did not arrive), and the histogram would be a
            # different run's shape wearing this run's label.
            steps = sum(hist)
            iterations = capture.get("iterations") or 0
            if iterations and iterations != steps:
                raise RunnerError(
                    f"engine reported {iterations} speculative iterations but the tokens "
                    f"emitted imply {steps} verification steps (output_tokens="
                    f"{output_tokens}, accepted={sum(k * n for k, n in enumerate(hist))}). "
                    "Acceptance statistics from outside this request have leaked in, or "
                    "its final iteration never arrived. Refusing to record."
                )
            proposed = capture.get("num_draft_tokens") or None
            return hist, (proposed if proposed else derived)

        # V1: per-request counts of tokens accepted at each speculative position.
        per_pos = getattr(out, "num_accepted_tokens_per_pos", None)
        tried.append("RequestOutput.num_accepted_tokens_per_pos")
        if per_pos:
            return self._hist_from_per_pos(list(per_pos), gamma, output_tokens)

        metrics = getattr(out, "metrics", None)
        for attr in ("spec_token_acceptance_counts", "accepted_token_counts"):
            counts = getattr(metrics, attr, None)
            tried.append(f"RequestOutput.metrics.{attr}")
            if counts:
                return self._hist_from_per_pos(list(counts), gamma, output_tokens)

        # V1: cumulative counters on the engine's stats objects, differenced across
        # this one request. Exact at batch 1, and reconciled against output_tokens
        # below -- if the diff does not describe the tokens actually emitted, that
        # reconciliation raises rather than recording a plausible wrong histogram.
        tried.append("engine spec-decode stats snapshot diff")
        after = _spec_stats_snapshot(self._llm)
        before = self._spec_before
        if after and before and "per_pos" in after and "per_pos" in before:
            width = max(len(after["per_pos"]), len(before["per_pos"]))
            a = list(after["per_pos"]) + [0] * (width - len(after["per_pos"]))
            b = list(before["per_pos"]) + [0] * (width - len(before["per_pos"]))
            delta = [x - y for x, y in zip(a, b)]
            if any(d < 0 for d in delta):
                raise RunnerError(
                    f"cumulative acceptance counters went backwards across one request "
                    f"({before['per_pos']} -> {after['per_pos']}). The engine reset them "
                    "mid-measurement, so the difference is not this request's acceptance."
                )
            if any(d > 0 for d in delta):
                return self._hist_from_per_pos(delta, gamma, output_tokens)

        # V0: the spec-decode worker aggregated per step on the engine's stat logger.
        engine = getattr(self._llm, "llm_engine", None)
        spec_metrics = getattr(engine, "_last_spec_decode_metrics", None)
        tried.append("llm_engine._last_spec_decode_metrics")
        if spec_metrics is not None:
            accepted = getattr(spec_metrics, "accepted_tokens", None)
            drafted = getattr(spec_metrics, "draft_tokens", None)
            if accepted is not None and drafted is not None and gamma > 0:
                steps = int(drafted) // gamma
                if steps > 0:
                    # Aggregate-only: we know the totals but not the shape. A flat
                    # histogram would fake a distribution we did not measure, so this
                    # path is refused for figure 06 and reported as such.
                    raise RunnerError(
                        "only aggregate acceptance totals are available from this vLLM "
                        "build (accepted=%s, drafted=%s). Figure 06 needs the per-step "
                        "distribution and a flattened histogram would be fabricated. "
                        "Use a vLLM build exposing per-request acceptance counts."
                        % (accepted, drafted)
                    )

        raise RunnerError(
            "could not extract per-request acceptance counts from vLLM. Looked at: "
            + ", ".join(tried)
            + ". Wire up the accessor for this vLLM version in "
            "runners/vllm_runner.py::_acceptance rather than estimating acceptance."
        )

    @staticmethod
    def _hist_from_per_pos(per_pos: list[int], gamma: int, output_tokens: int) -> tuple[list[int], int]:
        """Convert per-position acceptance counts into a run-length histogram.

        ``per_pos[i]`` is the number of steps in which the token drafted at position
        ``i`` was accepted. Acceptance is prefix-closed -- position ``i`` can only be
        accepted if every earlier position was -- so ``per_pos`` is non-increasing and
        the number of steps with a run of exactly ``k`` is ``per_pos[k-1] - per_pos[k]``.

        The zero-acceptance bin cannot be read off ``per_pos`` at all, but it is exactly
        recoverable: every verification step emits precisely one token the target chose
        itself, so ``steps == output_tokens - accepted_tokens``, and ``hist[0]`` is
        whatever is left after the non-zero runs. Leaving that bin at zero instead would
        bias ``mean_accept_length`` upward by exactly the steps that accepted nothing --
        the worst-performing steps, silently deleted.
        """
        if not per_pos:
            raise RunnerError("empty per-position acceptance counts")
        counts = [int(x) for x in per_pos[:gamma]]
        if any(b > a for a, b in zip(counts, counts[1:])):
            raise RunnerError(
                f"per-position acceptance counts are not non-increasing ({counts}); the "
                "prefix-closure assumption behind the histogram does not hold for this "
                "engine, so the conversion would be wrong."
            )

        hist = [0] * (gamma + 1)
        for k in range(1, gamma + 1):
            higher = counts[k] if k < len(counts) else 0
            hist[k] = counts[k - 1] - higher

        accepted = sum(counts)
        steps = int(output_tokens) - accepted
        zero_bin = steps - counts[0]
        if steps <= 0 or zero_bin < 0:
            raise RunnerError(
                f"acceptance counts are inconsistent with the output length: "
                f"output_tokens={output_tokens}, accepted={accepted}, steps={steps}, "
                f"steps_with_acceptance={counts[0]}. Refusing to record a histogram that "
                "cannot be reconciled with the tokens actually emitted."
            )
        hist[0] = zero_bin
        return hist, gamma * steps
