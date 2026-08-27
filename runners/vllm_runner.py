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
                        "num_accepted_tokens_per_position")
_SPEC_TOTAL_FIELDS = ("num_accepted_tokens", "num_draft_tokens", "num_drafts")


def _find_spec_stats(root: Any, depth: int = 0, seen: set[int] | None = None) -> Any:
    """Locate an object carrying cumulative per-position acceptance counts."""
    seen = seen if seen is not None else set()
    if root is None or id(root) in seen or depth > 3:
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
        if callable(child) or isinstance(
            child, (str, bytes, int, float, bool, list, tuple, dict, set)
        ):
            continue
        found = _find_spec_stats(child, depth + 1, seen)
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
        if isinstance(value, (list, tuple)) and value:
            snap["per_pos"] = [int(x) for x in value]
            break
    for name in _SPEC_TOTAL_FIELDS:
        value = getattr(stats, name, None)
        if isinstance(value, int):
            snap[name] = value
    return snap or None


# Quantization methods vLLM may report for a 4-bit weight / 16-bit activation checkpoint.
_W4A16_METHODS = frozenset({"awq", "awq_marlin", "gptq", "gptq_marlin", "compressed-tensors"})
_FP8_METHODS = frozenset({"fp8", "compressed-tensors", "modelopt"})


class VLLMRunner:
    """One vLLM engine for exactly one condition. Never reused across conditions."""

    def __init__(self, config: RunConfig, *, log_extra: dict[str, Any] | None = None) -> None:
        if config.stack != "vllm":
            raise RunnerError(f"VLLMRunner got stack={config.stack!r}")
        self.config = config
        self.resolved: dict[str, Any] = {}
        self._llm: Any = None
        self._sampling: Any = None
        self._spec_before: dict[str, Any] | None = None
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

        # TTFT first, on a matched single-token request. See _measure_ttft_ms.
        ttft_ms = self._measure_ttft_ms(batch)

        # Snapshot cumulative acceptance AFTER the TTFT probe, so that probe's own
        # drafting is not attributed to this measurement.
        self._spec_before = _spec_stats_snapshot(self._llm)

        _cuda_sync()
        t0 = time.perf_counter()
        outputs = self._llm.generate(batch, self._sampling, use_tqdm=False)
        _cuda_sync()
        total_ms = (time.perf_counter() - t0) * 1000.0

        target = self._match_target(outputs, prompt)
        return self._to_result(target, wall_ttft_ms=ttft_ms, wall_total_ms=total_ms)

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

        hist, proposed = self._acceptance(out, output_tokens=len(token_ids))
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
