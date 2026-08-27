"""HF Transformers runner -- drafter experiments and quality scoring.

Used for the things vLLM cannot express as a knob: MTP drafters, and acceptance
behaviour under ``num_assistant_tokens`` / ``num_assistant_tokens_schedule``, where the
schedule itself is the independent variable.

**Every record this runner emits carries ``latency_valid: False``, and the analysis
layer refuses to let such a record reach a speed figure.** This is not caution, it is
arithmetic: HF's per-token Python dispatch overhead sits in the same range as the effect
the study measures, and it is paid once per *step* rather than once per token. Because
speculative decoding emits several tokens per step, that overhead is amortized unevenly
between the speculative and baseline conditions, and it biases measured speedup
*downward* -- flattening exactly the curve figures 03 and 04 are built on. An HF speedup
number is not noisy, it is wrong in a known direction.

What HF is trusted for is acceptance counts, which are exact integers read from the
candidate generator itself.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator

from core.schema import RunConfig
from runners.base import GenResult, RunnerError

_DTYPE_MAP = {"bf16": "bfloat16"}


class HFRunner:
    """One model (plus optional assistant) for exactly one condition."""

    def __init__(
        self,
        config: RunConfig,
        *,
        device: str = "auto",
        num_assistant_tokens_schedule: str | None = None,
    ) -> None:
        if config.stack != "hf":
            raise RunnerError(f"HFRunner got stack={config.stack!r}")
        self.config = config
        self.device = device
        self.schedule = num_assistant_tokens_schedule
        self.resolved: dict[str, Any] = {}
        self._model: Any = None
        self._assistant: Any = None
        self._tokenizer: Any = None

    # -- lifecycle ------------------------------------------------------------------

    def setup(self) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RunnerError("transformers/torch are not installed in this interpreter.") from exc

        cfg = self.config
        if cfg.target_dtype not in _DTYPE_MAP:
            raise RunnerError(
                f"the HF arm runs bf16 only; target_dtype={cfg.target_dtype!r} belongs to "
                "the vLLM arm, which has the production quantization kernels."
            )
        dtype = getattr(torch, _DTYPE_MAP[cfg.target_dtype])

        self._tokenizer = AutoTokenizer.from_pretrained(cfg.target_model)
        self._model = AutoModelForCausalLM.from_pretrained(
            cfg.target_model, torch_dtype=dtype, device_map=self.device
        )
        self._model.eval()

        if cfg.spec_method != "none":
            if not cfg.draft_model:
                raise RunnerError(
                    f"spec_method={cfg.spec_method!r} on the HF arm requires an explicit "
                    "draft_model checkpoint to load as the assistant."
                )
            self._assistant = AutoModelForCausalLM.from_pretrained(
                cfg.draft_model, torch_dtype=dtype, device_map=self.device
            )
            self._assistant.eval()

        torch.manual_seed(cfg.seed)
        self.resolved = {
            "model_path": getattr(self._model.config, "_name_or_path", cfg.target_model),
            "assistant_path": (
                getattr(self._assistant.config, "_name_or_path", cfg.draft_model)
                if self._assistant is not None else None
            ),
            "speculative_config": {
                "method": cfg.spec_method,
                "num_assistant_tokens": cfg.num_speculative_tokens,
                "num_assistant_tokens_schedule": self.schedule or cfg.gamma_schedule,
            },
        }
        self._verify_resolution()

    def close(self) -> None:
        import contextlib
        import gc

        self._model = None
        self._assistant = None
        self._tokenizer = None
        gc.collect()
        with contextlib.suppress(ImportError):
            import torch

            torch.cuda.empty_cache()

    def _verify_resolution(self) -> None:
        """Catch a silent substitution, the same way the vLLM arm does."""
        import os

        got = str(self.resolved["model_path"])
        want = self.config.target_model
        if os.path.basename(got.rstrip("/")) != os.path.basename(want.rstrip("/")):
            raise RunnerError(f"resolved model {got!r} does not match requested {want!r}")

    # -- measurement -----------------------------------------------------------------

    def generate(self, prompt: str, fillers: list[str] | None = None) -> GenResult:
        if self._model is None:
            raise RunnerError("generate() called before setup()")
        if self.config.batch_size != 1:
            raise RunnerError("the HF arm measures batch 1 only; batch conditions run on vLLM.")

        import torch

        cfg = self.config
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
        prompt_tokens = int(inputs["input_ids"].shape[-1])

        gen_kwargs: dict[str, Any] = {
            "max_new_tokens": cfg.max_tokens,
            "min_new_tokens": cfg.max_tokens if cfg.ignore_eos else None,
            "do_sample": cfg.temperature > 0,
            "temperature": cfg.temperature if cfg.temperature > 0 else None,
            "return_dict_in_generate": True,
            "output_scores": False,
            "use_cache": True,
        }
        # ignore_eos has no direct HF equivalent; min_new_tokens == max_new_tokens plus a
        # suppressed EOS is the exact behavioural match, and the schema guard downstream
        # verifies the token count actually came out right.
        if cfg.ignore_eos:
            eos_id = self._tokenizer.eos_token_id
            if eos_id is None:
                raise RunnerError("tokenizer reports no eos_token_id; cannot enforce ignore_eos.")
            gen_kwargs["suppress_tokens"] = [eos_id]
        gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

        if self._assistant is not None:
            gen_kwargs["assistant_model"] = self._assistant
            gen_kwargs["num_assistant_tokens"] = cfg.num_speculative_tokens
            gen_kwargs["num_assistant_tokens_schedule"] = self.schedule or (
                "constant" if cfg.gamma_schedule == "constant" else "heuristic"
            )

        first_token_at: list[float] = []
        criteria = _FirstTokenTimer(first_token_at)
        gen_kwargs["stopping_criteria"] = [criteria]

        with _acceptance_recorder(active=self._assistant is not None) as matches:
            _sync()
            t0 = time.perf_counter()
            with torch.no_grad():
                out = self._model.generate(**inputs, **gen_kwargs)
            _sync()
            t1 = time.perf_counter()

        if not first_token_at:
            raise RunnerError("stopping criteria never fired; cannot measure TTFT.")

        sequences = out.sequences if hasattr(out, "sequences") else out
        new_tokens = sequences[0][prompt_tokens:]
        output_tokens = int(new_tokens.shape[-1])
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)

        hist: list[int] = []
        proposed: int | None = None
        if self._assistant is not None:
            if not matches:
                raise RunnerError(
                    "assisted generation ran but no acceptance counts were captured. The "
                    "candidate-generator hook did not fire -- check the transformers "
                    "version against runners/hf_runner.py::_acceptance_recorder rather "
                    "than recording a run with no acceptance data."
                )
            gamma = int(cfg.num_speculative_tokens)
            hist = [0] * (gamma + 1)
            for k in matches:
                hist[min(int(k), gamma)] += 1
            schedule = gen_kwargs.get("num_assistant_tokens_schedule", "constant")
            if schedule != "constant":
                # Under a heuristic schedule HF adapts the draft length every step, so
                # proposals cannot be derived from gamma and the step count. Deriving it
                # anyway would put a wrong denominator under every acceptance rate.
                raise RunnerError(
                    f"num_assistant_tokens_schedule={schedule!r} varies the per-step draft "
                    "length, which this runner cannot observe. Record acceptance for "
                    "heuristic schedules only once the per-step proposal count is captured."
                )
            proposed = gamma * len(matches)

        return GenResult(
            ttft_ms=(first_token_at[0] - t0) * 1000.0,
            total_ms=(t1 - t0) * 1000.0,
            output_tokens=output_tokens,
            output_text=text,
            prompt_tokens=prompt_tokens,
            accept_length_histogram=hist,
            draft_tokens_proposed=proposed,
            extra={"latency_valid": False},
        )


class _FirstTokenTimer:
    """Stopping criterion used only for its side effect: timestamping the first step.

    Returns False always, so it never stops anything.
    """

    def __init__(self, sink: list[float]) -> None:
        self._sink = sink

    def __call__(self, input_ids: Any, scores: Any, **kwargs: Any) -> bool:
        if not self._sink:
            _sync()
            self._sink.append(time.perf_counter())
        return False


def _sync() -> None:
    """Block until queued GPU work is done, so timestamps mean what they say."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


@contextmanager
def _acceptance_recorder(*, active: bool) -> Iterator[list[int]]:
    """Record ``num_matches`` per verification step from the candidate generator.

    ``CandidateGenerator.update_candidate_strategy(input_ids, scores, num_matches)`` is
    called by HF once per step with the exact number of drafted tokens accepted. Reading
    it there gives integer counts rather than an inference from output length -- which
    is the whole reason the HF arm is worth running.
    """
    matches: list[int] = []
    if not active:
        yield matches
        return

    try:
        from transformers.generation import candidate_generator as cg_module
    except ImportError as exc:
        raise RunnerError(
            "transformers.generation.candidate_generator is unavailable; cannot capture "
            "acceptance counts on this version."
        ) from exc

    target = getattr(cg_module, "AssistedCandidateGenerator", None)
    if target is None or not hasattr(target, "update_candidate_strategy"):
        raise RunnerError(
            "AssistedCandidateGenerator.update_candidate_strategy not found; the hook "
            "this runner reads acceptance from has moved in this transformers version."
        )

    original = target.update_candidate_strategy

    def patched(self: Any, input_ids: Any, scores: Any, num_matches: int, *args: Any, **kwargs: Any):
        matches.append(int(num_matches))
        return original(self, input_ids, scores, num_matches, *args, **kwargs)

    target.update_candidate_strategy = patched
    try:
        yield matches
    finally:
        target.update_candidate_strategy = original
