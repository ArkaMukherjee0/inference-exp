"""llama.cpp runner -- the CPU wall-clock arm.

A subprocess wrapper around llama.cpp's speculative binary, with one unusual property:
it refuses to trust its own knowledge of the command line.

llama.cpp has renamed its draft-length and draft-model flags more than once, and unknown
flags are frequently accepted-and-ignored rather than rejected. A hardcoded flag list
would therefore fail *silently*, producing a full sweep of non-speculative runs labelled
as speculative -- the whole CPU arm, quietly worthless, with no symptom except a
suspiciously flat speedup curve. So every flag is resolved by parsing ``--help`` at
startup, and an absent flag is a hard error.

Timing semantics on this arm
----------------------------
llama.cpp reports its own performance counters, which separate prefill from decode
cleanly and exclude model load. We use them directly:

* ``ttft_ms``  -- the prompt eval (prefill) time.
* ``total_ms`` -- prefill + decode.

TTFT here is therefore prefill time rather than submit-to-first-byte at the process
boundary, which would fold in interpreter and model-load overhead. Cross-stack TTFT is
never compared anyway (``stack`` is part of every grouping key); what matters is that
the definition is identical across every condition *within* this arm.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from core.schema import RunConfig
from runners.base import GenResult, RunnerError

# Flag aliases, newest first. The first alias present in --help wins.
_FLAG_ALIASES: dict[str, tuple[str, ...]] = {
    "model": ("-m", "--model"),
    "draft_model": ("-md", "--model-draft"),
    # Newest first matters here beyond tidiness: b10655 still *lists* the removed
    # "--draft, --draft-n, --draft-max" line in --help, as a notice telling you to use
    # --spec-draft-n-max instead. Matching the old name would bind a flag the binary
    # rejects, so the current name has to be tried first.
    "draft_len": ("--spec-draft-n-max", "--draft-max", "--draft", "-ndraft", "--n-draft"),
    "n_predict": ("-n", "--n-predict", "--predict"),
    "prompt": ("-p", "--prompt"),
    "threads": ("-t", "--threads"),
    "seed": ("-s", "--seed"),
    "temp": ("--temp", "--temperature"),
    "ignore_eos": ("--ignore-eos",),
    "no_warmup": ("--no-warmup",),
}

# llama.cpp's perf counters. Format has been stable for a long time, but every one of
# these is asserted present rather than defaulted.
_RE_PROMPT_EVAL = re.compile(
    r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*tokens", re.IGNORECASE
)
# Negative lookbehind on "prompt ": llama.cpp prints both "prompt eval time" (prefill)
# and "eval time" (decode), and a pattern that matched either would silently read
# prefill as decode -- inverting the two halves of every CPU timing.
_RE_EVAL = re.compile(
    r"(?<!prompt )eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*(\d+)\s*runs", re.IGNORECASE
)
# The speculative example times prefill and the decode loop itself. On builds where the
# target context batches gamma+1 tokens per verification step, these are the only honest
# split -- see the note in parse_output.
_RE_ENCODED = re.compile(r"encoded\s+(\d+)\s+tokens in\s+([0-9.]+)\s+seconds")
_RE_DECODED = re.compile(r"decoded\s+(\d+)\s+tokens in\s+([0-9.]+)\s+seconds")
_RE_N_DRAFTED = re.compile(r"n_drafted\s*=\s*(\d+)")
_RE_N_ACCEPT = re.compile(r"n_accept\s*=\s*(\d+)")
_RE_N_DRAFT = re.compile(r"n_draft\s*=\s*(\d+)")
# Per-step acceptance, emitted by builds that log each verification step.
_RE_STEP_ACCEPT = re.compile(r"accepted\s+(\d+)\s+draft", re.IGNORECASE)


class LlamaCppRunner:
    """One llama.cpp subprocess invocation per prompt."""

    def __init__(
        self,
        config: RunConfig,
        *,
        binary: str | os.PathLike[str],
        model_path: str | os.PathLike[str],
        draft_model_path: str | os.PathLike[str] | None = None,
        baseline_binary: str | os.PathLike[str] | None = None,
        extra_args: list[str] | None = None,
        baseline_extra_args: list[str] | None = None,
    ) -> None:
        if config.stack != "llamacpp":
            raise RunnerError(f"LlamaCppRunner got stack={config.stack!r}")
        self.config = config
        self.model_path = str(model_path)
        self.draft_model_path = str(draft_model_path) if draft_model_path else None

        # Two binaries, one build. llama.cpp's speculative example *requires* a draft
        # model, so it cannot produce the gamma=0 baseline; the plain completion tool
        # produces the baseline and cannot speculate. Both must come from the same
        # source tree and compiler flags, or the baseline and the speculative arm are
        # measuring different CPU kernels and the speedup is an artefact of the build.
        if config.spec_method == "none":
            if not baseline_binary:
                raise RunnerError(
                    "spec_method='none' on the llama.cpp arm needs model.baseline_binary: "
                    "llama-speculative refuses to run without a draft model, so it cannot "
                    "measure the non-speculative baseline."
                )
            self.binary = str(baseline_binary)
            self.extra_args = list(extra_args or []) + list(baseline_extra_args or [])
        else:
            self.binary = str(binary)
            self.extra_args = list(extra_args or [])
        self.resolved: dict[str, Any] = {}
        self._flags: dict[str, str] = {}

    # -- lifecycle ------------------------------------------------------------------

    def setup(self) -> None:
        if shutil.which(self.binary) is None and not Path(self.binary).is_file():
            raise RunnerError(f"llama.cpp binary not found: {self.binary!r}")
        for path, label in ((self.model_path, "model"), (self.draft_model_path, "draft model")):
            if path and not Path(path).is_file():
                raise RunnerError(f"{label} GGUF not found: {path!r}")

        if self.config.spec_method not in ("none", "draft_model"):
            raise RunnerError(
                f"llama.cpp arm supports spec_method 'none' or 'draft_model', not "
                f"{self.config.spec_method!r}."
            )
        if self.config.spec_method == "draft_model" and not self.draft_model_path:
            raise RunnerError("spec_method='draft_model' requires draft_model_path")
        if self.config.num_threads is None:
            raise RunnerError(
                "num_threads must be set for the CPU arm. On a hybrid P/E-core desktop an "
                "unpinned, unbounded thread count makes timings wander between runs."
            )

        self._flags = self._resolve_flags()
        self.resolved = {
            "model_path": self.model_path,
            "draft_model_path": self.draft_model_path,
            "flags": dict(self._flags),
            "binary": self.binary,
        }

    def close(self) -> None:  # nothing persistent to tear down
        return

    def _resolve_flags(self) -> dict[str, str]:
        """Parse ``--help`` and bind each flag we need to a name this build accepts."""
        try:
            proc = subprocess.run(
                [self.binary, "--help"], capture_output=True, text=True, timeout=60
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise RunnerError(f"could not run {self.binary!r} --help: {exc}") from exc
        help_text = (proc.stdout or "") + (proc.stderr or "")
        if not help_text.strip():
            raise RunnerError(f"{self.binary!r} --help produced no output; cannot resolve flags.")

        needed = ["model", "n_predict", "prompt", "threads", "seed", "temp"]
        if self.config.ignore_eos:
            needed.append("ignore_eos")
        if self.config.spec_method == "draft_model":
            needed += ["draft_model", "draft_len"]

        resolved: dict[str, str] = {}
        missing: list[str] = []
        for key in needed:
            for alias in _FLAG_ALIASES[key]:
                if re.search(rf"(?<![\w-]){re.escape(alias)}(?![\w-])", help_text):
                    resolved[key] = alias
                    break
            else:
                missing.append(f"{key} (tried {', '.join(_FLAG_ALIASES[key])})")

        if missing:
            raise RunnerError(
                f"{self.binary!r} does not advertise required flag(s): {'; '.join(missing)}. "
                "Refusing to guess a flag name -- an ignored flag would silently produce "
                "non-speculative runs labelled as speculative."
            )
        # Optional, but worth using when present: llama.cpp's own warmup would otherwise
        # land inside our first measured run.
        for alias in _FLAG_ALIASES["no_warmup"]:
            if alias in help_text:
                resolved["no_warmup"] = alias
                break
        return resolved

    # -- measurement -----------------------------------------------------------------

    def generate(self, prompt: str, fillers: list[str] | None = None) -> GenResult:
        if not self._flags:
            raise RunnerError("generate() called before setup()")
        if self.config.batch_size != 1:
            raise RunnerError(
                f"batch_size={self.config.batch_size}: the llama.cpp arm measures batch-1 "
                "latency only. Batch conditions belong to the vLLM arm."
            )

        cmd = self._build_command(prompt)
        env = dict(os.environ)
        env["OMP_NUM_THREADS"] = str(self.config.num_threads)
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
        except subprocess.TimeoutExpired as exc:
            raise RunnerError(f"llama.cpp timed out after 3600s: {' '.join(cmd)}") from exc
        if proc.returncode != 0:
            raise RunnerError(
                f"llama.cpp exited {proc.returncode}.\ncmd: {' '.join(cmd)}\n"
                f"stderr tail:\n{(proc.stderr or '')[-2000:]}"
            )
        return self.parse_output(
            stdout=proc.stdout or "",
            stderr=proc.stderr or "",
            config=self.config,
            prompt=prompt,
        )

    def _build_command(self, prompt: str) -> list[str]:
        f = self._flags
        cfg = self.config
        cmd = [
            self.binary,
            f["model"], self.model_path,
            f["prompt"], prompt,
            f["n_predict"], str(cfg.max_tokens),
            f["threads"], str(cfg.num_threads),
            f["seed"], str(cfg.seed),
            f["temp"], str(cfg.temperature),
        ]
        if cfg.ignore_eos:
            cmd.append(f["ignore_eos"])
        if "no_warmup" in f:
            cmd.append(f["no_warmup"])
        if cfg.spec_method == "draft_model":
            cmd += [f["draft_model"], self.draft_model_path, f["draft_len"], str(cfg.num_speculative_tokens)]
        cmd += self.extra_args
        return cmd

    # -- parsing (pure, so it can be tested against a captured real sample) -----------

    @staticmethod
    def parse_output(
        *, stdout: str, stderr: str, config: RunConfig, prompt: str | None = None
    ) -> GenResult:
        """Turn llama.cpp's output into a GenResult.

        Kept static and side-effect free so ``tests/fixtures/llamacpp_speculative.txt``
        -- a captured sample of real output -- can exercise it without a binary.
        """
        blob = stdout + "\n" + stderr

        hist: list[int] = []
        proposed: int | None = None
        if config.spec_method == "none":
            # Non-speculative: one token per decode call, so llama.cpp's own counters
            # split prefill from decode exactly and 'unaccounted time' stays near 0%.
            m_prompt = _RE_PROMPT_EVAL.search(blob)
            if not m_prompt:
                raise RunnerError(
                    "could not find llama.cpp's 'prompt eval time' counter in the output; "
                    "without it there is no prefill measurement. Output tail:\n" + blob[-1500:]
                )
            prefill_ms, prompt_tokens = float(m_prompt.group(1)), int(m_prompt.group(2))

            m_eval = _RE_EVAL.search(blob)
            if not m_eval:
                raise RunnerError(
                    "could not find llama.cpp's 'eval time' counter in the output; there is "
                    "no decode measurement. Output tail:\n" + blob[-1500:]
                )
            decode_ms = float(m_eval.group(1))
            # 'runs' counts decode calls, and the first token is not one of them: it is
            # sampled from the logits the prefill batch already produced. Generated
            # tokens are therefore runs + 1, which is what ignore_eos pins to
            # max_tokens. Reading 'runs' directly reports one token short of every
            # generation and fails the ignore_eos guard on every baseline record.
            output_tokens = int(m_eval.group(2)) + 1
        else:
            # Speculative: the perf counters cannot be used. A verification step decodes
            # gamma+1 tokens at once, and llama.cpp books every multi-token decode under
            # 'prompt eval time' -- so that counter is prefill *plus* every verification
            # batch, while 'eval time' collapses to 0.00 ms / 1 runs. Reading them would
            # inflate TTFT and drop decode out of the total entirely: a plausible-looking
            # wrong number, which is exactly what this arm refuses to produce. The
            # example's own encode/decode clocks bracket the same two phases, exclude
            # model load, and are the only honest split under speculation.
            m_enc = _RE_ENCODED.search(blob)
            m_dec = _RE_DECODED.search(blob)
            if not m_enc or not m_dec:
                raise RunnerError(
                    "could not find the speculative example's 'encoded N tokens in X "
                    "seconds' / 'decoded N tokens in X seconds' lines, which are this "
                    "arm's prefill and decode measurements under speculation. The perf "
                    "counters are not a substitute for them. Output tail:\n" + blob[-1500:]
                )
            prompt_tokens = int(m_enc.group(1))
            prefill_ms = float(m_enc.group(2)) * 1000.0
            decode_ms = float(m_dec.group(2)) * 1000.0
            decoded_tokens = int(m_dec.group(1))
            n_drafted = _search_int(_RE_N_DRAFTED, blob, "n_drafted")
            n_accept = _search_int(_RE_N_ACCEPT, blob, "n_accept")
            gamma_reported = _RE_N_DRAFT.search(blob)
            if gamma_reported and int(gamma_reported.group(1)) != int(config.num_speculative_tokens):
                raise RunnerError(
                    f"requested gamma={config.num_speculative_tokens} but llama.cpp reports "
                    f"n_draft={gamma_reported.group(1)}; the flag was not honoured."
                )
            steps = [int(x) for x in _RE_STEP_ACCEPT.findall(blob)]
            if not steps:
                raise RunnerError(
                    "llama.cpp reported aggregate acceptance (n_drafted=%d, n_accept=%d) but "
                    "no per-step counts, so the accepted-run-length distribution figure 06 "
                    "needs was not measured. Rebuild with per-step speculative logging "
                    "enabled; a histogram synthesised from the totals would be fabricated."
                    % (n_drafted, n_accept)
                )
            if sum(steps) != n_accept:
                raise RunnerError(
                    f"per-step accepted counts sum to {sum(steps)} but n_accept={n_accept}; "
                    "the parse disagrees with llama.cpp's own total."
                )
            gamma = int(config.num_speculative_tokens)
            hist = [0] * (gamma + 1)
            for k in steps:
                if k > gamma:
                    raise RunnerError(f"a step accepted {k} tokens with gamma={gamma}")
                hist[k] += 1
            proposed = n_drafted
            output_tokens = n_accept + len(steps)
            if output_tokens != decoded_tokens:
                raise RunnerError(
                    f"accepted ({n_accept}) plus bonus tokens ({len(steps)}) is "
                    f"{output_tokens}, but llama.cpp decoded {decoded_tokens}; the step "
                    "counts do not describe the run that was timed."
                )

        return GenResult(
            ttft_ms=prefill_ms,
            total_ms=prefill_ms + decode_ms,
            output_tokens=output_tokens,
            output_text=_extract_completion(stdout, prompt),
            prompt_tokens=prompt_tokens,
            accept_length_histogram=hist,
            draft_tokens_proposed=proposed,
            extra={"prefill_ms": prefill_ms, "decode_ms": decode_ms},
        )


def _search_int(pattern: re.Pattern[str], blob: str, label: str) -> int:
    m = pattern.search(blob)
    if not m:
        raise RunnerError(
            f"could not find {label} in llama.cpp output; speculative statistics are "
            "missing, so this run cannot be recorded as a speculative measurement."
        )
    return int(m.group(1))


def _extract_completion(stdout: str, prompt: str | None = None) -> str:
    """The generated text, with the echoed prompt and any perf block stripped.

    llama.cpp writes the prompt back out ahead of the completion (after the BOS piece).
    Leaving it in would record the question as part of the model's answer, and GSM8K
    scoring reads the last number in the text -- which the prompt's own numbers could
    become for a completion that produced none.
    """
    text = stdout
    for marker in ("llama_perf_", "llama_print_timings", "common_perf_print"):
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    if prompt:
        idx = text.find(prompt)
        if idx != -1:
            text = text[idx + len(prompt):]
    return text.strip()
