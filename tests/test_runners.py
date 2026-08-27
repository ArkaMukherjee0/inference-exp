"""Runner tests.

The llama.cpp parser is exercised against a captured sample of real output committed
under ``tests/fixtures/``. The vLLM and HF runners cannot be exercised without a GPU, so
what is tested here is their *refusal* behaviour -- the guards that stop a plausible but
wrong measurement -- plus the pure record-assembly logic they share.
"""

from __future__ import annotations

import pytest

from core.schema import RunConfig, validate_record
from runners.base import GenResult, RunnerError, build_record
from runners.llamacpp_runner import LlamaCppRunner
from runners.vllm_runner import VLLMRunner

ENV = {
    "hostname": "test-host",
    "stack_version": "llamacpp:build 4231",
    "driver": "Windows 11",
    "env": {"captured_at": "2026-08-27T00:00:00.000+00:00"},
}


def _cpu_config(**overrides) -> RunConfig:
    fields = {
        "target_model": "Llama-3.1-8B-Instruct-Q4_K_M",
        "draft_model": "Llama-3.2-1B-Instruct-Q4_K_M",
        "target_dtype": "w4a16",
        "stack": "llamacpp",
        "platform": "cpu",
        "spec_method": "draft_model",
        "num_speculative_tokens": 4,
        "max_tokens": 73,
        "num_threads": 8,
    }
    fields.update(overrides)
    return RunConfig(**fields)


# -- llama.cpp parsing against real captured output ------------------------------------


def test_parses_the_captured_sample(llamacpp_output):
    cfg = _cpu_config()
    result = LlamaCppRunner.parse_output(stdout=llamacpp_output, stderr="", config=cfg)

    assert result.ttft_ms == pytest.approx(669.0)
    assert result.total_ms == pytest.approx(669.0 + 4644.0)
    assert result.prompt_tokens == 39
    # 53 accepted + 20 bonus tokens = 73, matching n_predict in the sample.
    assert result.output_tokens == 73
    assert result.draft_tokens_proposed == 76


def test_prefill_is_not_mistaken_for_decode(llamacpp_output):
    """The two halves of the timing must not be swapped or conflated.

    Under speculation the perf counters are actively misleading: a verification step
    decodes gamma+1 tokens at once, llama.cpp books every multi-token decode as
    'prompt eval time', and 'eval time' collapses to 0.00 ms / 1 runs. Reading them
    would inflate prefill by the whole verification cost and drop decode from the
    total -- which would look entirely plausible in the results.
    """
    cfg = _cpu_config()
    result = LlamaCppRunner.parse_output(stdout=llamacpp_output, stderr="", config=cfg)
    assert result.ttft_ms == pytest.approx(669.0)
    assert result.extra["decode_ms"] == pytest.approx(4644.0)
    # The counters this must NOT have used, present in the same sample.
    assert "eval time =       0.00 ms" in llamacpp_output
    assert result.ttft_ms < 1000.0


def test_histogram_matches_the_sampled_steps(llamacpp_output):
    cfg = _cpu_config()
    result = LlamaCppRunner.parse_output(stdout=llamacpp_output, stderr="", config=cfg)
    hist = result.accept_length_histogram
    # The sample has 20 steps: four 0s, one 1, four 2s, no 3s, eleven 4s.
    assert hist == [4, 1, 4, 0, 11]
    assert sum(hist) == 20
    assert sum(k * n for k, n in enumerate(hist)) == 53


def test_parsed_sample_builds_a_schema_valid_record(llamacpp_output):
    cfg = _cpu_config()
    result = LlamaCppRunner.parse_output(stdout=llamacpp_output, stderr="", config=cfg)
    rec = build_record(cfg=cfg, env=ENV, prompt_id="gsm8k-test-1", repeat_idx=0,
                       is_warmup=False, result=result)
    validate_record(rec)
    assert rec["acceptance_rate"] == pytest.approx(53 / 76)
    assert rec["mean_accept_length"] == pytest.approx(53 / 20 + 1.0)


def test_gamma_mismatch_is_refused(llamacpp_output):
    """A draft-length flag the binary ignored must not be recorded as honoured."""
    cfg = _cpu_config(num_speculative_tokens=7, max_tokens=73)
    with pytest.raises(RunnerError, match="n_draft"):
        LlamaCppRunner.parse_output(stdout=llamacpp_output, stderr="", config=cfg)


def test_missing_perf_counters_raise():
    """The baseline arm reads llama.cpp's counters, and refuses without them."""
    cfg = _cpu_config(spec_method="none", num_speculative_tokens=None, draft_model=None)
    with pytest.raises(RunnerError, match="prompt eval time"):
        LlamaCppRunner.parse_output(stdout="nothing useful here", stderr="", config=cfg)


def test_missing_encode_decode_clocks_raise():
    """A speculative run without the example's own clocks has no usable timing.

    The perf counters below are exactly what such a run prints, and are exactly what
    must not be substituted for the missing measurement.
    """
    blob = (
        "common_perf_print: prompt eval time =   1784.96 ms /    79 tokens\n"
        "common_perf_print:        eval time =      0.00 ms /     1 runs\n"
        "accepted 4 draft tokens\nn_draft   = 4\nn_drafted = 4\nn_accept  = 4\n"
    )
    with pytest.raises(RunnerError, match="decoded"):
        LlamaCppRunner.parse_output(stdout=blob, stderr="", config=_cpu_config())


def test_step_counts_must_describe_the_timed_run():
    """The steps and the decode clock must be talking about the same generation."""
    blob = (
        "encoded   10 tokens in    0.100 seconds, speed:  100.000 t/s\n"
        "decoded   40 tokens in    1.000 seconds, speed:   40.000 t/s\n"
        "accepted 4 draft tokens\naccepted 4 draft tokens\n"
        "n_draft   = 4\nn_drafted = 8\nn_accept  = 8\n"
    )
    with pytest.raises(RunnerError, match="decoded 40"):
        LlamaCppRunner.parse_output(stdout=blob, stderr="", config=_cpu_config())


def test_aggregate_only_output_is_refused():
    """Totals without per-step counts cannot make a distribution, so nothing is invented."""
    blob = (
        "encoded   10 tokens in    0.100 seconds, speed:  100.000 t/s\n"
        "decoded   73 tokens in    1.000 seconds, speed:   73.000 t/s\n"
        "n_draft   = 4\nn_drafted = 80\nn_accept  = 53\n"
    )
    with pytest.raises(RunnerError, match="per-step"):
        LlamaCppRunner.parse_output(stdout=blob, stderr="", config=_cpu_config())


def test_per_step_counts_must_agree_with_the_reported_total():
    blob = (
        "encoded   10 tokens in    0.100 seconds, speed:  100.000 t/s\n"
        "decoded    8 tokens in    1.000 seconds, speed:    8.000 t/s\n"
        "accepted 4 draft tokens\naccepted 2 draft tokens\n"
        "n_draft   = 4\nn_drafted = 8\nn_accept  = 53\n"
    )
    with pytest.raises(RunnerError, match="disagree"):
        LlamaCppRunner.parse_output(stdout=blob, stderr="", config=_cpu_config())


def test_cpu_runner_requires_pinned_threads():
    cfg = _cpu_config(num_threads=None)
    # Any real file stands in for the binary; setup() checks thread pinning before it
    # ever tries to execute it.
    runner = LlamaCppRunner(cfg, binary=__file__, model_path=__file__,
                            draft_model_path=__file__)
    with pytest.raises(RunnerError, match="num_threads"):
        runner.setup()


def test_cpu_runner_refuses_batch_conditions():
    runner = LlamaCppRunner(_cpu_config(batch_size=4), binary="x", model_path=__file__)
    runner._flags = {"model": "-m"}
    with pytest.raises(RunnerError, match="batch"):
        runner.generate("prompt")


# -- record assembly -------------------------------------------------------------------


def _result(**overrides) -> GenResult:
    fields = {
        "ttft_ms": 40.0, "total_ms": 40.0 + 20.0 * 63, "output_tokens": 64,
        "output_text": "the answer is 72", "prompt_tokens": 80,
        "accept_length_histogram": [], "draft_tokens_proposed": None,
    }
    fields.update(overrides)
    return GenResult(**fields)


def _gpu_config(**overrides) -> RunConfig:
    fields = {"target_model": "org/m", "target_dtype": "bf16", "stack": "vllm",
              "platform": "h100", "max_tokens": 64}
    fields.update(overrides)
    return RunConfig(**fields)


def test_build_record_refuses_a_short_generation():
    cfg = _gpu_config()
    with pytest.raises(RunnerError, match="ignore_eos"):
        build_record(cfg=cfg, env=ENV, prompt_id="p1", repeat_idx=0, is_warmup=False,
                     result=_result(output_tokens=50))


def test_build_record_derives_tpot_excluding_prefill():
    cfg = _gpu_config()
    rec = build_record(cfg=cfg, env=ENV, prompt_id="p1", repeat_idx=0, is_warmup=False,
                       result=_result())
    assert rec["tpot_ms"] == pytest.approx(20.0)
    validate_record(rec)


def test_build_record_refuses_spec_stats_on_a_baseline_condition():
    """Stats appearing where none were requested means the engine ignored the config."""
    cfg = _gpu_config(spec_method="none")
    with pytest.raises(RunnerError, match="non-speculative"):
        build_record(cfg=cfg, env=ENV, prompt_id="p1", repeat_idx=0, is_warmup=False,
                     result=_result(accept_length_histogram=[1, 2, 3]))


def test_build_record_refuses_an_empty_histogram_when_speculating():
    cfg = _gpu_config(spec_method="draft_model", num_speculative_tokens=4,
                      draft_model="org/d")
    with pytest.raises(RunnerError, match="empty accept_length_histogram"):
        build_record(cfg=cfg, env=ENV, prompt_id="p1", repeat_idx=0, is_warmup=False,
                     result=_result())


def test_hf_records_are_marked_latency_invalid():
    cfg = _gpu_config(stack="hf")
    rec = build_record(cfg=cfg, env=ENV, prompt_id="p1", repeat_idx=0, is_warmup=False,
                       result=_result(), latency_valid=False)
    assert rec["latency_valid"] is False
    validate_record(rec)


# -- vLLM acceptance conversion --------------------------------------------------------


def test_per_position_counts_become_a_run_length_histogram():
    """per_pos[i] counts steps accepting position i; runs are the differences."""
    # 10 steps accepted >=1, 6 accepted >=2, 3 accepted >=3, 1 accepted 4.
    per_pos = [10, 6, 3, 1]
    accepted = sum(per_pos)          # 20
    output_tokens = 32               # so steps = 32 - 20 = 12
    hist, proposed = VLLMRunner._hist_from_per_pos(per_pos, 4, output_tokens)
    assert hist == [2, 4, 3, 2, 1]
    assert sum(hist) == 12
    assert sum(k * n for k, n in enumerate(hist)) == accepted
    assert proposed == 4 * 12


def test_zero_acceptance_bin_is_recovered_not_left_empty():
    """Dropping the zero bin would silently delete the worst-performing steps."""
    per_pos = [8, 4]
    hist, _ = VLLMRunner._hist_from_per_pos(per_pos, 2, output_tokens=20)
    steps = 20 - 12
    assert sum(hist) == steps
    assert hist[0] == steps - 8 == 0 or hist[0] >= 0
    # mean accepted length must reflect every step, including the empty ones
    mean = sum(k * n for k, n in enumerate(hist)) / sum(hist)
    assert mean == pytest.approx(12 / steps)


def test_non_monotonic_counts_are_refused():
    """Acceptance is prefix-closed; violating that breaks the conversion."""
    with pytest.raises(RunnerError, match="non-increasing"):
        VLLMRunner._hist_from_per_pos([4, 9], 2, output_tokens=20)


def test_counts_inconsistent_with_output_length_are_refused():
    with pytest.raises(RunnerError, match="inconsistent"):
        VLLMRunner._hist_from_per_pos([10, 6], 2, output_tokens=8)


def test_vllm_runner_rejects_heuristic_gamma():
    cfg = _gpu_config(spec_method="draft_model", num_speculative_tokens=4,
                      draft_model="org/d", gamma_schedule="heuristic")
    runner = VLLMRunner(cfg)
    with pytest.raises(RunnerError, match="heuristic"):
        runner._speculative_config()


def test_vllm_speculative_config_uses_the_structured_form():
    cfg = _gpu_config(spec_method="eagle3", num_speculative_tokens=3, draft_model="org/head")
    spec = VLLMRunner(cfg)._speculative_config()
    assert spec["method"] == "eagle3"
    assert spec["num_speculative_tokens"] == 3


# -- precision resolution guards -------------------------------------------------------
#
# The guard that stops a BF16 run being recorded as FP8. Quantization is an experimental
# axis in this study (E1, E7), so a condition mislabelled by precision does not produce a
# slightly-off number -- it produces a headline finding about composition that is simply
# false. These exercise _verify_resolution against stub engine configs, since a real vLLM
# is not available here.


class _StubModelConfig:
    def __init__(self, model: str, quantization: str | None, dtype: str) -> None:
        self.model = model
        self.quantization = quantization
        self.dtype = dtype


class _StubVllmConfig:
    def __init__(self, model_config, speculative_config=None) -> None:
        self.model_config = model_config
        self.speculative_config = speculative_config


def _runner_with(model_cfg, spec_cfg=None, **config_overrides):
    """A VLLMRunner whose engine-config read is stubbed out."""
    cfg = _gpu_config(**config_overrides)
    runner = VLLMRunner(cfg)
    runner._engine_config = lambda: _StubVllmConfig(model_cfg, spec_cfg)  # noqa: SLF001
    return runner


def test_bf16_claim_against_a_quantized_checkpoint_raises():
    runner = _runner_with(
        _StubModelConfig("org/m", "awq_marlin", "torch.bfloat16"), target_dtype="bf16"
    )
    with pytest.raises(RunnerError, match="bf16 but the checkpoint resolved"):
        runner._verify_resolution(requested_spec=None)


def test_bf16_claim_against_a_non_bf16_dtype_raises():
    runner = _runner_with(_StubModelConfig("org/m", None, "torch.float16"), target_dtype="bf16")
    with pytest.raises(RunnerError, match="engine dtype resolved"):
        runner._verify_resolution(requested_spec=None)


def test_fp8_claim_against_an_unquantized_checkpoint_raises():
    """The costliest possible mislabel: a BF16 run entering the table as FP8."""
    runner = _runner_with(_StubModelConfig("org/m", None, "torch.bfloat16"), target_dtype="fp8")
    with pytest.raises(RunnerError, match="fp8 but resolved quantization is None"):
        runner._verify_resolution(requested_spec=None)


def test_w4a16_claim_against_a_gguf_checkpoint_raises():
    """GGUF k-quants are not W4A16.

    Q4_K_M is a llama.cpp mixed-precision block format built for CPU; W4A16 here means
    4-bit weights with 16-bit activations through Marlin/AWQ/GPTQ kernels. They differ in
    both quality and speed characteristics, so recording one as the other would put two
    different quantization schemes in the same column of the results table.
    """
    runner = _runner_with(_StubModelConfig("org/m", "gguf", "torch.bfloat16"),
                          target_dtype="w4a16")
    with pytest.raises(RunnerError, match="w4a16 but resolved quantization is 'gguf'"):
        runner._verify_resolution(requested_spec=None)


@pytest.mark.parametrize("method", ["awq", "awq_marlin", "gptq", "gptq_marlin",
                                    "compressed-tensors"])
def test_recognized_w4a16_methods_pass(method):
    runner = _runner_with(_StubModelConfig("org/m", method, "torch.bfloat16"),
                          target_dtype="w4a16")
    runner._verify_resolution(requested_spec=None)
    assert runner.resolved["quantization"] == method


@pytest.mark.parametrize("method", ["fp8", "compressed-tensors", "modelopt"])
def test_recognized_fp8_methods_pass(method):
    runner = _runner_with(_StubModelConfig("org/m", method, "torch.bfloat16"),
                          target_dtype="fp8")
    runner._verify_resolution(requested_spec=None)


def test_silent_model_substitution_raises():
    """A 7B config that quietly loaded something else after an OOM."""
    runner = _runner_with(_StubModelConfig("org/other-1b", None, "torch.bfloat16"),
                          target_dtype="bf16")
    with pytest.raises(RunnerError, match="does not match requested"):
        runner._verify_resolution(requested_spec=None)


def test_speculation_silently_disabled_raises():
    runner = _runner_with(
        _StubModelConfig("org/m", None, "torch.bfloat16"), spec_cfg=None,
        target_dtype="bf16", spec_method="draft_model", num_speculative_tokens=4,
        draft_model="org/d",
    )
    with pytest.raises(RunnerError, match="resolved none"):
        runner._verify_resolution(requested_spec={"num_speculative_tokens": 4})


def test_flashinfer_sampler_is_disabled_by_default():
    """FlashInfer JIT-compiles its sampler and needs nvcc; a driver-only box has none.

    Pinning it off also stops the sampler silently differing between instances depending
    on whether a CUDA toolkit happens to be installed.
    """
    import inspect

    source = inspect.getsource(VLLMRunner.setup)
    assert 'VLLM_USE_FLASHINFER_SAMPLER' in source
    # setdefault, not assignment: an operator with nvcc may opt back in.
    assert 'os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")' in source


def test_engine_env_vars_are_captured_as_provenance():
    """Which kernels ran is provenance, so the engine env goes in the record."""
    import inspect

    from core import env

    source = inspect.getsource(env.capture_env)
    assert "engine_env" in source
    assert "VLLM_" in source
