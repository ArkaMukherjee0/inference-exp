# llama.cpp build for the local CPU arm (E6)

The prebuilt llama.cpp release archives ship no speculative binary at all -- speculative
decoding now lives inside `llama-cli`, which in b10655 is a chat client that prints
throughput and nothing else: no per-step acceptance, no prefill/decode split. The
`examples/speculative` tool still exists in the source tree, so this arm builds it.

Two patches, both in `llamacpp-b10655-speculative.patch`:

1. **Per-step acceptance logging.** The stock example prints only the aggregate totals
   (`n_drafted`, `n_accept`). Figure 06 is a claim about the accepted-run-length
   *distribution*, and a histogram synthesised from a mean is fabricated data, so the
   runner refuses aggregate-only output. The patch logs the accepted count once per
   verification step; the lines sum to `n_accept` by construction.

2. **Stop at exactly `-n` tokens.** The stock loop finishes the verification step it is
   in and overshoots by up to the draft length -- so every speculative condition would
   emit a few more tokens than the baseline, by an amount that grows with the very axis
   being measured. Under `ignore_eos` the token budget must be identical across
   conditions or the timings are not comparable, and `validate_record` rejects the
   mismatch.

## Build

Requires MSVC and the CMake/Ninja that ship with Visual Studio.

```
git clone --depth 1 --branch b10655 https://github.com/ggml-org/llama.cpp.git C:\tools\llama.cpp-src
cd C:\tools\llama.cpp-src
git apply <this repo>\patches\llamacpp-b10655-speculative.patch

call "C:\Program Files\Microsoft Visual Studio\18\Community\VC\Auxiliary\Build\vcvars64.bat"
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release ^
      -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_CURL=OFF -DGGML_BACKEND_DL=OFF
cmake --build build --target llama-speculative llama-completion -j 24
```

Both binaries come from this one tree on purpose. `llama-speculative` refuses to run
without a draft model, so it cannot produce the gamma=0 baseline; `llama-completion` can
produce the baseline and cannot speculate. Building them separately -- or taking the
baseline from a release archive, which uses runtime CPU-kernel dispatch while this build
compiles a fixed `/arch:AVX2` -- would put the build difference into the speedup.

`configs/local_cpu.yaml` points `model.binary` and `model.baseline_binary` at
`build\bin`. Re-run `python -m scripts.setup_data --check-models configs/local_cpu.yaml`
after rebuilding.
