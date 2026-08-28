#!/usr/bin/env bash
# After instance-2 finishes, run the two GPU measurements figures 04 and 07 are missing,
# then render every figure instance-1 and instance-2 can support.
#
# These cannot run while a sweep is live: run_sweep asserts GPU exclusivity before each
# condition block, so a second CUDA process would abort the sweep at its next boundary.
# Hence the wait.
#
# Unlike chain_instance2.sh this does NOT gate on instance-2 completing. A partial
# instance-2 still leaves the GPU free and still supports figures 02/04/06/07 (which come
# from instance-1, already complete); only figure 03 needs instance-2's batch axis, and
# render_figures reports a figure it cannot draw as SKIPPED rather than drawing it wrong.
set -u

REPO=/home/exouser/inference-exp
PY=/home/exouser/vlm4engagement/.venv/bin/python
WATCH_PID="${1:?usage: chain_post_instance2.sh <instance2-driver-pid>}"
LOG="$REPO/logs/chain_post_instance2.log"

DRAFT=meta-llama/Llama-3.2-1B-Instruct
BF16=meta-llama/Llama-3.1-8B-Instruct
FP8=RedHatAI/Meta-Llama-3.1-8B-Instruct-FP8
W4A16=RedHatAI/Meta-Llama-3.1-8B-Instruct-quantized.w4a16

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" >> "$LOG"; }

cd "$REPO" || exit 1
log "waiting for instance-2 driver pid=$WATCH_PID to release the GPU"
while kill -0 "$WATCH_PID" 2>/dev/null; do sleep 60; done
log "instance-2 driver exited; waiting for CUDA memory to actually drain"

# The driver exiting does not mean the engine is gone; EngineCore is a child process and
# has been seen outliving it. Starting a model load into a still-occupied GPU would OOM.
for _ in $(seq 1 30); do
    if ! nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q .; then break; fi
    sleep 10
done
BUSY=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
if [ -n "$BUSY" ]; then
    log "ABORT: GPU still occupied after 5 min: $BUSY"
    exit 1
fi
log "GPU clear"

# ---- figure 04's input: the draft/target cost ratio, measured per precision ----------
log "measuring c (isolated batch-1 step times, speculation off)"
if "$PY" -m scripts.measure_c \
        --draft "$DRAFT" \
        --target "$BF16=bf16" --target "$FP8=fp8" --target "$W4A16=w4a16" \
        --out logs/measured_c.json >> "$LOG" 2>&1; then
    log "c written to logs/measured_c.json"
else
    log "WARNING: measure_c failed; figure 04 will be reported SKIPPED, not faked"
fi

# ---- figure 07's input: WikiText-2 perplexity per precision --------------------------
# Identical window settings across all three, or fig07 refuses to take deltas.
for pair in "bf16:$BF16" "fp8:$FP8" "w4a16:$W4A16"; do
    dt="${pair%%:*}"; model="${pair#*:}"
    log "perplexity: $dt ($model)"
    if "$PY" -m scripts.run_perplexity --model "$model" \
            --out "logs/ppl-$dt.json" --max-length 2048 --stride 512 >> "$LOG" 2>&1; then
        log "  ok -> logs/ppl-$dt.json"
    else
        log "  WARNING: perplexity failed for $dt; figure 07 will be SKIPPED"
    fi
done

# ---- build the report ----------------------------------------------------------------
LOGS="logs/instance-1.jsonl"
[ -s logs/instance-2.jsonl ] && LOGS="$LOGS logs/instance-2.jsonl"

PPL_ARGS=""
for dt in bf16 fp8 w4a16; do
    [ -s "logs/ppl-$dt.json" ] && PPL_ARGS="$PPL_ARGS $dt=logs/ppl-$dt.json"
done

C_ARG=""
[ -s logs/measured_c.json ] && C_ARG="--c-by-pair logs/measured_c.json"

log "building report: logs=[$LOGS] ppl=[$PPL_ARGS] c=[$C_ARG]"
# shellcheck disable=SC2086
"$PY" -m scripts.build_report --logs $LOGS --outdir report/full \
    --micro h100_tp1=logs/instance-2-videomme_micro.json \
    ${PPL_ARGS:+--perplexity $PPL_ARGS} \
    $C_ARG >> "$LOG" 2>&1

log "done; see report/full/ and the figure list above"
