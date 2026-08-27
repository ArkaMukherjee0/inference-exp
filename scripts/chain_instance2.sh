#!/usr/bin/env bash
# Wait for instance-1 to finish, then start instance-2 in the same screen session.
#
# Started detached (setsid + nohup) so it outlives the terminal and the Claude session
# that created it. It launches instance-2 in a NEW window of screen session "inf" --
# window 0 is busy running instance-1, so keystrokes must not be stuffed into it.
#
# The handoff is conditional. A sweep driver that exits because it was killed, OOMed or
# crashed looks exactly like one that finished, so exiting is not treated as completing:
# the queue is re-checked and instance-2 starts only if instance-1 has zero units left to
# run. Otherwise this logs why and stops, leaving the GPU free for a human to look at.
set -u

REPO=/home/exouser/inference-exp
PY=/home/exouser/vlm4engagement/.venv/bin/python
WATCH_PID="${1:?usage: chain_instance2.sh <instance1-driver-pid>}"
SESSION=inf
LOG="$REPO/logs/chain_instance2.log"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" >> "$LOG"; }

log "watching instance-1 driver pid=$WATCH_PID; will start instance-2 in screen '$SESSION'"

while kill -0 "$WATCH_PID" 2>/dev/null; do sleep 60; done
log "instance-1 driver $WATCH_PID exited; verifying the queue actually drained"

# Let any final append land before reading the log back.
sleep 10

REMAINING=$(cd "$REPO" && "$PY" - <<'EOF' 2>/dev/null
from core.config import load_sweep
from scripts.run_sweep import _resolve, completed_keys, remaining, load_prompts

cfg = load_sweep("configs/instance1_precision_spec.yaml")
queue = cfg.build_queue(load_prompts(cfg).ids)
print(len(remaining(queue, completed_keys(_resolve(cfg.log_path)))))
EOF
)
# load_prompts prints an allow_partial NOTE to stdout; the count is the last line.
REMAINING=$(printf '%s\n' "$REMAINING" | tail -1 | tr -dc '0-9')

if [ -z "$REMAINING" ]; then
    log "ABORT: could not read instance-1's remaining queue; not starting instance-2"
    exit 1
fi

if [ "$REMAINING" -ne 0 ]; then
    log "ABORT: instance-1 exited with $REMAINING units still pending -- it did not"
    log "       complete, so instance-2 is NOT being started. Investigate, then either"
    log "       resume instance-1 or launch instance-2 by hand."
    exit 1
fi

log "instance-1 complete (0 units pending); starting instance-2 in screen '$SESSION'"

if ! screen -S "$SESSION" -Q select . >/dev/null 2>&1; then
    log "ABORT: screen session '$SESSION' is gone; not starting instance-2"
    exit 1
fi

screen -S "$SESSION" -X screen -t instance2 \
    bash -lc "cd $REPO && $PY -u -m scripts.run_sweep --config configs/instance2_batch_drafter.yaml 2>&1 | tee logs/instance-2.console.log; echo; echo '--- instance-2 finished; press enter to close ---'; read"

log "instance-2 launched in a new window of screen '$SESSION' (attach: screen -r $SESSION)"
