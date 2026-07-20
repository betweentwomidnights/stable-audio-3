#!/bin/bash
# SA3 unattended pipeline — SUCCESSION dataset variant (forked from sa3_train_pipeline_keygen.sh).
# Same koan/kev/keygen recipe: iterative neutral pre-encode -> sanity gate -> 4k train.
# Dataset: 92 clips (.wav symlinks -> /home/kev/ace/training/succession,
# captioned via ace-step-training canonical sidecars + prep_succession_dataset.py).
# Logs:
#   /tmp/preencode_succession.log   full pre-encode output (+ ACHIEVED RMS summary)
#   /tmp/sa3_train_succession.log   full training output
#   /tmp/sa3_pipeline_succession.log high-level stage markers + gate decision
set -uo pipefail

PLOG=/tmp/sa3_pipeline_succession.log
: > "$PLOG"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$PLOG"; }

TARGET=0.90
DATA=/home/kev/sa3_training_succession         # 92 clips; .wav are symlinks -> /home/kev/ace/training/succession
STAGING=/home/kev/ace/training/succession      # symlink targets, must be bind-mounted
LAT=/home/kev/sa3_training_succession_latents_neutral
SAVE=/home/kev/sa3_lora_out_succession_neutral
DOCKER_COMMON=(--rm --gpus all --shm-size=8g --ipc=host
  -e PYTHONPATH=/workspace/sa3 -e HF_HOME=/cache/huggingface
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e WANDB_MODE=offline
  -w /workspace/sa3
  -v /home/kev/stable-audio-3:/workspace/sa3
  -v /home/kev/.cache/huggingface:/cache/huggingface)

say "STAGE 1/3 — iterative neutral pre-encode (SUCCESSION, target latent RMS=$TARGET)"
N_EXIST=$(ls "$LAT"/*.npy 2>/dev/null | wc -l)
if [ "$N_EXIST" -eq 93 ] && grep -q "ACHIEVED  RMS" /tmp/preencode_succession.log 2>/dev/null; then
  say "STAGE 1 — reusing $N_EXIST existing validated latents in $LAT (skip re-encode)"
else
  docker rm -f sa3-preencode-succession >/dev/null 2>&1
  docker run "${DOCKER_COMMON[@]}" \
    -v "$DATA":"$DATA" \
    -v "$STAGING":"$STAGING" \
    -v "$LAT":"$LAT" \
    --name sa3-preencode-succession sa3:spark-lightning \
    /opt/sa3-venv/bin/python scripts/pre_encode_dataset.py \
      --model same-l \
      --data_dir "$DATA" \
      --output_path "$LAT" \
      --per_track_target_latent_rms "$TARGET" \
      --norm_iters 4 --norm_tol 0.03 > /tmp/preencode_succession.log 2>&1
  PE_RC=$?
  if [ $PE_RC -ne 0 ]; then
    say "ABORT: pre-encode exited $PE_RC. Last lines:"
    tail -15 /tmp/preencode_succession.log | tee -a "$PLOG"
    exit 1
  fi
fi
grep -A6 "per-track norm\]" /tmp/preencode_succession.log | tee -a "$PLOG"

say "STAGE 2/3 — sanity gate"
GATE=$(python3 - "$LAT" "$TARGET" <<'PY'
import sys, glob, re
latdir, target = sys.argv[1], float(sys.argv[2])
log = open("/tmp/preencode_succession.log").read()
m = re.search(r"ACHIEVED\s+RMS\s+min=([\d.]+)\s+mean=([\d.]+)\s+median=([\d.]+)\s+max=([\d.]+)\s+std=([\d.]+)", log)
n_npy = len(glob.glob(f"{latdir}/*.npy"))  # 92 clips + silence.npy = 93
if not m:
    print("FAIL no ACHIEVED summary in log"); sys.exit()
mean, std = float(m.group(2)), float(m.group(5))
rel = abs(mean - target) / target
reasons = []
if n_npy != 93: reasons.append(f"npy count {n_npy} != 93")
if rel > 0.05:  reasons.append(f"mean {mean:.4f} off target {target} by {rel*100:.1f}% (>5%)")
if std > 0.12:  reasons.append(f"std {std:.4f} > 0.12 (spread too wide)")
if reasons: print("FAIL " + "; ".join(reasons))
else:       print(f"PASS mean={mean:.4f} std={std:.4f} npy={n_npy}")
PY
)
say "GATE: $GATE"
case "$GATE" in
  PASS*) ;;
  *) say "ABORT: latents not loudness-neutral — NOT training. Inspect /tmp/preencode_succession.log"; exit 1 ;;
esac

say "STAGE 3/3 — LoRA train, upstream defaults + --steps 4000 (only --num_workers 0 infra)"
mkdir -p "$SAVE"
docker rm -f sa3-train-succession >/dev/null 2>&1
docker run "${DOCKER_COMMON[@]}" \
  -v "$LAT":"$LAT" \
  -v "$SAVE":"$SAVE" \
  --name sa3-train-succession sa3:spark-lightning \
  /opt/sa3-venv/bin/python scripts/train_lora.py \
    --model medium-base \
    --encoded_dir "$LAT" \
    --save_dir "$SAVE" \
    --steps 4000 \
    --num_workers 0 \
    --demo_every 999999 > /tmp/sa3_train_succession.log 2>&1
TR_RC=$?
if [ $TR_RC -ne 0 ]; then
  say "TRAINING FAILED (exit $TR_RC). Last lines:"
  tail -20 /tmp/sa3_train_succession.log | tee -a "$PLOG"
  exit 1
fi
CKPTS=$(find "$SAVE" -name '*.ckpt' 2>/dev/null | wc -l)
say "DONE — training complete. $CKPTS checkpoints in $SAVE"
grep -iE "train/loss|epoch .* step" /tmp/sa3_train_succession.log | tail -3 | tee -a "$PLOG"
echo "PIPELINE_OK"
