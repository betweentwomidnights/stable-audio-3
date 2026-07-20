#!/bin/bash
# SA3 unattended pipeline — RATATAT dataset variant (forked from sa3_train_pipeline_keygen.sh).
# Same koan/kev/keygen/succession recipe: iterative neutral pre-encode -> sanity gate -> 4k train.
# Dataset: 10 full-length Ratatat tracks (real .wav, 184-328s, 44.1k stereo) + short
# genre-tag captions already in place as .txt sidecars.
# Logs:
#   /tmp/preencode_ratatat.log    full pre-encode output (+ ACHIEVED RMS summary)
#   /tmp/sa3_train_ratatat.log    full training output
#   /tmp/sa3_pipeline_ratatat.log high-level stage markers + gate decision
set -uo pipefail

PLOG=/tmp/sa3_pipeline_ratatat.log
: > "$PLOG"
say() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$PLOG"; }

TARGET=0.90
DATA=/home/kev/ratatat_classics                 # 10 clips, real .wav (no symlinks -> no STAGING mount)
LAT=/home/kev/sa3_training_ratatat_latents_neutral
SAVE=/home/kev/sa3_lora_out_ratatat_neutral
N_TOTAL=11                                      # 10 clips + silence.npy
DOCKER_COMMON=(--rm --gpus all --shm-size=8g --ipc=host
  -e PYTHONPATH=/workspace/sa3 -e HF_HOME=/cache/huggingface
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e WANDB_MODE=offline
  -w /workspace/sa3
  -v /home/kev/stable-audio-3:/workspace/sa3
  -v /home/kev/.cache/huggingface:/cache/huggingface)

mkdir -p "$LAT" "$SAVE"

say "STAGE 1/3 — iterative neutral pre-encode (RATATAT, target latent RMS=$TARGET)"
N_EXIST=$(ls "$LAT"/*.npy 2>/dev/null | wc -l)
if [ "$N_EXIST" -eq "$N_TOTAL" ] && grep -q "ACHIEVED  RMS" /tmp/preencode_ratatat.log 2>/dev/null; then
  say "STAGE 1 — reusing $N_EXIST existing validated latents in $LAT (skip re-encode)"
else
  docker rm -f sa3-preencode-ratatat >/dev/null 2>&1
  docker run "${DOCKER_COMMON[@]}" \
    -v "$DATA":"$DATA" \
    -v "$LAT":"$LAT" \
    --name sa3-preencode-ratatat sa3:spark-lightning \
    /opt/sa3-venv/bin/python scripts/pre_encode_dataset.py \
      --model same-l \
      --data_dir "$DATA" \
      --output_path "$LAT" \
      --per_track_target_latent_rms "$TARGET" \
      --norm_iters 4 --norm_tol 0.03 > /tmp/preencode_ratatat.log 2>&1
  PE_RC=$?
  if [ $PE_RC -ne 0 ]; then
    say "ABORT: pre-encode exited $PE_RC. Last lines:"
    tail -15 /tmp/preencode_ratatat.log | tee -a "$PLOG"
    exit 1
  fi
fi
grep -A6 "per-track norm\]" /tmp/preencode_ratatat.log | tee -a "$PLOG"

say "STAGE 2/3 — sanity gate"
GATE=$(python3 - "$LAT" "$TARGET" "$N_TOTAL" <<'PY'
import sys, glob, re
latdir, target, n_total = sys.argv[1], float(sys.argv[2]), int(sys.argv[3])
log = open("/tmp/preencode_ratatat.log").read()
m = re.search(r"ACHIEVED\s+RMS\s+min=([\d.]+)\s+mean=([\d.]+)\s+median=([\d.]+)\s+max=([\d.]+)\s+std=([\d.]+)", log)
n_npy = len(glob.glob(f"{latdir}/*.npy"))
if not m:
    print("FAIL no ACHIEVED summary in log"); sys.exit()
mean, std = float(m.group(2)), float(m.group(5))
rel = abs(mean - target) / target
reasons = []
if n_npy != n_total: reasons.append(f"npy count {n_npy} != {n_total}")
if rel > 0.05:  reasons.append(f"mean {mean:.4f} off target {target} by {rel*100:.1f}% (>5%)")
if std > 0.12:  reasons.append(f"std {std:.4f} > 0.12 (spread too wide)")
if reasons: print("FAIL " + "; ".join(reasons))
else:       print(f"PASS mean={mean:.4f} std={std:.4f} npy={n_npy}")
PY
)
say "GATE: $GATE"
case "$GATE" in
  PASS*) ;;
  *) say "ABORT: latents not loudness-neutral — NOT training. Inspect /tmp/preencode_ratatat.log"; exit 1 ;;
esac

say "STAGE 3/3 — LoRA train, upstream defaults + --steps 4000 (only --num_workers 0 infra)"
docker rm -f sa3-train-ratatat >/dev/null 2>&1
docker run "${DOCKER_COMMON[@]}" \
  -v "$LAT":"$LAT" \
  -v "$SAVE":"$SAVE" \
  --name sa3-train-ratatat sa3:spark-lightning \
  /opt/sa3-venv/bin/python scripts/train_lora.py \
    --model medium-base \
    --encoded_dir "$LAT" \
    --save_dir "$SAVE" \
    --steps 2000 \
    --duration 285.35 \
    --num_workers 0 \
    --demo_every 0 > /tmp/sa3_train_ratatat.log 2>&1
TR_RC=$?
if [ $TR_RC -ne 0 ]; then
  say "TRAINING FAILED (exit $TR_RC). Last lines:"
  tail -20 /tmp/sa3_train_ratatat.log | tee -a "$PLOG"
  exit 1
fi
CKPTS=$(find "$SAVE" -name '*.ckpt' 2>/dev/null | wc -l)
say "DONE — training complete. $CKPTS checkpoints in $SAVE"
grep -iE "train/loss|epoch .* step" /tmp/sa3_train_ratatat.log | tail -3 | tee -a "$PLOG"
echo "PIPELINE_OK"
