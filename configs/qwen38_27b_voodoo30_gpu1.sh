#!/usr/bin/env bash
# Single-GPU adaptation of qwen38_27b_voodoo30_tp4.sh for Qwen3.8-27B.
#
# Changes from the 4xGPU reference script:
#   - Removed torchrun / --nproc_per_node / --master_port
#   - Removed --tensor_parallel 4 (single GPU)
#   - Changed --device cuda:0 → --device cuda
#   - Dropped --imatrix (no BF16 GGUF to generate it from)
#   - Kept --lazy (mmap loading essential for 27B on one card)
#   - Kept --gradient_checkpointing (saves ~50% activation memory)
#   - Kept --no_compile (Triton kernel safety)
#
# Prerequisites (one-time):
#   1. make bootstrap   (or: python3 -m venv .venv && .venv/bin/pip install -e ".[torch,test]")
#      with: export VOODOO_GGML_LIB=/mnt/storage/llama.cpp/build/bin/libggml-base.so
#   2. Calibration data:
#      voodoo data --model /mnt/storage/models/swift-qwen3.8-27b \
#          --seq_len 8192 --n_samples 10000 --val_ratio 0.05 \
#          --output_dir data/qwen38-longctx
#   3. BF16 base + Q8 teacher:
#      voodoo make-teacher --model_dir /mnt/storage/models/swift-qwen3.8-27b \
#          --strip_prefix model.language_model. \
#          --output_dir checkpoints/Qwen3.8-27B
#   4. Optional imatrix (skill: imatrix / blackbeard tools).

set -euo pipefail
cd "$(dirname "$0")/.."

export VOODOO_GGML_LIB=/mnt/storage/llama.cpp/build/bin/libggml-base.so

PY=.venv/bin/python
OUT=checkpoints/Qwen3.8-27B/Voodoo30
MODEL_DIR=/mnt/storage/models/swift-qwen3.8-27b
BASE_CKPT=checkpoints/Qwen3.8-27B/swift-qwen3.8-27b_base.pt
TEACHER_Q8=checkpoints/Qwen3.8-27B/swift-qwen3.8-27b_teacher_q8_0.pt
DATA=data/qwen38-longctx
mkdir -p "$OUT" logs

# Hardware env defaults
export MALLOC_ARENA_MAX=2
export OMP_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export TRITON_PARALLEL_COMPILE=1
export VOODOO_FINALIZE_WORKERS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Auto-resume from journal if present
RESUME_ARGS=""
if [ -s "$OUT/partial.pt" ] && [ -s "$OUT/partial.meta.json" ]; then
    PREV=$($PY -c "import json;print(json.load(open('$OUT/partial.meta.json'))['completed_optimizer_steps'])" 2>/dev/null || echo 0)
    if [ "$PREV" -gt 0 ] && [ "$PREV" -lt 50 ]; then
        RESUME_ARGS="--resume_from_partial $OUT/partial.pt"
        echo "launcher: resuming from journal at optimizer step $PREV" >> "$OUT/train_gpu1.log"
    fi
fi

# Post-hoc guardrail: MLP one rung UP, attention one rung DOWN
UPGRADES='[
  {"pattern":"mlp\\.(gate|up|down)_proj$","levels":1},
  {"pattern":"linear_attn|self_attn","levels":-1}
]'

exec "$PY" -m voodoo_quant.cli train \
    --model "$MODEL_DIR" \
    --base_checkpoint "$BASE_CKPT" \
    --text_model_class qwen3_5_text \
    --teacher_quant Q8_0 \
    --teacher_checkpoint "$TEACHER_Q8" \
    --compression_ratio 0.30 \
    --budget_reduction 0.03 \
    --tensor_upgrades "$UPGRADES" \
    --seq_len 8192 \
    --batch_size 1 \
    --grad_accum_steps 1 \
    --max_steps 50 \
    --lr 0.5 \
    --size_weight 100.0 \
    --size_tolerance 0.02 \
    --lazy \
    --candidate_types Q8_0 Q5_K IQ3_S IQ2_S IQ2_XXS IQ1_S \
    --attention_candidates Q5_K IQ3_S IQ2_S IQ2_XXS \
    --non_attention_candidates IQ4_XS IQ3_S IQ2_S IQ2_XXS IQ1_S \
    --gradient_checkpointing \
    --data_dir "$DATA" \
    --data_name train_tokens.pt \
    --output_dir "$OUT" \
    --output_name Qwen3.8-27B-Voodoo30.pt \
    --log_file logs/qwen38_voodoo30_train.jsonl \
    --device cuda \
    --dtype bfloat16 \
    --partial_save_interval 1 \
    $RESUME_ARGS \
    --no_compile \
    >> "$OUT/train_gpu1.log" 2>&1
