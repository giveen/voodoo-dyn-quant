# Skill: train

Train a Voodoo mixed-precision assignment for a model. Output: a baked
checkpoint (`<slug>-Voodoo{NN}.pt`), a `quant_assignments.json` sidecar, and a
`partial.pt` journal.

## 0. Preconditions

- `voodoo doctor` healthy (GPU visible, libggml found).
- Calibration data exists (`data/<model>/train_tokens.pt`) — else run the
  [data skill](../data/SKILL.md) first.
- Check GPU headroom before launching (`voodoo doctor`, `rocm-smi`/`nvidia-smi`).
- Pick the size: `Voodoo{NN}` = NN% of the 8-bit original
  (`--compression_ratio 0.NN`). Standard sizes: 25–80 step 5.

## 1. Small/medium model, single GPU

```bash
.venv/bin/voodoo train \
    --model Qwen/Qwen3.5-0.8B-Base \
    --compression_ratio 0.45 \
    --ptqr \
    --max_steps 50 --lr 0.25 --distill_weight 2.0 --grad_accum_steps 1 \
    --size_weight 100.0 --size_tolerance 0.02 \
    --seq_len 512 --batch_size 1 \
    --data_dir data/qwen35-0.8b --data_name train_tokens.pt \
    --output_dir checkpoints/Qwen3.5-0.8B/Voodoo45 \
    --output_name Qwen3.5-0.8B-Voodoo45.pt \
    --device cuda --dtype bfloat16 --no_compile
```

Defaults that just work for 0.5B–8B (from the 0.8B PTQR iteration campaign):
`--ptqr` routing, non-lazy candidates, batch 1, 50 steps, lr 0.25,
distill_weight 2.0, grad_accum 1. First run quantizes the candidate cache
(slow init is normal — watch the progress line); later sizes start in seconds.

Notes behind those defaults:

- **PTQR > soft mixture > ST-Gumbel.** Per-token routing was the first config
  to beat the soft-mixture baseline at a matched budget and its forward
  converges to the deployed model as tau anneals. ST-Gumbel hard-fraction
  mode is rejected (the hard/soft chimera never converged across 4 configs).
- **lr is the dominant lever**: 0.25 (stable) and 1.0 (fast) both land well;
  0.5 is a measured local worst. **50 steps is enough** — 100 gave no benefit.
- **grad_accum 1**: accum=4 lowered closing train KL yet worsened eval PPL.
- Judge iterations by torch KLD vs the BF16 teacher, not PPL and not closing
  train KL (see the eval skill).

## 2. Large model, tensor parallel

Prerequisites for large models: a full-weights base `.pt` (and optionally a
Q8 teacher), created once from any HF checkpoint directory:

```bash
voodoo make-teacher --model_dir ./<model-dir> \
    --strip_prefix model.language_model. \   # only for multimodal wrappers
    --output_dir checkpoints/<family>
```

Then launch with `voodoo tp` (applies hardware env defaults, then torchrun):

```bash
.venv/bin/voodoo tp --nproc 4 -- \
    --model <model-or-local-config-dir> \
    --base_checkpoint <full-weights.pt> \
    --compression_ratio 0.30 --budget_reduction 0.03 \
    --tensor_upgrades '[{"pattern":"mlp\\.(gate|up|down)_proj$","levels":1},{"pattern":"linear_attn|self_attn","levels":-1}]' \
    --seq_len 8192 --batch_size 1 --grad_accum_steps 1 \
    --max_steps 50 --lr 0.5 --size_weight 100.0 --size_tolerance 0.02 \
    --lazy --imatrix <imatrix.gguf> \
    --candidate_types Q8_0 Q5_K IQ3_S IQ2_S IQ2_XXS IQ1_S \
    --attention_candidates Q5_K IQ3_S IQ2_S IQ2_XXS \
    --non_attention_candidates IQ4_XS IQ3_S IQ2_S IQ2_XXS IQ1_S \
    --tensor_parallel 4 \
    --gradient_checkpointing \
    --data_dir data/qwen38-longctx --data_name train_tokens.pt \
    --output_dir checkpoints/Qwen3.8-27B/Voodoo30 \
    --output_name Qwen3.8-27B-Voodoo30.pt \
    --device cuda --dtype bfloat16 \
    --partial_save_interval 1 --no_compile
```

### Evidence-based defaults (from the divergence treatise)

- **Per-role menus with fine rungs both ways.** Attention and MLP get separate
  candidate lists. A missing mid rung makes "buy up" a +1.3 bpw cliff and the
  attention family hoards the MLP's budget — the single biggest measured
  failure mode.
- **Post-hoc guardrails**: `--tensor_upgrades` with `levels: -1` (down) is
  supported; `--budget_reduction 0.03` funds the MLP +1 rung without busting
  the target. No `--force_quant` pins (measured dead weight).
- **Sensitivity warm-start** (`--warm_start`): seed gates from a measured
  layout (real-activation per-tensor sensitivity → greedy knapsack) instead
  of zeros; the table persists to `sensitivity.pkl`. Recommended for large
  models where the cold-start mis-allocation failure is the dominant risk.
- **Knapsack polish** (`--polish`): after argmax, budget-aware one-rung swaps
  ranked by the measured table. Requires `--warm_start`'s table (persists to
  `sensitivity.pkl`); skips itself otherwise.
- **PTQR routing** (`--ptqr`): each token runs through ONE candidate,
  Gumbel-sampled per token from the gate probs (shares match probs in
  expectation; converges to the argmax assignment as tau anneals). Backward
  keeps the exact soft mixture gradient (straight-through). This is the
  recommended routing mode for new runs — it beat the soft-mixture baseline
  at matched budget and trains the model you actually deploy.
- **ST-Gumbel hardening** (`--st_gumbel_fraction`, e.g. 0.5): each layer
  forwards a one-hot Gumbel sample of its gates with that per-step probability
  while gradients stay soft/exact; `--st_gumbel_tau` tunes sample sharpness,
  `--st_gumbel_anneal START:END` ramps the fraction across training.
  Mutually exclusive with `--ptqr`, and REJECTED as a routing mode by the
  0.8B campaign (never converged) — kept for ablation only. Runs predating
  the Gumbel clamp-precedence fix (2026-08-20) silently forwarded candidate 0
  and are invalid.
- **Long context**: train at the seq_len you serve (8192); the quality gap of
  a bad allocation widens with context (5.3% → 9.2% from 512 → 8k).
- **Keep your own imatrix** (`--imatrix`); calibration volume is worth only
  ~0.2% PPL — assignment dominates ~15×.

## 3. Large model, single-GPU (no tensor parallel)

If the full bf16 model exceeds VRAM (e.g. Qwen3.8-27B ≈ 51 GB on a 32 GB
card), the trainer's default `source_model.to(device)` OOMs before
MixedQuant replacement. Fix: the trainer now defers `.to(device)` until
after replacement, so only lightweight gates/nodes materialize on the GPU.

Adapting a 4×GPU TP script to single-GPU:
- Drop `torchrun --nproc_per_node=N --master_port=P`.
- Drop `--tensor_parallel N`.
- Change `--device cuda:0` → `--device cuda`.
- Keep `--lazy` (mmap loading essential for 27B on one card).
- Keep `--gradient_checkpointing` (saves ~50% activation memory).
- Keep `--no_compile` (Triton kernel safety).
- If `--imatrix` points to a missing GGUF (no BF16 reference available),
  drop the flag entirely — assignment dominates ~15× over calibration.

Example adapted script (`configs/qwen38_27b_voodoo30_gpu1.sh`):

```bash
exec .venv/bin/python -m voodoo_quant.cli train \
    --model /mnt/storage/models/swift-qwen3.8-27b \
    --base_checkpoint checkpoints/Qwen3.8-27B/swift-qwen3.8-27b_base.pt \
    --text_model_class qwen3_5_text \
    --teacher_quant Q8_0 \
    --teacher_checkpoint checkpoints/Qwen3.8-27B/swift-qwen3.8-27b_teacher_q8_0.pt \
    --compression_ratio 0.30 \
    --budget_reduction 0.03 \
    --tensor_upgrades '[{\"pattern\":\"mlp\\.(gate|up|down)_proj$\",\"levels\":1},{\"pattern\":\"linear_attn|self_attn\",\"levels\":-1}]' \
    --seq_len 8192 --batch_size 1 --grad_accum_steps 1 \
    --max_steps 50 --lr 0.5 --size_weight 100.0 --size_tolerance 0.02 \
    --lazy \
    --candidate_types Q8_0 Q5_K IQ3_S IQ2_S IQ2_XXS IQ1_S \
    --attention_candidates Q5_K IQ3_S IQ2_S IQ2_XXS \
    --non_attention_candidates IQ4_XS IQ3_S IQ2_S IQ2_XXS IQ1_S \
    --gradient_checkpointing \
    --data_dir data/qwen38-longctx --data_name train_tokens.pt \
    --output_dir checkpoints/Qwen3.8-27B/Voodoo30 \
    --output_name Qwen3.8-27B-Voodoo30.pt \
    --log_file logs/qwen38_voodoo30_train.jsonl \
    --device cuda --dtype bfloat16 \
    --partial_save_interval 1 \
    --no_compile
```

### Pitfalls

- **make-teacher on multimodal checkpoints**: `voodoo make-teacher`
  requires `--strip_prefix model.language_model.` for vision-language
  wrappers. Without it, the vision MLP (`linear_fc1`, `linear_fc2`)
  fails Q8_0 quantization (its `in_features=4304` is not divisible by
  32). The skip list now includes `visual`/`merger` — vision tensors
  stay BF16 in the teacher, which is correct for a language-model-only
  Voodoo run. Always verify the teacher produced a `.pt` file and log
  shows `Done.` with file sizes.
- **imatrix optional**: if no BF16 GGUF exists to generate one from,
  drop `--imatrix`. The train skill notes assignment dominates ~15×.
- **Q8 teacher on CPU**: `--teacher_device cpu` materializes the Q8
  teacher on CPU, halving VRAM pressure at the cost of slower per-step
  reload. Useful on borderline cards (24 GB).

## 4. Launch patterns

- **Layer-wise sharding** (no torchrun, heterogeneous GPUs):
  add `--device_map cuda:0,cuda:1,...` instead of TP.
- **Q8 teacher for huge models**: `--teacher_quant Q8_0
  --teacher_checkpoint <q8.pt>` halves teacher VRAM.
- **Auto-resume**: TP launchers resume from `output_dir/partial.pt` when
  present (see configs/qwen38_27b_voodoo30_tp4.sh for the reference script).

## 5. Monitoring / recovery

- JSONL training log: `--log_file` (loss, tau, size_mb per step).
- Journal: `partial.pt` every `--partial_save_interval` optimizer steps.
- Crash during bake → rerun same command with
  `--finalize_from_partial <outdir>/partial.pt`.
- Interrupted training → `--resume_from_partial <outdir>/partial.pt`.
- Do not interrupt the init/candidate-quantization phase (it is quantizing,
  not hung). Kill `-USR1 <pid>` dumps a heap census to the log.

## 6. Verify before declaring done

- Final log line shows total bytes within `size_tolerance` of target.
- `quant_assignments.json` exists and covers every targeted tensor.
- For release: export (next skill) and sanity-check the GGUF size within a
  few % of target.

## Supported architectures (adapters)

Adapters live in `voodoo_quant/arch/` — qwen3.5/3.6/3.8 hybrids and
LFM2/LFM2.5 (LiquidAI shortconv+GQA hybrids). Anything `transformers` loads
trains single-GPU/layer-wise without an adapter; TP requires one (it refuses
rather than shard blindly). Adding one is a single file: subclass
`ArchAdapter`, `@register` it, define `shard_layer` + `gguf_name` +
`attention_segments`, and add mapping/TP-reconstruction tests to
`tests/test_arch.py` (see `test_lfm2_tp_reconstruction` for the pattern —
shard two ranks, prove the math reconstructs the unsharded forward).
