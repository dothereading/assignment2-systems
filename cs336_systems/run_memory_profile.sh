#!/bin/bash
# Memory profiling for 2.7B model
# Usage: bash cs336_systems/run_memory_profile.sh

mkdir -p results

# FP32 — forward and train for each context length
for ctx in 128 256 512; do
  for mode in forward train; do
    echo "=== 2.7B ctx=$ctx mode=$mode fp32 ==="
    uv run python cs336_systems/benchmarking_script.py \
      --model_size 2.7B \
      --context_length "$ctx" \
      --mode "$mode" \
      --memory_profile
  done
done

# BF16 mixed precision (question c) — ctx=256 only
for mode in forward train; do
  echo "=== 2.7B ctx=256 mode=$mode bf16 ==="
  uv run python cs336_systems/benchmarking_script.py \
    --model_size 2.7B \
    --context_length 256 \
    --mode "$mode" \
    --memory_profile \
    --mixed_precision
done

echo "Done! Download .pickle files from results/ and view at pytorch.org/memory_viz"
