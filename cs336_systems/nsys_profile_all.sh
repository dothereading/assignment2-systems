#!/bin/bash
# Run nsys profiling for all model sizes, context lengths, and modes.
# Usage: bash cs336_systems/nsys_profile_all.sh
#
# Results go into results/ directory as .nsys-rep files.
# After profiling, run: bash cs336_systems/nsys_summarize.sh
# to generate readable summaries.

set +e  # Don't exit on non-zero — nsys can return non-zero even on success

SCRIPT="cs336_systems/benchmarking_script_cuda.py"
OUTDIR="results"
mkdir -p "$OUTDIR"

MODEL_SIZES=("small" "medium" "large" "xl" "2.7B")
CONTEXT_LENGTHS=(128 256 512 1024)
MODES=("forward" "forward+backward" "train")

for model in "${MODEL_SIZES[@]}"; do
  for ctx in "${CONTEXT_LENGTHS[@]}"; do
    for mode in "${MODES[@]}"; do
      # Clean name for output file
      mode_tag="${mode//+/_}"
      outname="${model}_ctx${ctx}_${mode_tag}"

      echo "============================================"
      echo "Profiling: model=$model ctx=$ctx mode=$mode"
      echo "Output: $OUTDIR/$outname.nsys-rep"
      echo "============================================"

      nsys profile \
        --capture-range=cudaProfilerApi \
        --force-overwrite=true \
        -o "$OUTDIR/$outname" \
        uv run python "$SCRIPT" \
          --model_size "$model" \
          --context_length "$ctx" \
          --mode "$mode" \
        2>&1 || echo ">>> FAILED or OOM: model=$model ctx=$ctx mode=$mode"

      echo ""
    done
  done
done

echo "Done! Now run: bash cs336_systems/nsys_summarize.sh"
