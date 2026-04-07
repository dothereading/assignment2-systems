#!/bin/bash
# Generate readable summaries from all nsys profile results.
# Usage: bash cs336_systems/nsys_summarize.sh
#
# Produces a summary text file for each .nsys-rep in results/

set -e

OUTDIR="results"
SUMMARY="$OUTDIR/nsys_summary.txt"

> "$SUMMARY"  # Clear summary file

for rep in "$OUTDIR"/*.nsys-rep; do
  [ -f "$rep" ] || continue
  name=$(basename "$rep" .nsys-rep)

  echo "Processing: $name"

  {
    echo "========================================================================"
    echo "PROFILE: $name"
    echo "========================================================================"

    echo ""
    echo "--- NVTX Range Summary (CPU-side timing) ---"
    nsys stats --force-export=true --report nvtx_sum "$rep" 2>/dev/null || echo "(no NVTX data)"

    echo ""
    echo "--- CUDA GPU Kernel Summary (actual GPU time) ---"
    nsys stats --force-export=true --report cuda_gpu_kern_sum "$rep" 2>/dev/null || echo "(no GPU kernel data)"

    echo ""
    echo "--- NVTX Kernel Summary (GPU kernels per NVTX range, for question e) ---"
    nsys stats --force-export=true --report nvtx_kern_sum "$rep" 2>/dev/null || echo "(no NVTX kernel data)"

    echo ""
    echo "--- CUDA API Summary ---"
    nsys stats --force-export=true --report cuda_api_sum "$rep" 2>/dev/null || echo "(no CUDA API data)"

    echo ""
    echo ""
  } >> "$SUMMARY"
done

echo ""
echo "All summaries written to: $SUMMARY"
echo "View with: less $SUMMARY"
