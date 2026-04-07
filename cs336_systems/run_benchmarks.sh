#!/bin/bash
# End-to-end benchmarks for questions (b) and (c)
# Usage: bash cs336_systems/run_benchmarks.sh

# Question (b) — forward and backward sweeps
echo "=== Question (b): Forward sweep ==="
uv run python cs336_systems/benchmarking_script.py --sweep --mode forward --device cuda

echo ""
echo "=== Question (b): Forward+backward sweep ==="
uv run python cs336_systems/benchmarking_script.py --sweep --mode forward+backward --device cuda

# Question (c) — varying warmup steps
for w in 0 1 2; do
  echo ""
  echo "=== Question (c): Forward sweep, warmup=$w ==="
  uv run python cs336_systems/benchmarking_script.py --sweep --mode forward --device cuda --warmup_steps "$w"
done

for w in 0 1 2; do
  echo ""
  echo "=== Question (c): Forward+backward sweep, warmup=$w ==="
  uv run python cs336_systems/benchmarking_script.py --sweep --mode forward+backward --device cuda --warmup_steps "$w"
done

# Mixed precision sweeps
echo ""
echo "=== Mixed precision: Forward sweep ==="
uv run python cs336_systems/benchmarking_script.py --sweep --mode forward --device cuda --mixed_precision

echo ""
echo "=== Mixed precision: Forward+backward sweep ==="
uv run python cs336_systems/benchmarking_script.py --sweep --mode forward+backward --device cuda --mixed_precision
