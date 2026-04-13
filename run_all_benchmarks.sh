#!/usr/bin/env bash
set -e

ATTN=cs336_systems/attn_benchmarking_script.py
BENCH=cs336_systems/benchmarking_script.py

echo "=== Attention benchmark (vanilla) ==="
uv run python $ATTN

echo "=== Attention benchmark (compiled) ==="
uv run python $ATTN --compiled

for MODE in forward train; do
    echo "=== Transformer sweep ($MODE, vanilla) ==="
    uv run python $BENCH --sweep --mode $MODE

    echo "=== Transformer sweep ($MODE, compiled) ==="
    uv run python $BENCH --sweep --mode $MODE --compiled
done

echo "=== All benchmarks done ==="
