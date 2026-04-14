"""Benchmarking script with NVTX annotations and nsys profiling support.

Usage:
  # Profile forward only
  nsys profile --capture-range=cudaProfilerApi --force-overwrite=true \
    -o results/small_ctx128_forward \
    uv run python cs336_systems/benchmarking_script_cuda.py \
    --model_size small --context_length 128 --mode forward

  # Profile full training step (forward + backward + AdamW)
  nsys profile --capture-range=cudaProfilerApi --force-overwrite=true \
    -o results/small_ctx128_train \
    uv run python cs336_systems/benchmarking_script_cuda.py \
    --model_size small --context_length 128 --mode train

  # Generate stats:
  nsys stats --force-export=true --report cuda_gpu_kern_sum results/small_ctx128_forward.nsys-rep
  nsys stats --force-export=true --report nvtx_sum results/small_ctx128_forward.nsys-rep
"""

import argparse
import timeit

import torch
import torch.cuda.nvtx as nvtx

import cs336_basics.model
from cs336_basics.model import BasicsTransformerLM, annotated_scaled_dot_product_attention
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

# Monkey-patch so attention layers use the NVTX-annotated version
cs336_basics.model.scaled_dot_product_attention = annotated_scaled_dot_product_attention

MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def benchmark(model, optimizer, inputs, targets, mode, device, warmup_steps, num_steps):
    def step():
        with nvtx.range("forward"):
            logits = model(inputs)
        if mode in ("forward+backward", "train"):
            B, S, V = logits.shape
            with nvtx.range("backward"):
                loss = cross_entropy(logits.reshape(B * S, V), targets.reshape(B * S))
                loss.backward()
        if mode == "train" and optimizer is not None:
            with nvtx.range("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad()
        if device == "cuda":
            torch.cuda.synchronize()

    # Warmup (not captured by profiler)
    print(f"  Warmup ({warmup_steps} steps)...")
    for i in range(warmup_steps):
        step()
        print(f"    warmup {i + 1}/{warmup_steps}")

    torch.cuda.cudart().cudaProfilerStart()

    times = []
    print(f"  Profiling ({num_steps} steps)...")
    for i in range(num_steps):
        start = timeit.default_timer()
        step()
        end = timeit.default_timer()
        times.append(end - start)
        print(f"    step {i + 1}/{num_steps} — {end - start:.4f}s")

    torch.cuda.cudart().cudaProfilerStop()

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    return avg, std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--theta", type=int, default=10000)
    parser.add_argument("--model_size", type=str, default=None, choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument(
        "--mode",
        type=str,
        default="forward",
        choices=["forward", "forward+backward", "train"],
        help="forward: inference only. forward+backward: no optimizer. train: full training step with AdamW.",
    )
    args = parser.parse_args()

    if args.model_size:
        config = MODEL_CONFIGS[args.model_size]
        args.d_model = config["d_model"]
        args.d_ff = config["d_ff"]
        args.num_layers = config["num_layers"]
        args.num_heads = config["num_heads"]

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.theta,
    ).to(args.device)

    inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)

    optimizer = None
    if args.mode == "train":
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    avg, std = benchmark(model, optimizer, inputs, targets, args.mode, args.device, args.warmup_steps, args.num_steps)

    size_label = args.model_size or "custom"
    print(f"Model: {size_label} | Mode: {args.mode} | Context: {args.context_length}")
    print(f"Average time per step: {avg:.4f}s | Std dev: {std:.4f}s")


if __name__ == "__main__":
    main()
