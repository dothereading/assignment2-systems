import argparse
import timeit

import pandas as pd
import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy

MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 1600, "d_ff": 6400,  "num_layers": 48, "num_heads": 25},
    "2.7B":   {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def benchmark(model, inputs, targets, mode, device, warmup_steps, num_steps):
    def step():
        logits = model(inputs)
        if mode == "forward+backward":
            B, S, V = logits.shape
            loss = cross_entropy(logits.reshape(B * S, V), targets.reshape(B * S))
            loss.backward()
        if device == "cuda":
            torch.cuda.synchronize()

    for _ in range(warmup_steps):
        step()

    times = []
    for _ in range(num_steps):
        start = timeit.default_timer()
        step()
        end = timeit.default_timer()
        times.append(end - start)

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5
    return avg, std


def main():
    parser = argparse.ArgumentParser(description="Benchmark transformer model forward and backward passes.")

    parser.add_argument("--device", type=str, default="cuda", help='Device: "cpu", "mps" (Mac), or "cuda"')
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--context_length", type=int, default=256, help="Context length of LM.")
    parser.add_argument("--vocab_size", type=int, default=10000, help="Size of the vocabulary.")
    parser.add_argument("--theta", type=int, default=10000, help="Theta value for RoPE.")

    # Model config: either pick a preset or specify manually
    parser.add_argument("--model_size", type=str, default=None,
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Use a preset model config from the assignment.")
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)

    # Benchmarking
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warm-up steps.")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of timed steps.")
    parser.add_argument("--mode", type=str, default="forward+backward",
                        choices=["forward", "forward+backward"],
                        help="Whether to benchmark forward only or forward+backward.")

    # Sweep mode: run all model sizes and produce a table
    parser.add_argument("--sweep", action="store_true",
                        help="Run all model sizes and output a results table.")

    args = parser.parse_args()

    if args.sweep:
        run_sweep(args)
    else:
        run_single(args)


def run_single(args):
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

    avg, std = benchmark(model, inputs, targets, args.mode, args.device, args.warmup_steps, args.num_steps)

    size_label = args.model_size or "custom"
    print(f"Model: {size_label} | Mode: {args.mode} | Context: {args.context_length}")
    print(f"Average time per step: {avg:.4f}s | Std dev: {std:.4f}s")


def run_sweep(args):
    rows = []
    for size_name, config in MODEL_CONFIGS.items():
        print(f"Benchmarking {size_name}...")
        model = BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            d_model=config["d_model"],
            num_layers=config["num_layers"],
            num_heads=config["num_heads"],
            d_ff=config["d_ff"],
            rope_theta=args.theta,
        ).to(args.device)

        inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)
        targets = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)

        avg, std = benchmark(model, inputs, targets, args.mode, args.device, args.warmup_steps, args.num_steps)
        rows.append({"size": size_name, "avg_time_s": avg, "std_time_s": std})

        # Free GPU memory before next model
        del model, inputs, targets
        if args.device == "cuda":
            torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    print(f"\nMode: {args.mode} | Context length: {args.context_length}")
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    main()
