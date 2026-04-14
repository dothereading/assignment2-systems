import argparse
import timeit
from contextlib import nullcontext

import pandas as pd
import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

MODEL_CONFIGS = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 1600, "d_ff": 6400, "num_layers": 48, "num_heads": 25},
    "2.7B": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
}


def benchmark(model, inputs, targets, mode, device, warmup_steps, num_steps, autocast_ctx=None, optimizer=None):
    if autocast_ctx is None:
        autocast_ctx = nullcontext

    def step():
        with autocast_ctx():
            logits = model(inputs)
            if mode in ("forward+backward", "train"):
                B, S, V = logits.shape
                loss = cross_entropy(logits.reshape(B * S, V), targets.reshape(B * S))
        if mode in ("forward+backward", "train"):
            loss.backward()
        if mode == "train" and optimizer is not None:
            optimizer.step()
            optimizer.zero_grad()
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
    parser.add_argument("--compiled", action="store_true")

    # Model config: either pick a preset or specify manually
    parser.add_argument(
        "--model_size",
        type=str,
        default=None,
        choices=list(MODEL_CONFIGS.keys()),
        help="Use a preset model config from the assignment.",
    )
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--num_layers", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)

    # Benchmarking
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warm-up steps.")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of timed steps.")
    parser.add_argument(
        "--mode",
        type=str,
        default="forward+backward",
        choices=["forward", "forward+backward", "train"],
        help="forward: inference only. forward+backward: no optimizer. train: full step with AdamW.",
    )
    parser.add_argument("--mixed_precision", action="store_true", help="Use BF16 mixed precision via torch.autocast.")
    parser.add_argument(
        "--memory_profile", action="store_true", help="Run memory profiler and save snapshot as .pickle file."
    )

    # Sweep mode: run all model sizes and produce a table
    parser.add_argument("--sweep", action="store_true", help="Run all model sizes and output a results table.")

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

    model = torch.compile(model) if args.compiled else model

    inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)

    if args.mixed_precision:
        autocast_ctx = lambda: torch.autocast(device_type=args.device, dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext

    optimizer = None
    if args.mode == "train":
        optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    if args.memory_profile:
        run_memory_profile(model, optimizer, inputs, targets, args)
    else:
        avg, std = benchmark(
            model,
            inputs,
            targets,
            args.mode,
            args.device,
            args.warmup_steps,
            args.num_steps,
            autocast_ctx=autocast_ctx,
            optimizer=optimizer,
        )
        size_label = args.model_size or "custom"
        precision = "bf16" if args.mixed_precision else "fp32"
        print(f"Model: {size_label} | Mode: {args.mode} | Context: {args.context_length} | Precision: {precision}")
        print(f"Average time per step: {avg:.4f}s | Std dev: {std:.4f}s")


def run_memory_profile(model, optimizer, inputs, targets, args):
    if args.mixed_precision:
        autocast_ctx = lambda: torch.autocast(device_type=args.device, dtype=torch.bfloat16)
    else:
        autocast_ctx = nullcontext

    def step():
        with autocast_ctx():
            logits = model(inputs)
            if args.mode in ("forward+backward", "train"):
                B, S, V = logits.shape
                loss = cross_entropy(logits.reshape(B * S, V), targets.reshape(B * S))
        if args.mode in ("forward+backward", "train"):
            loss.backward()
        if args.mode == "train" and optimizer is not None:
            optimizer.step()
            optimizer.zero_grad()
        if args.device == "cuda":
            torch.cuda.synchronize()

    # Warmup (1 step to keep it cheap)
    step()

    # Start memory recording
    torch.cuda.memory._record_memory_history(max_entries=1000000)

    # Run 1 profiled step
    step()

    # Save snapshot
    size_label = args.model_size or "custom"
    precision = "bf16" if args.mixed_precision else "fp32"
    filename = f"results/memory_{size_label}_ctx{args.context_length}_{args.mode}_{precision}.pickle"
    torch.cuda.memory._dump_snapshot(filename)
    torch.cuda.memory._record_memory_history(enabled=None)

    peak_mb = torch.cuda.max_memory_allocated() / (1024**2)
    print(f"Memory snapshot saved to: {filename}")
    print(f"Peak memory allocated: {peak_mb:.1f} MB")


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

        if args.compiled:
            model = torch.compile(model)

        inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)
        targets = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)

        if args.mixed_precision:
            autocast_ctx = lambda: torch.autocast(device_type=args.device, dtype=torch.bfloat16)
        else:
            autocast_ctx = nullcontext
        avg, std = benchmark(
            model, inputs, targets, args.mode, args.device, args.warmup_steps, args.num_steps, autocast_ctx=autocast_ctx
        )
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
