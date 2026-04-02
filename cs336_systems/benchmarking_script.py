import argparse
import timeit

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy


def main():
    parser = argparse.ArgumentParser(description="Benchmark transformer model forward and backward passes.")

    parser.add_argument("--device", type=str, default="cuda", help='Device: "cpu", "mps" (Mac), or "cuda"')

    # data
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
    parser.add_argument("--context_length", type=int, default=256, help="Context length of LM.")

    # transformer
    parser.add_argument("--vocab_size", type=int, default=10000, help="Size of the vocabulary.")
    parser.add_argument("--d_model", type=int, default=768, help="Dimensions of model.")
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers for Transformer.")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads for Transformer.")
    parser.add_argument("--d_ff", type=int, default=3072, help="Feed-forward hidden dimension.")
    parser.add_argument("--theta", type=int, default=10000, help="Theta value for RoPE.")

    # benchmarking
    parser.add_argument("--warmup_steps", type=int, default=5, help="Number of warm-up steps.")
    parser.add_argument("--num_steps", type=int, default=10, help="Number of timed steps.")
    parser.add_argument("--mode", type=str, default="forward+backward", choices=["forward", "forward+backward"],
                        help="Whether to benchmark forward only or forward+backward.")

    args = parser.parse_args()

    # Initialize model
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.theta,
    ).to(device=args.device)

    # Generate random batch of data
    inputs = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)
    targets = torch.randint(0, args.vocab_size, (args.batch_size, args.context_length), device=args.device)

    def step():
        logits = model(inputs)
        if args.mode == "forward+backward":
            B, S, V = logits.shape
            loss = cross_entropy(logits.reshape(B * S, V), targets.reshape(B * S))
            loss.backward()
        if args.device == "cuda":
            torch.cuda.synchronize()

    # Warm-up
    for _ in range(args.warmup_steps):
        step()

    # Timed steps
    times = []
    for _ in range(args.num_steps):
        start = timeit.default_timer()
        step()
        end = timeit.default_timer()
        times.append(end - start)

    avg = sum(times) / len(times)
    std = (sum((t - avg) ** 2 for t in times) / len(times)) ** 0.5

    print(f"Mode: {args.mode}")
    print(f"Average time per step: {avg:.4f}s")
    print(f"Std dev: {std:.4f}s")


if __name__ == "__main__":
    main()
