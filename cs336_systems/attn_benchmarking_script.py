
import argparse
import itertools
import timeit
from datetime import datetime
from pathlib import Path
import torch
import pandas as pd
from cs336_basics.model import scaled_dot_product_attention


D_MODEL = [16, 32, 64, 128]
SEQ_LEN = [256, 1024, 4096, 8192, 16384]
BATCH = 8
DEVICE = "cuda"

parser = argparse.ArgumentParser()
parser.add_argument("--compiled", action="store_true")
args = parser.parse_args()
COMPILED = args.compiled

results = []

for d_model, seq_len in list(itertools.product(D_MODEL, SEQ_LEN)):
    print(f"d_model: {d_model}, seq_len: {seq_len}")

    Q = torch.randn((BATCH, seq_len, d_model), device=DEVICE, requires_grad=True)
    K = torch.randn((BATCH, seq_len, d_model), device=DEVICE, requires_grad=True)
    V = torch.randn((BATCH, seq_len, d_model), device=DEVICE, requires_grad=True)

    try:
    # warmup
        for _ in range(10):
            attn_fn = torch.compile(scaled_dot_product_attention) if COMPILED else scaled_dot_product_attention
            out = attn_fn(Q, K, V)
            torch.cuda.synchronize()

        f_time_start = timeit.default_timer()
        for _ in range(100):
            out = attn_fn(Q, K, V)
            torch.cuda.synchronize()
        f_time = timeit.default_timer() - f_time_start

        # Memory in use before backward starts (after a single forward).
        out = attn_fn(Q, K, V)
        torch.cuda.synchronize()
        mem_before_bwd = torch.cuda.memory_allocated()

        b_time_start = timeit.default_timer()
        for _ in range(100):
            out = attn_fn(Q, K, V)
            out.backward(gradient=torch.ones_like(out))
            torch.cuda.synchronize()
            Q.grad = K.grad = V.grad = None
        b_time = timeit.default_timer() - b_time_start - f_time

        results.append({
            "d_model": d_model,
            "seq_len": seq_len,
            "forward_s": f_time,
            "backward_s": b_time,
            "mem_before_bwd_MB": mem_before_bwd / (1024 ** 2),
        })

    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        results.append({
            "d_model": d_model,
            "seq_len": seq_len,
            "forward_s": "OOM",
            "backward_s": "OOM",
            "mem_before_bwd_MB": "OOM",
        })

df = pd.DataFrame(results)
print("\n")
print(df.to_markdown(index=False))

out_path = Path(__file__).parent.parent / "results" / f"attn_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}_comp_{COMPILED}.csv"
df.to_csv(out_path, index=False)
print(f"Saved to {out_path}")


    


