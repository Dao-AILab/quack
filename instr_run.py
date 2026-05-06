"""Tiny repro to capture instrumentation prints from the gated forward path
at the buggy cocktail (tile_m=128, cm=2, clc=True, gather=True).

Uses small M to keep output manageable. The instrumentation print()s in
quack/gemm_base.py (D path) and quack/epi_ops.py (TileStore.to_params, aux
path) will fire during kernel construction and dump epi_tile + smem_layout
+ tma_atom for both D and aux side-by-side.
"""
import os
import sys
import torch

from quack.gemm_config import GemmConfig
from quack.gemm_interface import gemm_gated_tuned, gemm_gated_ref


def main():
    import os
    M = int(os.environ.get("M", "4096"))
    H = int(os.environ.get("H", "256"))
    I = int(os.environ.get("I", "128"))
    E = int(os.environ.get("E", "4"))
    device = torch.device("cuda:0")
    dtype = torch.float16
    g = torch.Generator(device=device).manual_seed(0)
    counts = torch.full((E,), M // E, dtype=torch.int32, device=device)
    cu = torch.zeros(E + 1, dtype=torch.int32, device=device)
    cu[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
    T = M // 4
    x = (0.02 * torch.randn(T, H, generator=g, device=device, dtype=torch.float32)).to(dtype)
    A_idx = torch.randint(0, T, (M,), dtype=torch.int32, device=device, generator=g)
    w = torch.empty(E, 2 * I, H, dtype=torch.float32, device=device)
    torch.nn.init.normal_(w, mean=0.0, std=0.02, generator=g)
    w1 = w.to(dtype).permute(1, 2, 0).permute(2, 1, 0)

    cluster_m = int(os.environ.get("CLUSTER_M", "2"))
    cfg = GemmConfig(
        tile_m=128, tile_n=256, cluster_m=cluster_m, cluster_n=1,
        swap_ab=False, max_swizzle_size=8,
        is_dynamic_persistent=True, use_tma_gather=True,
        pingpong=False, device_capacity=10,
    )
    print(f"\n>>> cluster_m={cluster_m} (warp-shape: {(2,2) if cluster_m==2 else (4,1)})\n", flush=True)
    pre = torch.empty(M, 2 * I, dtype=dtype, device=device)
    post = torch.empty(M, I, dtype=dtype, device=device)

    print("\n========== INVOKING gemm_gated_tuned.fn (buggy cocktail) ==========\n", flush=True)
    gemm_gated_tuned.fn(
        x, w1, pre, post, None, None, "swiglu", cu, A_idx, False, config=cfg,
    )
    torch.cuda.synchronize()
    print("\n========== KERNEL EXECUTION DONE ==========\n", flush=True)

    pre_ref, post_ref = gemm_gated_ref(
        x, w1, bias=None, activation="swiglu",
        cu_seqlens_m=cu, A_idx=A_idx,
        store_preact=True, concat_layout=None,
    )
    pre_diff = (pre.float() - pre_ref.float()).abs()
    post_diff = (post.float() - post_ref.float()).abs()
    print(f"\npreact rel  = {pre_diff.max().item() / max(pre_ref.float().abs().max().item(), 1e-12):.4e}")
    print(f"postact rel = {post_diff.max().item() / max(post_ref.float().abs().max().item(), 1e-12):.4e}")
    # Inspect output values at row 0, columns 0..15: which pattern of corruption?
    print("\n  postact   row=0 cols=0..15:", post[0, :16].float().tolist())
    print("  postact_ref row=0 cols=0..15:", post_ref[0, :16].float().tolist())
    print("\n  postact   row=0 cols=64..79:", post[0, 64:80].float().tolist())
    print("  postact_ref row=0 cols=64..79:", post_ref[0, 64:80].float().tolist())
    # Check if postact has zeros (skipped writes) or shifted/scrambled values.
    n_zeros = (post == 0).sum().item()
    print(f"\n  postact: total elems = {post.numel()}, n_zeros = {n_zeros} ({100*n_zeros/post.numel():.1f}%)")
    n_zeros_ref = (post_ref == 0).sum().item()
    print(f"  postact_ref: n_zeros = {n_zeros_ref}")


if __name__ == "__main__":
    main()
