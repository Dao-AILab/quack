"""Minimal A/B test for the gated SM100 fix.

Phase A — _valid_2cta_m returns (128, 256): the original buggy default.
Phase B — _valid_2cta_m returns (256,):       the patched default.

Each phase runs in its own subprocess with its own QUACK_CACHE_DIR so the
disk-backed jit_cache (whose key doesn't include use_2cta_instrs) can't serve
a stale cross-phase kernel.

Usage: CUDA_VISIBLE_DEVICES=<gpu> python solo_ab_min.py
"""
import json, os, shutil, subprocess, sys, tempfile, torch


def child(phase):
    valid = (128, 256) if phase == "A" else (256,)
    from quack.gemm_act import GemmGatedMixin
    GemmGatedMixin._valid_2cta_m = lambda self, _v=valid: _v

    from quack.gemm_config import GemmConfig
    from quack.gemm_interface import gemm_gated_tuned, gemm_gated_ref

    M, H, I, E = 32768, 1024, 512, 8
    dtype, dev = torch.float16, torch.device("cuda:0")
    g = torch.Generator(device=dev).manual_seed(0)
    counts = torch.full((E,), M // E, dtype=torch.int32, device=dev)
    cu = torch.zeros(E + 1, dtype=torch.int32, device=dev)
    cu[1:] = torch.cumsum(counts, dim=0).to(torch.int32)
    T = M // 4
    x = (0.02 * torch.randn(T, H, generator=g, device=dev, dtype=torch.float32)).to(dtype)
    A_idx = torch.randint(0, T, (M,), dtype=torch.int32, device=dev, generator=g)
    w = torch.empty(E, 2 * I, H, dtype=torch.float32, device=dev)
    torch.nn.init.normal_(w, mean=0.0, std=0.02, generator=g)
    w1 = w.to(dtype).permute(1, 2, 0).permute(2, 1, 0)

    cfg = GemmConfig(tile_m=128, tile_n=256, cluster_m=2, cluster_n=1,
                     swap_ab=False, max_swizzle_size=8,
                     is_dynamic_persistent=True, use_tma_gather=True,
                     pingpong=False, device_capacity=10)
    pre, post = torch.empty(M, 2 * I, dtype=dtype, device=dev), torch.empty(M, I, dtype=dtype, device=dev)
    gemm_gated_tuned.fn(x, w1, pre, post, None, None, "swiglu", cu, A_idx, False, config=cfg)
    torch.cuda.synchronize()
    pre_ref, post_ref = gemm_gated_ref(x, w1, bias=None, activation="swiglu",
                                       cu_seqlens_m=cu, A_idx=A_idx,
                                       store_preact=True, concat_layout=None)

    print("PHASE_RESULT " + json.dumps({
        "phase": phase, "valid_2cta_m": list(valid),
        "preact_max_abs": (pre.float() - pre_ref.float()).abs().max().item(),
        "preact_max_ref": pre_ref.float().abs().max().item(),
        "postact_max_abs": (post.float() - post_ref.float()).abs().max().item(),
        "postact_max_ref": post_ref.float().abs().max().item(),
    }), flush=True)


def run_phase(phase):
    cache = tempfile.mkdtemp(prefix=f"quack_cache_{phase}_")
    env = {**os.environ, "QUACK_CACHE_DIR": cache}
    try:
        out = subprocess.run([sys.executable, "-u", __file__, phase],
                             capture_output=True, text=True, env=env, timeout=600)
    finally:
        shutil.rmtree(cache, ignore_errors=True)
    for line in out.stdout.splitlines():
        if line.startswith("PHASE_RESULT "):
            return json.loads(line[len("PHASE_RESULT "):])
    sys.exit(f"phase {phase} produced no result\n{out.stdout}\n{out.stderr}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("A", "B"):
        child(sys.argv[1]); return

    a, b = run_phase("A"), run_phase("B")
    for d in (a, b):
        rel = d["preact_max_abs"] / max(d["preact_max_ref"], 1e-12)
        print(f"phase {d['phase']}  _valid_2cta_m={str(tuple(d['valid_2cta_m'])):<10}  "
              f"preact rel={rel:.4e}  ({'FAIL' if rel > 0.05 else 'PASS'})")


if __name__ == "__main__":
    main()
