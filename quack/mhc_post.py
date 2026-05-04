# Copyright (c) 2026, QuACK team.

import math
from typing import Type

import torch
from torch import Tensor

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr

from quack import copy_utils, layout_utils
from quack.cache_utils import jit_cache
from quack.compile_utils import make_fake_tensor as fake_tensor
from quack.cute_dsl_utils import torch2cute_dtype_map


def _ensure_contiguous(t: Tensor) -> Tensor:
    if torch.compiler.is_compiling():
        return t.contiguous()
    return t if t.is_contiguous() else t.contiguous()


class MHCPost:
    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        mhc: int,
        hidden: int,
        tokens: int | torch.SymInt,
        h_blk: int = 1024,
    ):
        self.dtype = dtype
        self.mhc = mhc
        self.hidden = hidden
        self.max_threads_per_block = 1024
        self.preferred_rows_per_block = (2, 1) if hidden >= 4096 else (1,)
        self.h_blk = math.gcd(hidden, h_blk)
        self.copy_vecsize = math.gcd(hidden, 128 // dtype.width)
        self.threads_per_row = self._threads_per_row(
            dtype,
            hidden,
            h_blk,
            self.max_threads_per_block,
        )
        rows_per_block = self._rows_per_block(tokens)
        assert rows_per_block in self._valid_rows_per_block(), (
            "rows_per_block would over-allocate CUDA block threads"
        )
        self.rows_per_block = rows_per_block
        self.num_threads = self.rows_per_block * self.threads_per_row
        self.num_stages = 2

    @staticmethod
    def _threads_per_row(
        dtype: Type[cutlass.Numeric],
        hidden: int,
        h_blk: int = 1024,
        max_threads_per_row: int = 1024,
    ) -> int:
        h_blk = math.gcd(hidden, h_blk)
        copy_vecsize = math.gcd(hidden, 128 // dtype.width)
        vecs_per_row = h_blk // copy_vecsize
        return min(max_threads_per_row, 1 << int(math.log2(vecs_per_row)))

    def _valid_rows_per_block(self) -> tuple[int, ...]:
        return tuple(
            rows
            for rows in self.preferred_rows_per_block
            if rows * self.threads_per_row <= self.max_threads_per_block
        )

    def _rows_per_block(self, tokens: int | torch.SymInt) -> int:
        if isinstance(tokens, torch.SymInt):
            return 1
        for rows in self._valid_rows_per_block():
            if tokens % rows == 0:
                return rows
        return 1

    def _get_x_tiled_copy(self):
        tiled_copy = copy_utils.tiled_copy_2d(
            self.dtype,
            threads_per_row=self.threads_per_row,
            num_threads=self.num_threads,
            num_copy_elems=self.copy_vecsize,
        )
        return tiled_copy, (self.rows_per_block, self.h_blk)

    def _get_res_tiled_copy(self):
        tiled_copy = copy_utils.tiled_copy_2d(
            self.dtype,
            threads_per_row=self.threads_per_row,
            num_threads=self.num_threads,
            num_copy_elems=self.copy_vecsize,
        )
        return tiled_copy, (self.rows_per_block, self.h_blk, self.mhc)

    @cute.jit
    def __call__(
        self,
        mX: cute.Tensor,  # (tokens, hidden)
        mRes: cute.Tensor,  # (tokens, mhc, hidden)
        mPostLayerMix: cute.Tensor,  # (tokens, mhc)
        mCombResMix: cute.Tensor,  # (tokens, mhc, mhc)
        mO: cute.Tensor,  # (tokens, mhc, hidden)
        stream: cuda.CUstream,
    ):
        assert mX.element_type == self.dtype
        assert mRes.element_type == self.dtype
        assert mO.element_type == self.dtype
        assert mPostLayerMix.element_type == Float32
        assert mCombResMix.element_type == Float32
        assert mX.shape[1] == self.hidden
        assert mRes.shape[1] == self.mhc
        assert mRes.shape[2] == self.hidden
        assert mPostLayerMix.shape[1] == self.mhc
        assert mCombResMix.shape[1] == self.mhc
        assert mCombResMix.shape[2] == self.mhc
        mRes = layout_utils.select(mRes, mode=[0, 2, 1])
        mO = layout_utils.select(mO, mode=[0, 2, 1])
        mPostLayerMix = cute.make_tensor(
            mPostLayerMix.iterator,
            cute.make_layout(
                ((self.threads_per_row, mPostLayerMix.shape[0]), self.mhc),
                stride=((0, self.mhc), 1),
            ),
        )
        mCombResMix = cute.make_tensor(
            mCombResMix.iterator,
            cute.make_layout(
                ((self.threads_per_row, mCombResMix.shape[0]), self.mhc, self.mhc),
                stride=((0, self.mhc * self.mhc), self.mhc, 1),
            ),
        )
        tiled_copy_x, tiler_x = self._get_x_tiled_copy()
        tiled_copy_r, tiler_r = self._get_res_tiled_copy()
        tiler_plm = ((self.threads_per_row, self.rows_per_block), self.mhc)
        tiler_crm = ((self.threads_per_row, self.rows_per_block), self.mhc, self.mhc)
        self.kernel(
            mX,
            mRes,
            mPostLayerMix,
            mCombResMix,
            mO,
            tiler_x,
            tiler_r,
            tiler_plm,
            tiler_crm,
            tiled_copy_x,
            tiled_copy_r,
        ).launch(
            grid=[mX.shape[0] // self.rows_per_block, 1, 1],
            block=[self.num_threads, 1, 1],
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        mX: cute.Tensor,
        mRes: cute.Tensor,
        mPostLayerMix: cute.Tensor,
        mCombResMix: cute.Tensor,
        mO: cute.Tensor,
        tiler_x: cute.Shape,
        tiler_r: cute.Shape,
        tiler_plm: cute.Shape,
        tiler_crm: cute.Shape,
        tiled_copy_x: cute.TiledCopy,
        tiled_copy_r: cute.TiledCopy,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row_block_idx, _, _ = cute.arch.block_idx()

        smem = cutlass.utils.SmemAllocator()
        sX = smem.allocate_tensor(
            mX.element_type,
            cute.make_ordered_layout(
                (self.rows_per_block, self.h_blk, self.num_stages),
                order=(1, 0, 2),
            ),
            byte_alignment=16,
        )
        sRes = smem.allocate_tensor(
            mRes.element_type,
            cute.make_ordered_layout(
                (self.rows_per_block, self.h_blk, self.mhc, self.num_stages),
                order=(1, 0, 2, 3),
            ),
            byte_alignment=16,
        )
        num_h_blocks = const_expr(self.hidden // self.h_blk)

        gX = cute.local_tile(mX, tiler_x, (row_block_idx, None))
        gRes = cute.local_tile(mRes, tiler_r, (row_block_idx, None, 0))
        gO = cute.local_tile(mO, tiler_r, (row_block_idx, None, 0))
        gPLM = cute.local_tile(mPostLayerMix, tiler_plm, (row_block_idx, 0))
        gCRM = cute.local_tile(mCombResMix, tiler_crm, (row_block_idx, 0, 0))

        thr_copy_X = tiled_copy_x.get_slice(tidx)
        thr_copy_R = tiled_copy_r.get_slice(tidx)
        tXgX = thr_copy_X.partition_S(gX)
        tXsX = thr_copy_X.partition_D(sX)
        tRgRes = thr_copy_R.partition_S(gRes)
        tRsRes = thr_copy_R.partition_D(sRes)
        tRgO = thr_copy_R.partition_D(gO)
        tXrX = cute.make_rmem_tensor_like(tXgX[None, None, None, 0])
        tRrRes = cute.make_rmem_tensor_like(tRsRes[None, None, None, None, 0])
        tRrO = cute.make_rmem_tensor_like(tRgO[None, None, None, None, 0])
        tRrAcc = cute.make_rmem_tensor(tRrO.shape, Float32)
        rPLM = cute.make_rmem_tensor(self.mhc, Float32)
        rCRM = cute.make_rmem_tensor(
            cute.make_layout((self.mhc, self.mhc), stride=(self.mhc, 1)),
            Float32,
        )

        stage = Int32(0)
        copy_utils.copy(
            tXgX[None, None, None, 0],
            tXsX[None, None, None, stage],
            is_async=True,
        )
        copy_utils.copy(
            tRgRes[None, None, None, None, 0],
            tRsRes[None, None, None, None, stage],
            is_async=True,
        )
        cute.arch.cp_async_commit_group()

        for mhc_out in cutlass.range_constexpr(self.mhc):
            rPLM[mhc_out] = gPLM[tidx, mhc_out]
            for mhc_in in cutlass.range_constexpr(self.mhc):
                rCRM[mhc_in, mhc_out] = gCRM[tidx, mhc_in, mhc_out]

        for h_block in cutlass.range_constexpr(num_h_blocks):
            if const_expr(h_block + 1 < num_h_blocks):
                copy_utils.copy(
                    tXgX[None, None, None, h_block + 1],
                    tXsX[None, None, None, stage ^ 1],
                    is_async=True,
                )
                copy_utils.copy(
                    tRgRes[None, None, None, None, h_block + 1],
                    tRsRes[None, None, None, None, stage ^ 1],
                    is_async=True,
                )
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(1)
            else:
                cute.arch.cp_async_wait_group(0)
            cute.autovec_copy(tXsX[None, None, None, stage], tXrX)
            cute.autovec_copy(tRsRes[None, None, None, None, stage], tRrRes)
            x = tXrX.load().to(Float32)
            for mhc_out in cutlass.range_constexpr(self.mhc):
                acc = x * rPLM[mhc_out]
                for mhc_in in cutlass.range_constexpr(self.mhc):
                    r = tRrRes[None, None, None, mhc_in].load().to(Float32)
                    acc += rCRM[mhc_in, mhc_out] * r
                tRrAcc[None, None, None, mhc_out].store(acc)
            tRrO.store(tRrAcc.load().to(tRrO.element_type))
            copy_utils.copy(
                tRrO,
                tRgO[None, None, None, None, h_block],
            )
            stage ^= 1


@jit_cache
def _compile_mhc_post_fwd(dtype, mhc, hidden, tokens):
    tokens_sym = cute.sym_int()
    div = math.gcd(128 // dtype.width, hidden)
    post_layer_mix_div = math.gcd(128 // Float32.width, mhc)
    comb_res_mix_div = math.gcd(128 // Float32.width, mhc * mhc)
    x_cute = fake_tensor(dtype, (tokens_sym, hidden), div)
    residual_cute = fake_tensor(dtype, (tokens_sym, mhc, hidden), div)
    post_layer_mix_cute = fake_tensor(Float32, (tokens_sym, mhc), post_layer_mix_div)
    comb_res_mix_cute = fake_tensor(Float32, (tokens_sym, mhc, mhc), comb_res_mix_div)
    out_cute = fake_tensor(dtype, (tokens_sym, mhc, hidden), div)
    return cute.compile(
        MHCPost(dtype, mhc, hidden, tokens),
        x_cute,
        residual_cute,
        post_layer_mix_cute,
        comb_res_mix_cute,
        out_cute,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


@torch.library.custom_op(
    "quack::_mhc_post_fwd",
    mutates_args={"out"},
    device_types="cuda",
)
def _mhc_post_fwd(
    x: Tensor,
    residual: Tensor,
    post_layer_mix: Tensor,
    comb_res_mix: Tensor,
    out: Tensor,
) -> None:
    assert x.dim() == 2, "x must have shape (tokens, hidden)"
    assert residual.dim() == 3, "residual must have shape (tokens, mhc, hidden)"
    assert post_layer_mix.dim() == 2, "post_layer_mix must have shape (tokens, mhc)"
    assert comb_res_mix.dim() == 3, "comb_res_mix must have shape (tokens, mhc, mhc)"
    assert out.shape == residual.shape, "out shape must match residual"
    assert x.shape[0] == residual.shape[0] == post_layer_mix.shape[0] == comb_res_mix.shape[0], (
        "all inputs must have the same token count"
    )
    assert x.shape[1] == residual.shape[2], "x and residual hidden dimensions must match"
    mhc = residual.shape[1]
    hidden = x.shape[1]
    assert post_layer_mix.shape[1] == mhc, "post_layer_mix mhc dimension must match residual"
    assert comb_res_mix.shape[1:] == (mhc, mhc), "comb_res_mix must be square in mhc dimensions"
    assert x.dtype == torch.bfloat16, "x must be bfloat16"
    assert residual.dtype == x.dtype and out.dtype == x.dtype, (
        "residual and out must have the same dtype as x"
    )
    assert post_layer_mix.dtype == torch.float32, "post_layer_mix must be float32"
    assert comb_res_mix.dtype == torch.float32, "comb_res_mix must be float32"
    if out.numel() == 0:
        return
    dtype = torch2cute_dtype_map[x.dtype]
    _compile_mhc_post_fwd(dtype, mhc, hidden, x.shape[0])(
        x, residual, post_layer_mix, comb_res_mix, out
    )


@_mhc_post_fwd.register_fake
def _mhc_post_fwd_fake(
    x: Tensor,
    residual: Tensor,
    post_layer_mix: Tensor,
    comb_res_mix: Tensor,
    out: Tensor,
) -> None:
    from quack.cache_utils import COMPILE_ONLY

    if COMPILE_ONLY and not isinstance(x.size(1), torch.SymInt):
        dtype = torch2cute_dtype_map[x.dtype]
        _compile_mhc_post_fwd(dtype, residual.size(1), x.size(1), x.size(0))


def mhc_post(
    x: Tensor,
    residual: Tensor,
    post_layer_mix: Tensor,
    comb_res_mix: Tensor,
) -> Tensor:
    assert x.dim() == 3, "x must have shape (n0, n1, hidden)"
    assert residual.dim() == 4, "residual must have shape (n0, n1, mhc, hidden)"
    assert post_layer_mix.dim() == 4, (
        "post_layer_mix must have shape (n0, n1, mhc, 1) for the current kernel"
    )
    assert comb_res_mix.dim() == 4, "comb_res_mix must have shape (n0, n1, mhc, mhc)"
    n0, n1, hidden = x.shape
    assert residual.shape[:2] == (n0, n1), "residual leading dimensions must match x"
    mhc = residual.shape[2]
    assert residual.shape[3] == hidden, "residual hidden dimension must match x"
    assert post_layer_mix.shape == (n0, n1, mhc, 1), (
        "post_layer_mix must have shape (n0, n1, mhc, 1)"
    )
    assert comb_res_mix.shape == (n0, n1, mhc, mhc), (
        "comb_res_mix must have shape (n0, n1, mhc, mhc)"
    )
    assert x.dtype == torch.bfloat16, "x must be bfloat16"
    assert residual.dtype == torch.bfloat16, "residual must be bfloat16"
    assert post_layer_mix.dtype == torch.float32, "post_layer_mix must be float32"
    assert comb_res_mix.dtype == torch.float32, "comb_res_mix must be float32"

    x = _ensure_contiguous(x)
    residual = _ensure_contiguous(residual)
    post_layer_mix = _ensure_contiguous(post_layer_mix)
    comb_res_mix = _ensure_contiguous(comb_res_mix)

    tokens = n0 * n1
    x_2d = x.reshape(tokens, hidden)
    residual_3d = residual.reshape(tokens, mhc, hidden)
    post_layer_mix_2d = post_layer_mix.reshape(tokens, mhc)
    comb_res_mix_3d = comb_res_mix.reshape(tokens, mhc, mhc)
    out_3d = torch.empty_like(residual_3d)
    _mhc_post_fwd(x_2d, residual_3d, post_layer_mix_2d, comb_res_mix_3d, out_3d)
    return out_3d.reshape(n0, n1, mhc, hidden)


def mhc_post_ref(
    x: Tensor,
    residual: Tensor,
    post_layer_mix: Tensor,
    comb_res_mix: Tensor,
) -> Tensor:
    """PyTorch reference implementation for MHC post-processing."""
    term2 = torch.einsum("abmn,abmc->abnc", comb_res_mix, residual.float())
    return (x.float().unsqueeze(-2) * post_layer_mix + term2).bfloat16()


__all__ = ["MHCPost", "mhc_post", "mhc_post_ref"]
