# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Copyright (c) 2026, Tri Dao.
#
# Device code adapted for Quack from ROCm/FlyDSL commit
# ddaa507f56aa3fe9c08ebe6161a717b755540248.

"""Direct eager RMSNorm forward for ROCm gfx950 using FlyDSL.

:meth:`RMSNorm.compile`'s kernel is laid out section for section against
``quack/rmsnorm.py``'s ``RMSNorm.kernel``, over FlyDSL siblings of the same
helpers (:mod:`quack.flydsl_copy_utils`, :mod:`quack.flydsl_reduce`,
:mod:`quack.flydsl_reduction_base`). Where it departs, it is because CDNA
differs from the SM90+ target the CuTe kernel is written for:

* no cluster tier, so no distributed shared memory and no mbarriers;
* no ``cp.async``, so a tile lands in registers rather than being staged through
  LDS -- which also means a masked lane keeps its old value instead of being
  zero-filled, and the fragment is cleared before a predicated load;
* a 128-bit ceiling on a buffer access, so operands of different element widths
  need different atoms over one tile (see :mod:`quack.flydsl_copy_utils`);
* rows too wide to keep in registers are walked in several tiles, where the CuTe
  kernel always covers a row with one.
"""

import functools
import hashlib
import math
import numbers
from functools import partial
from pathlib import Path

import torch

from quack._platform import IS_ROCM_BUILD
from quack.flydsl_constants import MAX_ACCESS_BITS

if not IS_ROCM_BUILD:
    raise ImportError("quack.rmsnorm_flydsl requires a ROCm PyTorch build")

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr
from flydsl.expr import math as fmath
from flydsl.expr.typing import ReductionOp

import quack.flydsl_copy_utils as copy_utils
from quack.flydsl_reduce import make_reduction_buffer, row_reduce
from quack.flydsl_reduction_base import ReductionBase
from quack.flydsl_runtime import (
    SUPPORTED_DTYPES as _SUPPORTED_DTYPES,
)
from quack.flydsl_runtime import (
    Launcher,
    current_raw_stream,
    dtype_spec,
    empty_placeholder,
    packed_rows,
    run_compiled,
)
from quack.rmsnorm_flydsl_config import (
    MAX_N,
    RmsNormFwdConfig,
)

__all__ = ["rmsnorm_fwd"]

_SUPPORTED_ARCHES = frozenset({"gfx950"})
_MAX_ROWS = 2**31 - 1
_MAX_BUFFER_BYTES = 1 << 32
# rstd's buffer descriptor carries a 32-bit num_records over fp32 elements.
_MAX_RSTD_ROWS = (2**32 - 1) // 4
_KERNEL_HELPER_FILES = (
    "flydsl_constants.py",
    "flydsl_copy_utils.py",
    "flydsl_reduce.py",
)


def _kernel_helper_digest() -> str:
    """Fingerprint helper sources that FlyDSL's module lookup cannot see."""
    digest = hashlib.sha256()
    source_dir = Path(__file__).parent
    for name in _KERNEL_HELPER_FILES:
        source = (source_dir / name).read_bytes()
        digest.update(name.encode())
        digest.update(len(source).to_bytes(8, "little"))
        digest.update(source)
    return digest.hexdigest()


_KERNEL_HELPER_DIGEST = _kernel_helper_digest()


def _ceil_div_positive_i32(value, divisor: int):
    """Ceil-divide a positive Int32 without overflowing at its upper limit."""
    one = fx.Int32(1)
    return (value - one) // fx.Int32(divisor) + one


def _buffer_tile_fits(tensor: torch.Tensor, rows_per_block: int, m: int, n: int) -> bool:
    """Whether one CTA's byte span fits the buffer descriptor's 32-bit offset."""
    rows = min(rows_per_block, m)
    span_elements = (rows - 1) * tensor.stride(0) + n
    return span_elements * tensor.element_size() < _MAX_BUFFER_BYTES


@functools.cache
def _forward_config(n: int, input_bits: int) -> RmsNormFwdConfig:
    """Share the host geometry lookup between layout gating and compilation."""
    return RmsNormFwdConfig.for_forward(n, input_bits)


class RMSNorm(ReductionBase):
    """RMSNorm forward for one feature set, as ``quack.rmsnorm.RMSNorm`` is.

    Everything the geometry depends on is fixed here on the host, because the
    kernel is specialized per feature set anyway: a FlyDSL ``TiledCopy`` is a
    bundle of MLIR values and cannot be passed in from outside the trace the way
    the CuTe kernel receives one.
    """

    def __init__(
        self,
        dtype,
        N: int,
        config: RmsNormFwdConfig,
        num_heads: int,
        *,
        has_weight: bool,
        has_bias: bool,
        has_residual: bool,
        store_residual: bool,
        store_rstd: bool,
        apply_weight_offset: bool,
        use_buffer_copy: bool,
    ):
        super().__init__(dtype, N)
        self.config = config
        self.num_heads = num_heads
        self.has_weight = has_weight
        self.has_bias = has_bias
        self.has_residual = has_residual
        self.store_residual = store_residual
        self.store_rstd = store_rstd
        self.apply_weight_offset = apply_weight_offset
        self.use_buffer_copy = use_buffer_copy

    def _threads_per_row(self):
        return self.config.num_threads

    def _num_threads(self):
        return self.config.num_threads * self.config.rows_per_block

    def _blocks_per_tile(self, vecsize: int):
        # A row wide enough that keeping it resident would spill takes one block
        # per tile and is read twice; anything else is covered by a single tile.
        return 1 if self.config.reload_from_gmem else self._num_blocks_n(vecsize)

    def compile(self) -> Launcher:
        """Trace and compile this specialization.

        Stands in for the CuTe class's ``@cute.jit __call__`` plus
        ``@cute.kernel kernel`` pair: FlyDSL rewrites control flow into ``scf``
        only within the body it decorates, so the device code cannot be split
        across methods the way CuTe's can.

        Every value the kernel body branches on is bound to a local first, and
        the body reads no attribute of ``self``. That is what makes two feature
        sets distinct to FlyDSL: it keys an artifact on the kernel's source plus
        the scalar closure values it can see, and an object reference tells it
        nothing, so a flag reached through ``self`` would let the first feature
        set compiled answer for every other one.
        """
        N, num_heads = self.N, self.num_heads
        has_weight, has_bias, has_residual = self.has_weight, self.has_bias, self.has_residual
        store_residual, store_rstd = self.store_residual, self.store_rstd
        apply_weight_offset = self.apply_weight_offset
        use_buffer_copy = self.use_buffer_copy

        vecsize = self.config.vecsize
        n_tiles = self._num_tiles_n(vecsize)
        threads_per_row, num_threads = self._threads_per_row(), self._num_threads()
        tiler_mn = self._get_tiler(vecsize)
        reduction_shape = self._reduction_buffer_shape()
        is_even_N = N == tiler_mn[1] * n_tiles
        rows_per_block = tiler_mn[0]
        kernel_helper_digest = _KERNEL_HELPER_DIGEST
        # A tile one row tall makes the grid cover the rows exactly, so no block
        # overhangs and the M-dimension guard is dead. Only a tile that packs
        # several short rows can have its last block run past the last row.
        grid_covers_rows = rows_per_block == 1
        launch_bounds = {} if num_threads <= 256 else {"known_block_size": [num_threads, 1, 1]}

        @flyc.kernel(**launch_bounds)
        def rmsnorm_kernel(
            mX: fx.Tensor,  # (M, H, N)
            mW: fx.Tensor,  # (H, N)
            mB: fx.Tensor,  # (H, N)
            mRes: fx.Tensor,  # (M, H, N)
            mO: fx.Tensor,  # (M, H, N)
            mResO: fx.Tensor,  # (M, H, N)
            mRstd: fx.Tensor,  # (M, H)
            num_rows: fx.Int32,
            eps: fx.Float32,
            weight_offset: fx.Float32,
        ):
            tidx = fx.thread_idx.x
            flat_block = fx.block_idx.x
            bidx = flat_block // fx.Int32(num_heads)
            bidz = flat_block % fx.Int32(num_heads)

            tiled_copy = copy_utils.TiledCopy2d(
                threads_per_row,
                num_threads,
                vecsize,
                tidx,
                use_buffer_copy=use_buffer_copy,
            )
            reduction_buffer = make_reduction_buffer(*reduction_shape)

            # Drop the operands this specialization does not carry, so every use
            # below is the compile-time test against None that the CuTe kernel
            # makes against its Optional arguments.
            mW = mW if const_expr(has_weight) else None
            mB = mB if const_expr(has_bias) else None
            mRes = mRes if const_expr(has_residual) else None
            mResO = mResO if const_expr(store_residual) else None
            mRstd = mRstd if const_expr(store_rstd) else None

            # Slice per head. Inputs are always rank 3 and parameters rank 2, so
            # a plain RMSNorm is the H == 1 case of the per-head one.
            mX, mRes, mO, mResO = [
                mT[None, bidz, None] if const_expr(mT is not None) else None
                for mT in (mX, mRes, mO, mResO)
            ]
            mW, mB = [mT[bidz, None] if const_expr(mT is not None) else None for mT in (mW, mB)]
            mRstd = mRstd[None, bidz] if const_expr(mRstd is not None) else None

            shape = (num_rows, N)

            # Weight and bias repeat down a tile, rstd across it, so one tiler
            # covers every operand.
            mW, mB = [copy_utils.expand(mT, dim=0, size=tiler_mn[0]) for mT in (mW, mB)]
            mRstd = copy_utils.expand(mRstd, dim=1, size=N)

            # Slice for CTAs, then use a buffer descriptor only when every active
            # multi-row tile fits its 32-bit byte offset. The fallback keeps the
            # raw 64-bit pointers for UniversalCopy. The trailing None keeps the
            # mode the row's N-tiles are indexed by.
            gX, gRes, gO, gResO, gRstd = [
                copy_utils.buffer_tensor(
                    copy_utils.local_tile(mT, tiler_mn, (bidx, None)),
                    use_buffer_copy,
                )
                for mT in (mX, mRes, mO, mResO, mRstd)
            ]
            gW, gB = [
                copy_utils.buffer_tensor(
                    copy_utils.local_tile(mT, tiler_mn, (0, None)),
                    use_buffer_copy,
                )
                for mT in (mW, mB)
            ]

            tXgW = tiled_copy.partition_S(gW)
            tXgB = tiled_copy.partition_S(gB)
            tXgX = tiled_copy.partition_S(gX)
            tXgRes = tiled_copy.partition_S(gRes)
            tXgO = tiled_copy.partition_D(gO)
            tXgResO = tiled_copy.partition_D(gResO)
            tXrRstd = tiled_copy.partition_D(gRstd)
            tXcX = copy_utils.thread_coords(tiled_copy, bidx, tiler_mn)

            # allocate fragments for gmem->rmem, one tile wide
            tXrW, tXrB, tXrX, tXrRes, tXrO, tXrResO = [
                copy_utils.make_fragment(t) for t in (tXgW, tXgB, tXgX, tXgRes, tXgO, tXgResO)
            ]

            tXpX = (
                copy_utils.predicate_k(tXcX, limit=shape[1]) if const_expr(not is_even_N) else None
            )
            # Each copy will use the same predicate
            copy = partial(
                copy_utils.copy,
                pred=tXpX,
                use_buffer_copy=use_buffer_copy,
            )

            in_row = True if const_expr(grid_covers_rows) else tXcX.row < num_rows

            # Both passes walk the row a tile at a time. A row that fits in
            # registers is a single tile, so the fragments this pass fills are
            # still live for the epilogue; a wider row re-copies them below.
            sum_sq_x = fx.Float32(0.0)
            for tile in range(n_tiles):
                if in_row:
                    copy(tXgX, tXrX, tile=tile)
                    if const_expr(mRes is not None):
                        copy(tXgRes, tXrRes, tile=tile)
                x = copy_utils.load(tXrX, fx.Float32)
                if const_expr(mRes is not None):
                    x = x + copy_utils.load(tXrRes, fx.Float32)
                if const_expr(mResO is not None):
                    copy_utils.store(tXrResO, x)
                    if in_row:
                        copy(tXrResO, tXgResO, tile=tile)
                sum_sq_x = sum_sq_x + (x * x).reduce(ReductionOp.ADD)

            sum_sq_x = row_reduce(sum_sq_x, ReductionOp.ADD, threads_per_row, reduction_buffer)
            rstd = fmath.rsqrt(sum_sq_x / fx.Float32(shape[1]) + eps)
            if const_expr(mRstd is not None):  # noqa: SIM102 - compile-time guard
                # Only the thread corresponding to column 0 writes out the rstd to gmem
                if in_row:  # noqa: SIM102 - keep runtime predicates separate
                    if tXcX.col == fx.Int32(0):
                        tXrRstd[(0, 0), 0, 0, 0] = rstd

            for tile in range(n_tiles):
                if const_expr(n_tiles > 1):  # noqa: SIM102 - compile-time guard
                    # The tile the first pass left in registers is gone by now.
                    if in_row:
                        copy(tXgX, tXrX, tile=tile)
                        if const_expr(mRes is not None):
                            copy(tXgRes, tXrRes, tile=tile)
                x = copy_utils.load(tXrX, fx.Float32)
                if const_expr(mRes is not None):
                    x = x + copy_utils.load(tXrRes, fx.Float32)
                y = x * rstd
                if const_expr(mW is not None):
                    copy(tXgW, tXrW, tile=tile)
                    w = copy_utils.load(tXrW, fx.Float32)
                    if const_expr(apply_weight_offset):
                        # fp32 add so e.g. (1 + w) doesn't round through the weight dtype
                        w = w + weight_offset
                    y = y * w
                if const_expr(mB is not None):
                    copy(tXgB, tXrB, tile=tile)
                    y = y + copy_utils.load(tXrB, fx.Float32)
                copy_utils.store(tXrO, y)
                if in_row:
                    copy(tXrO, tXgO, tile=tile)

        @flyc.jit
        def launch_rmsnorm(
            mX: fx.Tensor,
            mW: fx.Tensor,
            mB: fx.Tensor,
            mRes: fx.Tensor,
            mO: fx.Tensor,
            mResO: fx.Tensor,
            mRstd: fx.Tensor,
            m: fx.Int32,
            eps: fx.Float32,
            weight_offset: fx.Float32,
            stream: fx.Stream = fx.Stream(None),  # noqa: B008 - FlyDSL traced ABI
        ):
            # Module-qualified helper calls are invisible to FlyDSL's source
            # walker, so capture their content digest in the artifact key.
            _ = kernel_helper_digest
            blocks_per_head = _ceil_div_positive_i32(m, rows_per_block)
            rmsnorm_kernel(mX, mW, mB, mRes, mO, mResO, mRstd, m, eps, weight_offset).launch(
                grid=(
                    blocks_per_head * fx.Int32(num_heads),
                    1,
                    1,
                ),
                block=(num_threads, 1, 1),
                stream=stream,
            )

        return Launcher(flyc.compile[{"fastmath": "fast"}](launch_rmsnorm))


@functools.cache
def _compiled_forward(
    device_index: int,
    n: int,
    input_torch_dtype: torch.dtype,
    output_torch_dtype: torch.dtype,
    weight_torch_dtype: torch.dtype,
    bias_torch_dtype: torch.dtype,
    residual_torch_dtype: torch.dtype,
    residual_out_torch_dtype: torch.dtype,
    has_weight: bool,
    has_bias: bool,
    has_residual: bool,
    store_residual: bool,
    store_rstd: bool,
    num_heads: int,
    apply_weight_offset: bool,
    use_buffer_copy: bool,
):
    """Build and memoize one feature-specialized forward launcher.

    Arguments are the cache key. ``device_index`` is in it because FlyDSL keys
    artifacts by argument signature alone (see :mod:`quack.flydsl_runtime`).
    torch dtypes resolve here, so device code captures only FlyDSL types.
    """
    input_dtype, input_bits = dtype_spec(input_torch_dtype)

    config = _forward_config(n, input_bits)
    # An input span is always one copy, so a cached tile stays in the input dtype.
    # A wider operand takes several atoms over the same tile, each built where
    # the tile is partitioned (see :mod:`quack.flydsl_copy_utils`).
    assert config.vecsize * input_bits <= MAX_ACCESS_BITS

    return RMSNorm(
        input_dtype,
        n,
        config,
        num_heads,
        has_weight=has_weight,
        has_bias=has_bias,
        has_residual=has_residual,
        store_residual=store_residual,
        store_rstd=store_rstd,
        apply_weight_offset=apply_weight_offset,
        use_buffer_copy=use_buffer_copy,
    ).compile()


def _validate_inputs(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    out_dtype: torch.dtype | None,
    residual_dtype: torch.dtype | None,
    store_rstd: bool,
    weight_offset: float,
) -> tuple[int, int, int, bool]:
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"x must be a torch.Tensor, got {type(x).__name__}")
    if x.ndim < 1:
        raise ValueError("x must have at least one dimension")
    optional = (("weight", weight), ("bias", bias), ("residual", residual))
    for name, tensor in optional:
        if tensor is not None and not isinstance(tensor, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor or None, got {type(tensor).__name__}")

    parameter_ranks = {tensor.ndim for tensor in (weight, bias) if tensor is not None}
    if not parameter_ranks.issubset({1, 2}):
        raise ValueError("weight and bias must be 1-D or 2-D")
    if len(parameter_ranks) > 1:
        raise ValueError("weight and bias must use the same rank")
    per_head = parameter_ranks == {2}
    if per_head:
        if x.ndim < 2:
            raise ValueError("per-head RMSNorm requires an input with at least two dimensions")
        num_heads, n = x.shape[-2:]
        if num_heads < 1:
            raise ValueError("per-head RMSNorm requires at least one head")
        parameter_shape = (num_heads, n)
    else:
        num_heads, n = 1, x.shape[-1]
        parameter_shape = (n,)

    if not 1 <= n <= MAX_N:
        raise ValueError(f"x normalized dimension must be between 1 and {MAX_N}, got {n}")
    for name, tensor in (("weight", weight), ("bias", bias)):
        if tensor is not None and tuple(tensor.shape) != parameter_shape:
            raise ValueError(f"{name} shape must be {parameter_shape}, got {tuple(tensor.shape)}")
    if residual is not None and residual.shape != x.shape:
        raise ValueError(
            f"residual shape must match x, got {tuple(residual.shape)}/{tuple(x.shape)}"
        )

    if x.dtype not in _SUPPORTED_DTYPES:
        raise TypeError(f"x dtype must be float16, bfloat16, or float32, got {x.dtype}")
    alignment = MAX_ACCESS_BITS // (x.element_size() * 8)
    if n % alignment:
        raise ValueError(f"x normalized dimension must be a multiple of {alignment}, got {n}")
    for name, dtype in (("out_dtype", out_dtype), ("residual_dtype", residual_dtype)):
        if dtype is not None and dtype not in _SUPPORTED_DTYPES:
            raise TypeError(f"{name} must be float16, bfloat16, or float32, got {dtype}")

    device = x.device
    if device.type != "cuda":
        raise ValueError(f"x must be on a ROCm device, got {device}")
    if x.layout != torch.strided:
        raise ValueError(f"x must use torch.strided layout, got {x.layout}")
    for name, tensor in optional:
        if tensor is None:
            continue
        if tensor.dtype not in _SUPPORTED_DTYPES:
            raise TypeError(
                f"{name} dtype must be float16, bfloat16, or float32, got {tensor.dtype}"
            )
        if tensor.layout != torch.strided:
            raise ValueError(f"{name} must use torch.strided layout, got {tensor.layout}")
        if tensor.device != device:
            raise ValueError(
                f"x and {name} must be on the same device, got {device}/{tensor.device}"
            )

    if weight is None and weight_offset != 0.0:
        raise ValueError("weight_offset requires an explicit weight")

    m = x.numel() // (num_heads * n)
    normalized_rows = m * num_heads
    if normalized_rows > _MAX_ROWS:
        raise ValueError(
            f"x has {normalized_rows} normalized rows, but the kernel addresses at most {_MAX_ROWS}"
        )
    if store_rstd and normalized_rows > _MAX_RSTD_ROWS:
        raise ValueError(
            f"rstd has {normalized_rows} rows, but its buffer descriptor addresses at most "
            f"{_MAX_RSTD_ROWS}"
        )
    return m, n, num_heads, per_head


def _validate_scalars(eps, store_rstd, weight_offset):
    """Check the Python scalars before the dispatcher can coerce them.

    Outside the custom op, since the dispatcher casts to the schema's types:
    `eps=True` would reach the body as 1.0 and `store_rstd=1` as True, so
    torch.compile would accept what eager rejects. The `type(x) is float` fast
    paths skip numbers.Real's __instancecheck__ on the common case.
    """
    if type(eps) is not float:
        if isinstance(eps, bool) or not isinstance(eps, numbers.Real):
            raise TypeError(f"eps must be a real number, got {type(eps).__name__}")
        eps = float(eps)
    if not 0.0 < eps < math.inf:
        raise ValueError(f"eps must be finite and positive, got {eps}")
    if store_rstd is not True and store_rstd is not False:
        raise TypeError(f"store_rstd must be a bool, got {type(store_rstd).__name__}")
    if type(weight_offset) is not float:
        if isinstance(weight_offset, bool) or not isinstance(weight_offset, numbers.Real):
            raise TypeError(
                f"weight_offset must be a real number, got {type(weight_offset).__name__}"
            )
        weight_offset = float(weight_offset)
    if not -math.inf < weight_offset < math.inf:
        raise ValueError(f"weight_offset must be finite, got {weight_offset}")
    return eps, weight_offset


def _output_dtypes(x, residual, out_dtype, residual_dtype):
    """Resolve the two output dtypes and whether the residual sum is stored."""
    output_dtype = x.dtype if out_dtype is None else out_dtype
    residual_out_dtype = (
        residual_dtype
        if residual_dtype is not None
        else (residual.dtype if residual is not None else x.dtype)
    )
    store_residual = residual is not None or (
        residual_dtype is not None and residual_dtype != x.dtype
    )
    return output_dtype, residual_out_dtype, store_residual


def _absent(x, dtype):
    """Stand-in for an output this call does not produce: a custom op has fixed
    arity and its fake must predict shapes, so "no rstd" cannot be None here."""
    return torch.empty(0, device=x.device, dtype=dtype)


def _rmsnorm_fwd_core(
    x, weight, bias, residual, out_dtype, residual_dtype, eps, store_rstd, weight_offset
):
    """Shared body; absent outputs are None, not sentinel tensors. Two empty CUDA
    tensors cost 2.1us, an eighth of the host path, and only the op needs them."""
    m, n, num_heads, _ = _validate_inputs(
        x, weight, bias, residual, out_dtype, residual_dtype, store_rstd, weight_offset
    )
    output_dtype, residual_out_dtype, store_residual = _output_dtypes(
        x, residual, out_dtype, residual_dtype
    )

    if m == 0:
        return (
            torch.empty(x.shape, device=x.device, dtype=output_dtype),
            torch.empty(x.shape, device=x.device, dtype=residual_out_dtype)
            if store_residual
            else None,
            torch.empty(x.shape[:-1], device=x.device, dtype=torch.float32) if store_rstd else None,
        )

    # Rank 3 on the device even without heads, so the kernel has one shape to
    # partition rather than a per-head and a plain variant.
    x_flat = packed_rows(x.reshape(-1, num_heads, n))
    weight_arg = (
        packed_rows(weight.reshape(num_heads, n))
        if weight is not None
        else empty_placeholder(x.device, x.dtype)
    )
    bias_arg = (
        packed_rows(bias.reshape(num_heads, n))
        if bias is not None
        else empty_placeholder(x.device, x.dtype)
    )
    residual_arg = (
        packed_rows(residual.reshape(-1, num_heads, n))
        if residual is not None
        else empty_placeholder(x.device, x.dtype)
    )

    out_flat = torch.empty(x_flat.shape, device=x.device, dtype=output_dtype)
    residual_out_flat = (
        torch.empty(x_flat.shape, device=x.device, dtype=residual_out_dtype)
        if store_residual
        else empty_placeholder(x.device, residual_out_dtype)
    )
    rstd_flat = (
        torch.empty(m, num_heads, device=x.device, dtype=torch.float32)
        if store_rstd
        else empty_placeholder(x.device, torch.float32)
    )

    config = _forward_config(n, x_flat.element_size() * 8)
    # One-row CTAs never put the runtime row pitch in a descriptor offset.
    if config.rows_per_block == 1:
        use_buffer_copy = True
    else:
        use_buffer_copy = (
            _buffer_tile_fits(x_flat, config.rows_per_block, m, n)
            and _buffer_tile_fits(out_flat, config.rows_per_block, m, n)
            and (residual is None or _buffer_tile_fits(residual_arg, config.rows_per_block, m, n))
            and (
                not store_residual
                or _buffer_tile_fits(residual_out_flat, config.rows_per_block, m, n)
            )
        )

    device_index = x.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    # Positional, not keyword: matching sixteen keywords through lru_cache costs
    # ~0.4us per launch on gfx950, against a ~20us host path at small M.
    launcher = _compiled_forward(
        device_index,
        n,
        x_flat.dtype,
        output_dtype,
        weight_arg.dtype,
        bias_arg.dtype,
        residual_arg.dtype,
        residual_out_dtype,
        weight is not None,
        bias is not None,
        residual is not None,
        store_residual,
        store_rstd,
        num_heads,
        weight_offset != 0.0,
        use_buffer_copy,
    )
    args = (
        x_flat,
        weight_arg,
        bias_arg,
        residual_arg,
        out_flat,
        residual_out_flat,
        rstd_flat,
        m,
        eps,
        weight_offset,
        current_raw_stream(x.device),
    )
    run_compiled(launcher, x.device, args, supported=_SUPPORTED_ARCHES, kernel="RMSNorm")

    return (
        out_flat.reshape(x.shape),
        residual_out_flat.reshape(x.shape) if store_residual else None,
        rstd_flat.reshape(x.shape[:-1]) if store_rstd else None,
    )


def _rmsnorm_fwd_impl(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    residual: torch.Tensor | None,
    out_dtype: torch.dtype | None,
    residual_dtype: torch.dtype | None,
    eps: float,
    store_rstd: bool,
    weight_offset: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Functional, not mutating: inductor skips cudagraphs for a region whose op
    writes into buffers it did not allocate."""
    out, residual_out, rstd = _rmsnorm_fwd_core(
        x, weight, bias, residual, out_dtype, residual_dtype, eps, store_rstd, weight_offset
    )
    _, residual_out_dtype, _ = _output_dtypes(x, residual, out_dtype, residual_dtype)
    return (
        out,
        residual_out if residual_out is not None else _absent(x, residual_out_dtype),
        rstd if rstd is not None else _absent(x, torch.float32),
    )


_rmsnorm_fwd_op = torch.library.custom_op(
    "quack::flydsl_rmsnorm_fwd",
    _rmsnorm_fwd_impl,
    mutates_args=(),
    device_types="cuda",
)


@_rmsnorm_fwd_op.register_fake
def _(x, weight, bias, residual, out_dtype, residual_dtype, eps, store_rstd, weight_offset):
    # Shapes only: running the body here would pay a FlyDSL compile at trace
    # time, and would reject shape/dtype combinations the kernel means to.
    output_dtype, residual_out_dtype, store_residual = _output_dtypes(
        x, residual, out_dtype, residual_dtype
    )
    return (
        torch.empty(x.shape, device=x.device, dtype=output_dtype),
        torch.empty(x.shape, device=x.device, dtype=residual_out_dtype)
        if store_residual
        else _absent(x, residual_out_dtype),
        torch.empty(x.shape[:-1], device=x.device, dtype=torch.float32)
        if store_rstd
        else _absent(x, torch.float32),
    )


def rmsnorm_fwd(
    x: torch.Tensor,
    weight: torch.Tensor | None = None,
    bias: torch.Tensor | None = None,
    residual: torch.Tensor | None = None,
    out_dtype: torch.dtype | None = None,
    residual_dtype: torch.dtype | None = None,
    eps: float = 1e-6,
    store_rstd: bool = False,
    weight_offset: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Run eager RMSNorm over the last dimension and return CuTe-compatible outputs.

    Routes through the registered op only under Dynamo, for the opaque graph
    node. The CuTe aliasing -- residual_out is x when nothing was accumulated,
    rstd is None unless requested -- is applied here, since an op may return
    neither one of its inputs nor a None.
    """
    eps, weight_offset = _validate_scalars(eps, store_rstd, weight_offset)
    if torch.compiler.is_compiling():
        out, residual_out, rstd = _rmsnorm_fwd_op(
            x, weight, bias, residual, out_dtype, residual_dtype, eps, store_rstd, weight_offset
        )
        _, _, store_residual = _output_dtypes(x, residual, out_dtype, residual_dtype)
        if not store_residual:
            residual_out = None
        if not store_rstd:
            rstd = None
    else:
        out, residual_out, rstd = _rmsnorm_fwd_core(
            x, weight, bias, residual, out_dtype, residual_dtype, eps, store_rstd, weight_offset
        )
    return out, (x if residual_out is None else residual_out), rstd
