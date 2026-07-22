# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025 FlyDSL Project Contributors
# Copyright (c) 2026 Dao-AILab
"""Shared FlyDSL primitives for Quack-owned RMSNorm kernels."""

import flydsl.expr as fx
from flydsl.expr.vector import full

WARP_SIZE = 64


def dtype_to_elem_type(dtype_str: str):
    if dtype_str == "f32":
        return fx.Float32
    if dtype_str == "f16":
        return fx.Float16
    if dtype_str == "bf16":
        return fx.BFloat16
    raise ValueError(f"unsupported dtype {dtype_str!r}")


def torch_dtype_to_str(dtype) -> str:
    import torch

    if dtype == torch.float32:
        return "f32"
    if dtype == torch.float16:
        return "f16"
    if dtype == torch.bfloat16:
        return "bf16"
    raise ValueError(f"unsupported torch dtype {dtype}")


def make_reduction_storage(red_slots: int):
    @fx.struct
    class SharedStorage:
        values: fx.Array[fx.Float32, red_slots, 16]

    return SharedStorage


def load_scalar(copy_atom, elem_dtype, divided_tensor, index):
    registers = fx.make_rmem_tensor(1, elem_dtype)
    fx.copy_atom_call(copy_atom, fx.slice(divided_tensor, (None, index)), registers)
    return fx.memref_load_vec(registers)[0]


def store_scalar(copy_atom, elem_dtype, divided_tensor, index, value):
    registers = fx.make_rmem_tensor(1, elem_dtype)
    fx.memref_store_vec(full(1, elem_dtype(value), elem_dtype), registers)
    fx.copy_atom_call(copy_atom, registers, fx.slice(divided_tensor, (None, index)))


def atomic_add_f32(copy_atom, divided_tensor, index, value):
    """Atomically add one FP32 register value through FlyDSL's public atom API."""

    registers = fx.make_rmem_tensor(1, fx.Float32)
    fx.memref_store_vec(full(1, fx.Float32(value), fx.Float32), registers)
    fx.copy_atom_call(
        copy_atom,
        registers,
        fx.slice(divided_tensor, (None, index)),
    )
