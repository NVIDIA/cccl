# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Simplified Mamba selective scan forward kernel (Numba + cuda.coop).

This is an incremental port of the CUDA C++ selective scan forward kernel,
initially covering the non-complex, fixed B/C, single-chunk case.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numba
import numpy as np
from numba import cuda, types
from numba.core import cgutils
from numba.core.extending import (
    lower_builtin,
    make_attribute_wrapper,
    models,
    register_model,
    type_callable,
    typeof_impl,
)
from numba.core.imputils import lower_constant

import cuda.coop as coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

DEFAULT_ITEMS_PER_THREAD = 4


class Float2:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def construct(this):
        zero = numba.float32(0.0)
        this[0] = Float2(zero, zero)

    def assign(this, that):
        this[0] = Float2(that[0].x, that[0].y)


class Float2Type(types.Type):
    def __init__(self):
        super().__init__(name="Float2")


float2_type = Float2Type()
float2_type.methods = {
    "construct": Float2.construct,
    "assign": Float2.assign,
}


@typeof_impl.register(Float2)
def typeof_float2(val, c):
    return float2_type


@type_callable(Float2)
def type__float2(context):
    def typer(x, y):
        if isinstance(x, types.Float) and isinstance(y, types.Float):
            return float2_type

    return typer


@register_model(Float2Type)
class Float2Model(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("x", types.float32), ("y", types.float32)]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(Float2Type, "x", "x")
make_attribute_wrapper(Float2Type, "y", "y")


@lower_builtin(Float2, types.Float, types.Float)
def impl_float2(context, builder, sig, args):
    typ = sig.return_type
    x, y = args
    state = cgutils.create_struct_proxy(typ)(context, builder)
    state.x = x
    state.y = y
    return state._getvalue()


@lower_constant(Float2Type)
def lower_constant_float2(context, builder, typ, value):
    state = cgutils.create_struct_proxy(typ)(context, builder)
    state.x = context.get_constant(types.float32, value.x)
    state.y = context.get_constant(types.float32, value.y)
    return state._getvalue()


@cuda.jit(device=True)
def ssm_scan_op(result_ptr, lhs_ptr, rhs_ptr):
    a0 = lhs_ptr[0].x
    b0 = lhs_ptr[0].y
    a1 = rhs_ptr[0].x
    b1 = rhs_ptr[0].y
    result_ptr[0] = Float2(a1 * a0, a1 * b0 + b1)


class SSMScanPrefixCallbackOp:
    def __init__(self, running_prefix):
        self.running_prefix = running_prefix

    def __call__(self_ptr, block_aggregate, out_ptr):
        old_prefix = self_ptr[0].running_prefix
        block_val = block_aggregate[0]
        new_prefix = Float2(
            block_val.x * old_prefix.x,
            block_val.x * old_prefix.y + block_val.y,
        )
        self_ptr[0] = SSMScanPrefixCallbackOp(new_prefix)
        out_ptr[0] = old_prefix


class SSMScanPrefixCallbackOpType(types.Type):
    def __init__(self):
        super().__init__(name="SSMScanPrefixCallbackOp")


ssm_prefix_callback_op_type = SSMScanPrefixCallbackOpType()


@typeof_impl.register(SSMScanPrefixCallbackOp)
def typeof_ssm_prefix_callback_op(val, c):
    return ssm_prefix_callback_op_type


@type_callable(SSMScanPrefixCallbackOp)
def type__ssm_prefix_callback_op(context):
    def typer(running_prefix):
        if running_prefix == float2_type:
            return ssm_prefix_callback_op_type

    return typer


@register_model(SSMScanPrefixCallbackOpType)
class SSMScanPrefixCallbackOpModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [("running_prefix", float2_type)]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(SSMScanPrefixCallbackOpType, "running_prefix", "running_prefix")


@lower_builtin(SSMScanPrefixCallbackOp, Float2Type)
def impl_ssm_prefix_callback_op(context, builder, sig, args):
    typ = sig.return_type
    (running_prefix,) = args
    state = cgutils.create_struct_proxy(typ)(context, builder)
    state.running_prefix = running_prefix
    return state._getvalue()


@dataclass
class KernelTraits:
    items_per_thread: int
    block_load_u: coop.block.load
    block_load_delta: coop.block.load
    block_store: coop.block.store
    block_scan: coop.block.scan


def make_kernel_traits(dtype, threads_per_block, items_per_thread):
    block_load_u = coop.block.load(
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm=coop.BlockLoadAlgorithm.WARP_TRANSPOSE,
    )
    block_load_delta = coop.block.load(
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm=coop.BlockLoadAlgorithm.WARP_TRANSPOSE,
    )
    block_store = coop.block.store(
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm=coop.BlockStoreAlgorithm.WARP_TRANSPOSE,
    )
    block_scan = coop.block.scan(
        dtype=float2_type,
        threads_per_block=threads_per_block,
        items_per_thread=items_per_thread,
        mode="inclusive",
        scan_op=ssm_scan_op,
        block_prefix_callback_op=coop.StatefulFunction(
            SSMScanPrefixCallbackOp,
            ssm_prefix_callback_op_type,
            name="ssm_scan_prefix",
        ),
        algorithm=coop.BlockScanAlgorithm.WARP_SCANS,
        methods=float2_type.methods,
    )
    traits = KernelTraits(
        items_per_thread=items_per_thread,
        block_load_u=block_load_u,
        block_load_delta=block_load_delta,
        block_store=block_store,
        block_scan=block_scan,
    )
    return coop.gpu_dataclass(traits)


def make_selective_scan_fwd_kernel(traits):
    link_files = []

    def _build_kernel(decorator):
        @decorator
        def kernel(u, delta, out, A, B, C, D, delta_bias, traits):
            if traits.items_per_thread != DEFAULT_ITEMS_PER_THREAD:
                return
            thread_data = coop.local.array(DEFAULT_ITEMS_PER_THREAD, dtype=float2_type)
            u_vals = coop.local.array(DEFAULT_ITEMS_PER_THREAD, dtype=u.dtype)
            delta_vals = coop.local.array(DEFAULT_ITEMS_PER_THREAD, dtype=delta.dtype)

            traits.block_load_u(u, u_vals)
            traits.block_load_delta(delta, delta_vals)

            # Build scan inputs (a, b).
            for i in range(DEFAULT_ITEMS_PER_THREAD):
                u_val = u_vals[i]
                delta_val = numba.float32(delta_vals[i] + delta_bias)
                a_val = numba.float32(math.exp(delta_val * A))
                b_val = numba.float32(delta_val * u_val * B)
                thread_data[i] = Float2(a_val, b_val)

            prefix_op = coop.local.array(1, dtype=ssm_prefix_callback_op_type)
            prefix_op[0] = SSMScanPrefixCallbackOp(
                Float2(numba.float32(1.0), numba.float32(0.0))
            )

            traits.block_scan(
                thread_data,
                thread_data,
                block_prefix_callback_op=prefix_op,
            )

            out_vals = coop.local.array(DEFAULT_ITEMS_PER_THREAD, dtype=out.dtype)
            for i in range(DEFAULT_ITEMS_PER_THREAD):
                out_vals[i] = D * u_vals[i] + thread_data[i].y * C

            traits.block_store(out, out_vals)

        return kernel

    if link_files:
        return _build_kernel(cuda.jit(link=link_files))

    return _build_kernel(cuda.jit)


def selective_scan_fwd_reference(u, delta, A, B, C, D, delta_bias):
    out = np.empty_like(u, dtype=np.float32)
    a_run = 1.0
    b_run = 0.0
    for i in range(u.shape[0]):
        delta_val = float(delta[i] + delta_bias)
        a_val = math.exp(delta_val * float(A))
        b_val = delta_val * float(u[i]) * float(B)
        a_run = a_val * a_run
        b_run = a_val * b_run + b_val
        out[i] = float(D) * float(u[i]) + b_run * float(C)
    return out
