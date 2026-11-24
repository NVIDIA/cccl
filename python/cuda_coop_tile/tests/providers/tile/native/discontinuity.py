# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes

from .harness import (
    DEFAULT_LATE_CALLBACK_SUFFIX,
    DISCONTINUITY_DECLS,
    DTYPE_KINDS,
    TILEIR_CONSTANT_ARGUMENT_DTYPE,
    TILEIR_CONSTANT_ARGUMENT_INT,
    TILEIR_CONSTANT_ARGUMENT_STRING,
    TILEIR_CONSTANT_PARAMETER_DTYPE,
    TILEIR_CONSTANT_PARAMETER_INT,
    TILEIR_CONSTANT_PARAMETER_STRING,
    TILEIR_DIM_CONCRETE,
    TILEIR_DTYPE_I32,
    TILEIR_LP_ACCESS_IN,
    TILEIR_LP_ACCESS_OUT,
    TILEIR_SYMBOLIC_EXPRESSION_DTYPE,
    TILEIR_SYMBOLIC_EXPRESSION_INT,
    TILEIR_SYMBOLIC_EXPRESSION_STRING,
    TILEIR_TYPE_TILE,
    TileIRConstantArgument,
    TileIRConstantParameter,
    TileIRDataType,
    TileIRDim,
    TileIRForeignDecl,
    TileIRForeignImpl,
    TileIRForeignType,
    TileIRLPCallsite,
    TileIRResolvedCall,
    TileIRSymbolicExpression,
    _artifact_identity,
    _concretize_resolved_call_for_late_callback,
    _execution_context,
    _fnv1a,
    _string_view,
)


def _discontinuity_const_values(
    num_items: int,
    dtype: str,
    block_threads: int,
    items_per_thread: int,
    mode: str,
    dtype_token: str | None = None,
):
    values = (num_items, dtype, mode)
    return (*values, dtype_token or dtype)


def _discontinuity_semantic_key(
    family: str,
    dtype: str,
    block_threads: int,
    items_per_thread: int,
    mode: str,
) -> str:
    primitive_meta = (
        f"block_threads={block_threads};dtype={dtype};"
        f"items_per_thread={items_per_thread};mode={mode}"
    )
    return f"coop::{DISCONTINUITY_DECLS[family]}|{primitive_meta}"


def _discontinuity_impl_symbol(
    family: str,
    dtype: str,
    block_threads: int,
    items_per_thread: int,
    mode: str,
    *,
    architecture: str = "sm_80",
) -> str:
    semantic_key = _discontinuity_semantic_key(
        family, dtype, block_threads, items_per_thread, mode
    )
    prefix = f"{DISCONTINUITY_DECLS[family]}_impl"
    digest = _fnv1a(_artifact_identity(semantic_key, architecture))
    return f"{prefix}__lpv4_{digest:x}"


def _discontinuity_types(
    family: str,
    num_items: int,
    dtype: str,
):
    shape = (TileIRDim * 1)()
    shape[0].kind = TILEIR_DIM_CONCRETE
    shape[0].value = num_items
    input_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        shape,
        len(shape),
        TileIRDataType(DTYPE_KINDS[dtype]),
    )
    output_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        shape,
        len(shape),
        TileIRDataType(TILEIR_DTYPE_I32),
    )
    if family == "flags":
        arguments = (TileIRForeignType * 2)(input_type, output_type)
        results = (TileIRForeignType * 1)(output_type)
    else:
        assert family == "heads_and_tails"
        arguments = (TileIRForeignType * 3)(
            input_type,
            output_type,
            output_type,
        )
        results = (TileIRForeignType * 2)(output_type, output_type)
    return shape, arguments, results


def _discontinuity_decl(
    arguments,
    results,
    family: str,
    num_items: int,
    dtype: str,
    block_threads: int,
    items_per_thread: int,
    mode: str,
):
    del num_items, dtype, block_threads, items_per_thread, mode
    names_and_kinds = [
        ("N", TILEIR_CONSTANT_PARAMETER_INT),
        ("T", TILEIR_CONSTANT_PARAMETER_DTYPE),
        ("Mode", TILEIR_CONSTANT_PARAMETER_STRING),
    ]
    names_and_kinds.append(("DTypeToken", TILEIR_CONSTANT_PARAMETER_STRING))
    const_params = (TileIRConstantParameter * len(names_and_kinds))(
        *(
            TileIRConstantParameter(_string_view(name), kind)
            for name, kind in names_and_kinds
        )
    )
    access_values = (
        (TILEIR_LP_ACCESS_IN, TILEIR_LP_ACCESS_OUT)
        if family == "flags"
        else (
            TILEIR_LP_ACCESS_IN,
            TILEIR_LP_ACCESS_OUT,
            TILEIR_LP_ACCESS_OUT,
        )
    )
    access = (ctypes.c_int * len(access_values))(*access_values)
    decl = TileIRForeignDecl(
        _string_view(DISCONTINUITY_DECLS[family]),
        const_params,
        len(const_params),
        arguments,
        len(arguments),
        results,
        len(results),
        access,
        len(access),
    )
    return decl, const_params, access


def _symbolic_discontinuity_arguments(
    num_items: int,
    dtype: str,
    block_threads: int,
    items_per_thread: int,
    mode: str,
    dtype_token: str | None = None,
):
    const_values = _discontinuity_const_values(
        num_items,
        dtype,
        block_threads,
        items_per_thread,
        mode,
        dtype_token,
    )
    values = (TileIRSymbolicExpression * len(const_values))()
    for index, value in enumerate(const_values):
        if index == 1:
            values[index].kind = TILEIR_SYMBOLIC_EXPRESSION_DTYPE
            values[index].dtype_val = TileIRDataType(DTYPE_KINDS[dtype])
        elif isinstance(value, int):
            values[index].kind = TILEIR_SYMBOLIC_EXPRESSION_INT
            values[index].int_val = value
        else:
            values[index].kind = TILEIR_SYMBOLIC_EXPRESSION_STRING
            values[index].string_val = _string_view(value)
    return values


def _concrete_discontinuity_arguments(
    num_items: int,
    dtype: str,
    block_threads: int,
    items_per_thread: int,
    mode: str,
    dtype_token: str | None = None,
):
    const_values = _discontinuity_const_values(
        num_items,
        dtype,
        block_threads,
        items_per_thread,
        mode,
        dtype_token,
    )
    values = (TileIRConstantArgument * len(const_values))()
    for index, value in enumerate(const_values):
        if index == 1:
            values[index].kind = TILEIR_CONSTANT_ARGUMENT_DTYPE
            values[index].dtype_val = TileIRDataType(DTYPE_KINDS[dtype])
        elif isinstance(value, int):
            values[index].kind = TILEIR_CONSTANT_ARGUMENT_INT
            values[index].int_val = value
        else:
            values[index].kind = TILEIR_CONSTANT_ARGUMENT_STRING
            values[index].string_val = _string_view(value)
    return values


def _discontinuity_callsite(
    family: str,
    mode: str,
    num_items: int = 256,
    dtype: str = "i32",
    block_threads: int = 128,
    items_per_thread: int | None = None,
    dtype_token: str | None = None,
):
    if items_per_thread is None:
        items_per_thread = (
            0 if block_threads == 0 else num_items // block_threads
        )
    shape, arguments, results = _discontinuity_types(family, num_items, dtype)
    decl, const_params, access = _discontinuity_decl(
        arguments,
        results,
        family,
        num_items,
        dtype,
        block_threads,
        items_per_thread,
        mode,
    )
    const_arguments = _symbolic_discontinuity_arguments(
        num_items,
        dtype,
        block_threads,
        items_per_thread,
        mode,
        dtype_token,
    )
    callsite = TileIRLPCallsite(
        41,
        43,
        ctypes.pointer(decl),
        const_arguments,
        len(const_arguments),
        arguments,
        len(arguments),
        results,
        len(results),
    )
    return callsite, (
        shape,
        arguments,
        results,
        decl,
        const_params,
        access,
        const_arguments,
    )


def _resolved_discontinuity_call(
    family: str,
    mode: str,
    num_items: int = 256,
    dtype: str = "i32",
    block_threads: int = 128,
    items_per_thread: int | None = None,
    resolved_block_threads: int | None = None,
    impl_symbol: str | None = None,
    architecture: str = "sm_80",
    dtype_token: str | None = None,
    concrete_suffix: str = DEFAULT_LATE_CALLBACK_SUFFIX,
):
    if items_per_thread is None:
        items_per_thread = (
            0 if block_threads == 0 else num_items // block_threads
        )
    shape, arguments, results = _discontinuity_types(family, num_items, dtype)
    decl, const_params, access = _discontinuity_decl(
        arguments,
        results,
        family,
        num_items,
        dtype,
        block_threads,
        items_per_thread,
        mode,
    )
    const_arguments = _concrete_discontinuity_arguments(
        num_items,
        dtype,
        block_threads,
        items_per_thread,
        mode,
        dtype_token,
    )
    resolved_threads = (
        block_threads
        if resolved_block_threads is None
        else resolved_block_threads
    )
    resolved_items_per_thread = (
        0
        if resolved_threads <= 0
        else num_items // resolved_threads
        if block_threads == 0 and items_per_thread == 0
        else items_per_thread
    )
    context = _execution_context(architecture)
    context.num_warps_per_cta = resolved_threads // 32
    impl = TileIRForeignImpl(
        _string_view(
            impl_symbol
            or _discontinuity_impl_symbol(
                family,
                dtype_token or dtype,
                resolved_threads,
                resolved_items_per_thread,
                mode,
                architecture=architecture,
            )
        ),
        decl.symbol,
        context,
    )
    call = TileIRResolvedCall(
        ctypes.pointer(decl),
        ctypes.pointer(impl),
        context,
        const_arguments,
        len(const_arguments),
        arguments,
        len(arguments),
    )
    _concretize_resolved_call_for_late_callback(call, concrete_suffix)
    return call, (
        shape,
        arguments,
        results,
        decl,
        const_params,
        access,
        const_arguments,
        context,
        impl,
    )
