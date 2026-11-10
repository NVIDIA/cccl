# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes

from .harness import (
    ADJACENT_DIFFERENCE_DECL,
    DEFAULT_LATE_CALLBACK_SUFFIX,
    DTYPE_KINDS,
    EXCHANGE_DECL,
    SHUFFLE_DECL,
    TILEIR_CONSTANT_ARGUMENT_DTYPE,
    TILEIR_CONSTANT_ARGUMENT_INT,
    TILEIR_CONSTANT_ARGUMENT_STRING,
    TILEIR_CONSTANT_PARAMETER_DTYPE,
    TILEIR_CONSTANT_PARAMETER_INT,
    TILEIR_CONSTANT_PARAMETER_STRING,
    TILEIR_DIM_CONCRETE,
    TILEIR_DTYPE_I32,
    TILEIR_LP_ACCESS_IN,
    TILEIR_LP_ACCESS_INOUT,
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


def _unary_tile_types(num_items: int, dtype: str, *, runtime_i32: bool = False):
    shape = (TileIRDim * 1)()
    shape[0].kind = TILEIR_DIM_CONCRETE
    shape[0].value = num_items
    value_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        shape,
        len(shape),
        TileIRDataType(DTYPE_KINDS[dtype]),
    )
    if runtime_i32:
        scalar_type = TileIRForeignType(
            TILEIR_TYPE_TILE,
            None,
            0,
            TileIRDataType(TILEIR_DTYPE_I32),
        )
        arguments = (TileIRForeignType * 3)(value_type, scalar_type, value_type)
    else:
        arguments = (TileIRForeignType * 2)(value_type, value_type)
    results = (TileIRForeignType * 1)(value_type)
    return shape, arguments, results


def _mode_const_arguments(
    num_items: int,
    dtype: str,
    block_threads: int,
    items_per_thread: int,
    mode: str,
    dtype_token: str | None,
    *,
    resolved: bool,
    exchange: bool = False,
):
    values = (num_items, dtype)
    if exchange:
        values += (block_threads,)
    values += (mode, dtype_token or dtype)
    value_type = (
        TileIRConstantArgument if resolved else TileIRSymbolicExpression
    )
    constants = (value_type * len(values))()
    for index, value in enumerate(values):
        if index == 1:
            constants[index].kind = (
                TILEIR_CONSTANT_ARGUMENT_DTYPE
                if resolved
                else TILEIR_SYMBOLIC_EXPRESSION_DTYPE
            )
            constants[index].dtype_val = TileIRDataType(DTYPE_KINDS[dtype])
        elif isinstance(value, int):
            constants[index].kind = (
                TILEIR_CONSTANT_ARGUMENT_INT
                if resolved
                else TILEIR_SYMBOLIC_EXPRESSION_INT
            )
            constants[index].int_val = value
        else:
            constants[index].kind = (
                TILEIR_CONSTANT_ARGUMENT_STRING
                if resolved
                else TILEIR_SYMBOLIC_EXPRESSION_STRING
            )
            constants[index].string_val = _string_view(value)
    return constants


def _mode_decl(symbol: str, arguments, results, access_values):
    names_and_kinds = (
        ("N", TILEIR_CONSTANT_PARAMETER_INT),
        ("T", TILEIR_CONSTANT_PARAMETER_DTYPE),
        ("Mode", TILEIR_CONSTANT_PARAMETER_STRING),
        ("DTypeToken", TILEIR_CONSTANT_PARAMETER_STRING),
    )
    if symbol == EXCHANGE_DECL:
        names_and_kinds = (
            *names_and_kinds[:2],
            ("BlockThreads", TILEIR_CONSTANT_PARAMETER_INT),
            *names_and_kinds[2:],
        )
    const_params = (TileIRConstantParameter * len(names_and_kinds))(
        *(
            TileIRConstantParameter(_string_view(name), kind)
            for name, kind in names_and_kinds
        )
    )
    access = (ctypes.c_int * len(access_values))(*access_values)
    decl = TileIRForeignDecl(
        _string_view(symbol),
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


def _mode_semantic_key(
    symbol: str,
    dtype: str,
    block_threads: int,
    items_per_thread: int,
    mode: str,
) -> str:
    return (
        f"coop::{symbol}|block_threads={block_threads};dtype={dtype};"
        f"items_per_thread={items_per_thread};mode={mode}"
    )


def _mode_impl_symbol(
    symbol: str,
    dtype: str,
    block_threads: int,
    items_per_thread: int,
    mode: str,
    *,
    architecture: str = "sm_80",
) -> str:
    identity = _mode_semantic_key(
        symbol, dtype, block_threads, items_per_thread, mode
    )
    digest = _fnv1a(_artifact_identity(identity, architecture))
    return f"{symbol}_impl__lpv4_{digest:x}"


def _adjacent_difference_semantic_key(
    dtype: str, block_threads: int, items_per_thread: int, mode: str
) -> str:
    return _mode_semantic_key(
        ADJACENT_DIFFERENCE_DECL, dtype, block_threads, items_per_thread, mode
    )


def _adjacent_difference_impl_symbol(
    dtype: str,
    block_threads: int,
    items_per_thread: int,
    mode: str,
    *,
    architecture: str = "sm_80",
) -> str:
    return _mode_impl_symbol(
        ADJACENT_DIFFERENCE_DECL,
        dtype,
        block_threads,
        items_per_thread,
        mode,
        architecture=architecture,
    )


def _adjacent_difference_callsite(
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
    shape, arguments, results = _unary_tile_types(num_items, dtype)
    decl, const_params, access = _mode_decl(
        ADJACENT_DIFFERENCE_DECL,
        arguments,
        results,
        (TILEIR_LP_ACCESS_IN, TILEIR_LP_ACCESS_OUT),
    )
    const_arguments = _mode_const_arguments(
        num_items,
        dtype,
        block_threads,
        items_per_thread,
        mode,
        dtype_token,
        resolved=False,
    )
    callsite = TileIRLPCallsite(
        61,
        67,
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


def _resolved_adjacent_difference_call(
    mode: str,
    num_items: int = 256,
    dtype: str = "i32",
    block_threads: int = 128,
    items_per_thread: int | None = None,
    resolved_block_threads: int | None = None,
    dtype_token: str | None = None,
    impl_symbol: str | None = None,
    architecture: str = "sm_80",
    concrete_suffix: str = DEFAULT_LATE_CALLBACK_SUFFIX,
):
    if items_per_thread is None:
        items_per_thread = (
            0 if block_threads == 0 else num_items // block_threads
        )
    callsite, keepalive = _adjacent_difference_callsite(
        mode, num_items, dtype, block_threads, items_per_thread, dtype_token
    )
    shape, arguments, results, decl, const_params, access, _ = keepalive
    const_arguments = _mode_const_arguments(
        num_items,
        dtype,
        block_threads,
        items_per_thread,
        mode,
        dtype_token,
        resolved=True,
    )
    resolved_threads = (
        block_threads
        if resolved_block_threads is None
        else resolved_block_threads
    )
    resolved_items = (
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
            or _adjacent_difference_impl_symbol(
                dtype_token or dtype,
                resolved_threads,
                resolved_items,
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
        callsite,
    )


def _shuffle_semantic_key(
    dtype: str,
    block_threads: int,
    items_per_thread: int,
    mode: str,
    *,
    distance: int | None = None,
) -> str:
    del distance
    return _mode_semantic_key(
        SHUFFLE_DECL, dtype, block_threads, items_per_thread, mode
    )


def _shuffle_impl_symbol(
    dtype: str,
    block_threads: int,
    items_per_thread: int,
    mode: str,
    distance: int = 1,
    *,
    architecture: str = "sm_80",
) -> str:
    del distance
    return _mode_impl_symbol(
        SHUFFLE_DECL,
        dtype,
        block_threads,
        items_per_thread,
        mode,
        architecture=architecture,
    )


def _shuffle_callsite(
    mode: str,
    num_items: int = 256,
    dtype: str = "i32",
    block_threads: int = 128,
    items_per_thread: int | None = None,
    distance: int = 1,
    dtype_token: str | None = None,
):
    if items_per_thread is None:
        items_per_thread = (
            0 if block_threads == 0 else num_items // block_threads
        )
    shape, arguments, results = _unary_tile_types(
        num_items, dtype, runtime_i32=True
    )
    decl, const_params, access = _mode_decl(
        SHUFFLE_DECL,
        arguments,
        results,
        (TILEIR_LP_ACCESS_IN, TILEIR_LP_ACCESS_IN, TILEIR_LP_ACCESS_OUT),
    )
    const_arguments = _mode_const_arguments(
        num_items,
        dtype,
        block_threads,
        items_per_thread,
        mode,
        dtype_token,
        resolved=False,
    )
    callsite = TileIRLPCallsite(
        71,
        73,
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
        distance,
    )


def _resolved_shuffle_call(
    mode: str,
    num_items: int = 256,
    dtype: str = "i32",
    block_threads: int = 128,
    items_per_thread: int | None = None,
    distance: int = 1,
    resolved_block_threads: int | None = None,
    dtype_token: str | None = None,
    impl_symbol: str | None = None,
    architecture: str = "sm_80",
    concrete_suffix: str = DEFAULT_LATE_CALLBACK_SUFFIX,
):
    if items_per_thread is None:
        items_per_thread = (
            0 if block_threads == 0 else num_items // block_threads
        )
    callsite, keepalive = _shuffle_callsite(
        mode,
        num_items,
        dtype,
        block_threads,
        items_per_thread,
        distance,
        dtype_token,
    )
    shape, arguments, results, decl, const_params, access, _, _ = keepalive
    const_arguments = _mode_const_arguments(
        num_items,
        dtype,
        block_threads,
        items_per_thread,
        mode,
        dtype_token,
        resolved=True,
    )
    resolved_threads = resolved_block_threads or block_threads
    resolved_items = (
        num_items // resolved_threads
        if block_threads == 0 and items_per_thread == 0
        else items_per_thread
    )
    context = _execution_context(architecture)
    context.num_warps_per_cta = resolved_threads // 32
    impl = TileIRForeignImpl(
        _string_view(
            impl_symbol
            or _shuffle_impl_symbol(
                dtype_token or dtype,
                resolved_threads,
                resolved_items,
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
        callsite,
        distance,
    )


def _exchange_semantic_key(
    dtype: str, block_threads: int, items_per_thread: int, mode: str
) -> str:
    return _mode_semantic_key(
        EXCHANGE_DECL, dtype, block_threads, items_per_thread, mode
    )


def _exchange_impl_symbol(
    dtype: str,
    block_threads: int,
    items_per_thread: int,
    mode: str,
    *,
    architecture: str = "sm_80",
) -> str:
    return _mode_impl_symbol(
        EXCHANGE_DECL,
        dtype,
        block_threads,
        items_per_thread,
        mode,
        architecture=architecture,
    )


def _exchange_callsite(
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
    shape = (TileIRDim * 1)()
    shape[0].kind = TILEIR_DIM_CONCRETE
    shape[0].value = num_items
    value_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        shape,
        len(shape),
        TileIRDataType(DTYPE_KINDS[dtype]),
    )
    arguments = (TileIRForeignType * 1)(value_type)
    results = (TileIRForeignType * 1)(value_type)
    decl, const_params, access = _mode_decl(
        EXCHANGE_DECL, arguments, results, (TILEIR_LP_ACCESS_INOUT,)
    )
    const_arguments = _mode_const_arguments(
        num_items,
        dtype,
        block_threads,
        items_per_thread,
        mode,
        dtype_token,
        resolved=False,
        exchange=True,
    )
    callsite = TileIRLPCallsite(
        47,
        53,
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


def _resolved_exchange_call(
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
    callsite, keepalive = _exchange_callsite(
        mode, num_items, dtype, block_threads, items_per_thread, dtype_token
    )
    shape, arguments, results, decl, const_params, access, _ = keepalive
    const_arguments = _mode_const_arguments(
        num_items,
        dtype,
        block_threads,
        items_per_thread,
        mode,
        dtype_token,
        resolved=True,
        exchange=True,
    )
    resolved_threads = resolved_block_threads or block_threads
    resolved_items = (
        num_items // resolved_threads
        if block_threads == 0 and items_per_thread == 0
        else items_per_thread
    )
    context = _execution_context(architecture)
    context.num_warps_per_cta = resolved_threads // 32
    impl = TileIRForeignImpl(
        _string_view(
            impl_symbol
            or _exchange_impl_symbol(
                dtype_token or dtype,
                resolved_threads,
                resolved_items,
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
        callsite,
    )
