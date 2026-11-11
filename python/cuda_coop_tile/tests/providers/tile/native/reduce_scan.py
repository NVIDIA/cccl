# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes

from .harness import (
    DEFAULT_LATE_CALLBACK_SUFFIX,
    DTYPE_KINDS,
    REDUCE_DECLS,
    SCAN_DECL,
    TILEIR_CONSTANT_ARGUMENT_DTYPE,
    TILEIR_CONSTANT_ARGUMENT_INT,
    TILEIR_CONSTANT_ARGUMENT_STRING,
    TILEIR_CONSTANT_PARAMETER_DTYPE,
    TILEIR_CONSTANT_PARAMETER_INT,
    TILEIR_CONSTANT_PARAMETER_STRING,
    TILEIR_DIM_CONCRETE,
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


def _fill_constants(values, dtype: str, *, resolved: bool):
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


def _reduce_const_values(
    family: str,
    items_per_thread: int,
    dtype: str,
    algorithm: str = "raking",
    binary_op: str = "max",
    *,
    block_threads: int = 128,
):
    values = (
        block_threads * items_per_thread,
        dtype,
        algorithm,
    )
    if family == "reduce":
        values = (*values, binary_op)
    return (*values, dtype)


def _reduce_semantic_key(
    family: str,
    items_per_thread: int,
    dtype: str,
    algorithm: str = "raking",
    binary_op: str = "max",
    *,
    block_threads: int = 128,
) -> str:
    primitive_meta = f"algorithm={algorithm};"
    if family == "reduce":
        primitive_meta += f"binary_op={binary_op};"
    primitive_meta += (
        f"block_threads={block_threads};dtype={dtype};"
        f"items_per_thread={items_per_thread}"
    )
    return f"coop::{REDUCE_DECLS[family]}|{primitive_meta}"


def _reduce_impl_symbol(
    family: str,
    items_per_thread: int,
    dtype: str,
    algorithm: str = "raking",
    binary_op: str = "max",
    *,
    block_threads: int = 128,
    architecture: str = "sm_80",
) -> str:
    identity = _reduce_semantic_key(
        family,
        items_per_thread,
        dtype,
        algorithm,
        binary_op,
        block_threads=block_threads,
    )
    return (
        f"{REDUCE_DECLS[family]}_impl__lpv4_"
        f"{_fnv1a(_artifact_identity(identity, architecture)):x}"
    )


def _reduce_call_parts(
    family: str,
    items_per_thread: int,
    dtype: str,
    algorithm: str,
    binary_op: str,
    *,
    block_threads: int,
    resolved: bool,
):
    shape = (TileIRDim * 1)()
    shape[0].kind = TILEIR_DIM_CONCRETE
    shape[0].value = block_threads * items_per_thread
    tile_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        shape,
        len(shape),
        TileIRDataType(DTYPE_KINDS[dtype]),
    )
    arguments = (TileIRForeignType * 2)(tile_type, tile_type)
    results = (TileIRForeignType * 1)(tile_type)
    names_and_kinds = [
        ("N", TILEIR_CONSTANT_PARAMETER_INT),
        ("T", TILEIR_CONSTANT_PARAMETER_DTYPE),
        ("Algorithm", TILEIR_CONSTANT_PARAMETER_STRING),
    ]
    if family == "reduce":
        names_and_kinds.append(("BinaryOp", TILEIR_CONSTANT_PARAMETER_STRING))
    names_and_kinds.append(("DTypeToken", TILEIR_CONSTANT_PARAMETER_STRING))
    const_params = (TileIRConstantParameter * len(names_and_kinds))(
        *(
            TileIRConstantParameter(_string_view(name), kind)
            for name, kind in names_and_kinds
        )
    )
    access = (ctypes.c_int * 2)(TILEIR_LP_ACCESS_IN, TILEIR_LP_ACCESS_OUT)
    decl = TileIRForeignDecl(
        _string_view(REDUCE_DECLS[family]),
        const_params,
        len(const_params),
        arguments,
        len(arguments),
        results,
        len(results),
        access,
        len(access),
    )
    values = _reduce_const_values(
        family,
        items_per_thread,
        dtype,
        algorithm,
        binary_op,
        block_threads=block_threads,
    )
    const_arguments = _fill_constants(values, dtype, resolved=resolved)
    return (
        shape,
        arguments,
        results,
        decl,
        const_params,
        access,
        const_arguments,
    )


def _reduce_callsite(
    family: str,
    items_per_thread: int,
    dtype: str,
    algorithm: str = "raking",
    binary_op: str = "max",
    *,
    block_threads: int = 128,
):
    parts = _reduce_call_parts(
        family,
        items_per_thread,
        dtype,
        algorithm,
        binary_op,
        block_threads=block_threads,
        resolved=False,
    )
    _, arguments, results, decl, _, _, const_arguments = parts
    return (
        TileIRLPCallsite(
            19,
            23,
            ctypes.pointer(decl),
            const_arguments,
            len(const_arguments),
            arguments,
            len(arguments),
            results,
            len(results),
        ),
        parts,
    )


def _batched_reduce_callsite(
    family: str,
    extent: int,
    batches: int,
    dtype: str,
    algorithm: str = "warp_reductions",
    binary_op: str = "max",
):
    callsite, parts = _reduce_callsite(
        family, 1, dtype, algorithm, binary_op, block_threads=extent
    )
    _, arguments, results, decl, original_params, _, original_constants = parts
    output_shape = (TileIRDim * 1)()
    output_shape[0].kind = TILEIR_DIM_CONCRETE
    output_shape[0].value = batches
    output_type = TileIRForeignType(
        TILEIR_TYPE_TILE, output_shape, 1, TileIRDataType(DTYPE_KINDS[dtype])
    )
    arguments[1] = output_type
    results[0] = output_type
    params = (TileIRConstantParameter * (len(original_params) + 1))(
        *original_params,
        TileIRConstantParameter(
            _string_view("Batches"), TILEIR_CONSTANT_PARAMETER_INT
        ),
    )
    constants = (TileIRSymbolicExpression * (len(original_constants) + 1))()
    for index, value in enumerate(original_constants):
        constants[index] = value
    constants[-1].kind = TILEIR_SYMBOLIC_EXPRESSION_INT
    constants[-1].int_val = batches
    decl.symbol = _string_view(REDUCE_DECLS[family] + "_batched")
    decl.const_params = params
    decl.num_const_params = len(params)
    callsite.const_arguments = constants
    callsite.num_const_arguments = len(constants)
    return callsite, (parts, output_shape, params, constants)


def _resolved_reduce_call(
    family: str,
    items_per_thread: int,
    dtype: str,
    algorithm: str = "raking",
    binary_op: str = "max",
    impl_symbol: str | None = None,
    architecture: str = "sm_80",
    concrete_suffix: str = DEFAULT_LATE_CALLBACK_SUFFIX,
    *,
    block_threads: int = 128,
):
    parts = _reduce_call_parts(
        family,
        items_per_thread,
        dtype,
        algorithm,
        binary_op,
        block_threads=block_threads,
        resolved=True,
    )
    _, arguments, _, decl, _, _, const_arguments = parts
    context = _execution_context(architecture)
    impl = TileIRForeignImpl(
        _string_view(
            impl_symbol
            or _reduce_impl_symbol(
                family,
                items_per_thread,
                dtype,
                algorithm,
                binary_op,
                block_threads=block_threads,
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
    return call, (*parts, context, impl)


def _scan_const_values(
    items_per_thread: int,
    dtype: str,
    *,
    block_threads: int = 128,
    mode: str = "exclusive",
    algorithm: str = "raking",
    scan_op: str = "sum",
    initial_present: int = 0,
    prefix_present: int = 0,
):
    return (
        block_threads * items_per_thread,
        dtype,
        mode,
        algorithm,
        scan_op,
        initial_present,
        prefix_present,
        dtype,
    )


def _scan_semantic_key(
    items_per_thread: int,
    dtype: str,
    *,
    block_threads: int = 128,
    mode: str = "exclusive",
    algorithm: str = "raking",
    scan_op: str = "sum",
    initial_present: int = 0,
    prefix_present: int = 0,
) -> str:
    primitive_meta = (
        f"algorithm={algorithm};block_threads={block_threads};dtype={dtype};"
        f"initial_present={initial_present};prefix_present={prefix_present};"
        f"items_per_thread={items_per_thread};mode={mode};scan_op={scan_op}"
    )
    return f"coop::{SCAN_DECL}|{primitive_meta}"


def _scan_impl_symbol(
    items_per_thread: int,
    dtype: str,
    *,
    architecture: str = "sm_80",
    **scan_options,
) -> str:
    identity = _scan_semantic_key(items_per_thread, dtype, **scan_options)
    artifact_hash = _fnv1a(_artifact_identity(identity, architecture))
    return f"{SCAN_DECL}_impl__lpv4_{artifact_hash:x}"


def _scan_call_parts(
    items_per_thread: int,
    dtype: str,
    *,
    resolved: bool,
    block_threads: int = 128,
    **scan_options,
):
    shape = (TileIRDim * 1)()
    shape[0].kind = TILEIR_DIM_CONCRETE
    shape[0].value = block_threads * items_per_thread
    tile_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        shape,
        len(shape),
        TileIRDataType(DTYPE_KINDS[dtype]),
    )
    scalar_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        None,
        0,
        TileIRDataType(DTYPE_KINDS[dtype]),
    )
    arguments = (TileIRForeignType * 4)(
        tile_type,
        scalar_type,
        scalar_type,
        tile_type,
    )
    results = (TileIRForeignType * 1)(tile_type)
    names_and_kinds = (
        ("N", TILEIR_CONSTANT_PARAMETER_INT),
        ("T", TILEIR_CONSTANT_PARAMETER_DTYPE),
        ("Mode", TILEIR_CONSTANT_PARAMETER_STRING),
        ("Algorithm", TILEIR_CONSTANT_PARAMETER_STRING),
        ("ScanOp", TILEIR_CONSTANT_PARAMETER_STRING),
        ("InitialPresent", TILEIR_CONSTANT_PARAMETER_INT),
        ("PrefixPresent", TILEIR_CONSTANT_PARAMETER_INT),
        ("DTypeToken", TILEIR_CONSTANT_PARAMETER_STRING),
    )
    const_params = (TileIRConstantParameter * len(names_and_kinds))(
        *(
            TileIRConstantParameter(_string_view(name), kind)
            for name, kind in names_and_kinds
        )
    )
    access = (ctypes.c_int * 4)(
        TILEIR_LP_ACCESS_IN,
        TILEIR_LP_ACCESS_IN,
        TILEIR_LP_ACCESS_IN,
        TILEIR_LP_ACCESS_OUT,
    )
    decl = TileIRForeignDecl(
        _string_view(SCAN_DECL),
        const_params,
        len(const_params),
        arguments,
        len(arguments),
        results,
        len(results),
        access,
        len(access),
    )
    values = _scan_const_values(
        items_per_thread,
        dtype,
        block_threads=block_threads,
        **scan_options,
    )
    const_arguments = _fill_constants(values, dtype, resolved=resolved)
    return (
        shape,
        arguments,
        results,
        decl,
        const_params,
        access,
        const_arguments,
    )


def _scan_callsite(
    items_per_thread: int,
    dtype: str,
    *,
    block_threads: int = 128,
    **scan_options,
):
    parts = _scan_call_parts(
        items_per_thread,
        dtype,
        resolved=False,
        block_threads=block_threads,
        **scan_options,
    )
    _, arguments, results, decl, _, _, const_arguments = parts
    return (
        TileIRLPCallsite(
            7,
            11,
            ctypes.pointer(decl),
            const_arguments,
            len(const_arguments),
            arguments,
            len(arguments),
            results,
            len(results),
        ),
        parts,
    )


def _resolved_scan_call(
    items_per_thread: int,
    dtype: str,
    impl_symbol: str | None = None,
    architecture: str = "sm_80",
    concrete_suffix: str = DEFAULT_LATE_CALLBACK_SUFFIX,
    *,
    block_threads: int = 128,
    **scan_options,
):
    parts = _scan_call_parts(
        items_per_thread,
        dtype,
        resolved=True,
        block_threads=block_threads,
        **scan_options,
    )
    _, arguments, _, decl, _, _, const_arguments = parts
    context = _execution_context(architecture)
    impl = TileIRForeignImpl(
        _string_view(
            impl_symbol
            or _scan_impl_symbol(
                items_per_thread,
                dtype,
                block_threads=block_threads,
                architecture=architecture,
                **scan_options,
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
    return call, (*parts, context, impl)
