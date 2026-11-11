# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes

from .harness import (
    DEFAULT_LATE_CALLBACK_SUFFIX,
    DTYPE_KINDS,
    HISTOGRAM_DECLS,
    RUN_LENGTH_DECODE_DECLS,
    TILEIR_CONSTANT_ARGUMENT_DTYPE,
    TILEIR_CONSTANT_ARGUMENT_INT,
    TILEIR_CONSTANT_ARGUMENT_STRING,
    TILEIR_CONSTANT_PARAMETER_DTYPE,
    TILEIR_CONSTANT_PARAMETER_INT,
    TILEIR_CONSTANT_PARAMETER_STRING,
    TILEIR_DIM_CONCRETE,
    TILEIR_DTYPE_I8,
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


def _histogram_is_fixed(family: str) -> bool:
    assert family in HISTOGRAM_DECLS
    return family == "histogram_256"


def _histogram_const_values(
    family: str,
    num_items: int,
    bins: int,
    block_threads: int,
    items_per_thread: int,
    dtype_token: str = "u8",
):
    values = (
        (num_items,)
        if _histogram_is_fixed(family)
        else (
            num_items,
            bins,
        )
    )
    return (*values, dtype_token)


def _histogram_semantic_key(
    family: str,
    bins: int,
    block_threads: int,
    items_per_thread: int,
) -> str:
    primitive_meta = (
        f"bins={bins};block_threads={block_threads};dtype=u8;"
        f"items_per_thread={items_per_thread};output_dtype=i32"
    )
    return f"coop::cuda_coop_tile_{family}|{primitive_meta}"


def _histogram_impl_symbol(
    family: str,
    bins: int,
    block_threads: int,
    items_per_thread: int,
    *,
    architecture: str = "sm_80",
) -> str:
    semantic_key = _histogram_semantic_key(
        family, bins, block_threads, items_per_thread
    )
    prefix = f"{HISTOGRAM_DECLS[family]}_impl"
    digest = _fnv1a(_artifact_identity(semantic_key, architecture))
    return f"{prefix}__lpv4_{digest:x}"


def _histogram_types(num_items: int, bins: int):
    input_shape = (TileIRDim * 1)()
    input_shape[0].kind = TILEIR_DIM_CONCRETE
    input_shape[0].value = num_items
    output_shape = (TileIRDim * 1)()
    output_shape[0].kind = TILEIR_DIM_CONCRETE
    output_shape[0].value = bins
    input_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        input_shape,
        len(input_shape),
        TileIRDataType(TILEIR_DTYPE_I8),
    )
    output_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        output_shape,
        len(output_shape),
        TileIRDataType(TILEIR_DTYPE_I32),
    )
    arguments = (TileIRForeignType * 2)(input_type, output_type)
    results = (TileIRForeignType * 1)(output_type)
    return input_shape, output_shape, arguments, results


def _histogram_decl(
    arguments,
    results,
    family: str,
    num_items: int,
    bins: int,
    block_threads: int,
    items_per_thread: int,
):
    del num_items, bins, block_threads, items_per_thread
    names_and_kinds = [("N", TILEIR_CONSTANT_PARAMETER_INT)]
    if not _histogram_is_fixed(family):
        names_and_kinds.append(("Bins", TILEIR_CONSTANT_PARAMETER_INT))
    names_and_kinds.extend([])
    names_and_kinds.append(("DTypeToken", TILEIR_CONSTANT_PARAMETER_STRING))
    const_params = (TileIRConstantParameter * len(names_and_kinds))(
        *(
            TileIRConstantParameter(_string_view(name), kind)
            for name, kind in names_and_kinds
        )
    )
    access = (ctypes.c_int * 2)(TILEIR_LP_ACCESS_IN, TILEIR_LP_ACCESS_OUT)
    decl = TileIRForeignDecl(
        _string_view(HISTOGRAM_DECLS[family]),
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


def _symbolic_histogram_arguments(
    family: str,
    num_items: int,
    bins: int,
    block_threads: int,
    items_per_thread: int,
    dtype_token: str = "u8",
):
    const_values = _histogram_const_values(
        family,
        num_items,
        bins,
        block_threads,
        items_per_thread,
        dtype_token,
    )
    values = (TileIRSymbolicExpression * len(const_values))()
    for index, value in enumerate(const_values):
        if isinstance(value, int):
            values[index].kind = TILEIR_SYMBOLIC_EXPRESSION_INT
            values[index].int_val = value
        else:
            values[index].kind = TILEIR_SYMBOLIC_EXPRESSION_STRING
            values[index].string_val = _string_view(value)
    return values


def _concrete_histogram_arguments(
    family: str,
    num_items: int,
    bins: int,
    block_threads: int,
    items_per_thread: int,
    dtype_token: str = "u8",
):
    const_values = _histogram_const_values(
        family,
        num_items,
        bins,
        block_threads,
        items_per_thread,
        dtype_token,
    )
    values = (TileIRConstantArgument * len(const_values))()
    for index, value in enumerate(const_values):
        if isinstance(value, int):
            values[index].kind = TILEIR_CONSTANT_ARGUMENT_INT
            values[index].int_val = value
        else:
            values[index].kind = TILEIR_CONSTANT_ARGUMENT_STRING
            values[index].string_val = _string_view(value)
    return values


def _histogram_callsite(
    family: str,
    num_items: int = 512,
    bins: int = 256,
    block_threads: int = 128,
    items_per_thread: int | None = None,
    dtype_token: str = "u8",
):
    if items_per_thread is None:
        items_per_thread = (
            0 if block_threads == 0 else num_items // block_threads
        )
    input_shape, output_shape, arguments, results = _histogram_types(
        num_items, bins
    )
    decl, const_params, access = _histogram_decl(
        arguments,
        results,
        family,
        num_items,
        bins,
        block_threads,
        items_per_thread,
    )
    const_arguments = _symbolic_histogram_arguments(
        family,
        num_items,
        bins,
        block_threads,
        items_per_thread,
        dtype_token,
    )
    callsite = TileIRLPCallsite(
        31,
        37,
        ctypes.pointer(decl),
        const_arguments,
        len(const_arguments),
        arguments,
        len(arguments),
        results,
        len(results),
    )
    return callsite, (
        input_shape,
        output_shape,
        arguments,
        results,
        decl,
        const_params,
        access,
        const_arguments,
    )


def _resolved_histogram_call(
    family: str,
    num_items: int = 512,
    bins: int = 256,
    block_threads: int = 128,
    items_per_thread: int | None = None,
    resolved_block_threads: int | None = None,
    impl_symbol: str | None = None,
    architecture: str = "sm_80",
    dtype_token: str = "u8",
    concrete_suffix: str = DEFAULT_LATE_CALLBACK_SUFFIX,
):
    if items_per_thread is None:
        items_per_thread = (
            0 if block_threads == 0 else num_items // block_threads
        )
    input_shape, output_shape, arguments, results = _histogram_types(
        num_items, bins
    )
    decl, const_params, access = _histogram_decl(
        arguments,
        results,
        family,
        num_items,
        bins,
        block_threads,
        items_per_thread,
    )
    const_arguments = _concrete_histogram_arguments(
        family,
        num_items,
        bins,
        block_threads,
        items_per_thread,
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
            or _histogram_impl_symbol(
                family,
                bins,
                resolved_threads,
                resolved_items_per_thread,
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
        input_shape,
        output_shape,
        arguments,
        results,
        decl,
        const_params,
        access,
        const_arguments,
        context,
        impl,
    )


def _run_length_decode_semantic_key(
    family: str,
    block_threads: int,
    runs_per_thread: int,
    decoded_items_per_thread: int,
    value_dtype: str,
    length_dtype: str,
) -> str:
    suffix = "_with_offsets" if family == "items_and_offsets" else ""
    primitive_meta = (
        f"block_threads={block_threads};"
        f"decoded_items_per_thread={decoded_items_per_thread};"
        f"length_dtype={length_dtype};"
        f"runs_per_thread={runs_per_thread};"
        f"value_dtype={value_dtype}"
    )
    return f"coop::cuda_coop_tile_run_length_decode{suffix}|{primitive_meta}"


def _run_length_decode_impl_symbol(
    family: str,
    block_threads: int,
    runs_per_thread: int,
    decoded_items_per_thread: int,
    value_dtype: str,
    length_dtype: str,
    *,
    architecture: str = "sm_80",
) -> str:
    identity = _run_length_decode_semantic_key(
        family,
        block_threads,
        runs_per_thread,
        decoded_items_per_thread,
        value_dtype,
        length_dtype,
    )
    return (
        f"{RUN_LENGTH_DECODE_DECLS[family]}_impl__lpv4_"
        f"{_fnv1a(_artifact_identity(identity, architecture)):x}"
    )


def _run_length_decode_types(
    family: str,
    run_extent: int,
    decoded_extent: int,
    value_dtype: str,
    length_dtype: str,
):
    run_shape = (TileIRDim * 1)()
    run_shape[0].kind = TILEIR_DIM_CONCRETE
    run_shape[0].value = run_extent
    decoded_shape = (TileIRDim * 1)()
    decoded_shape[0].kind = TILEIR_DIM_CONCRETE
    decoded_shape[0].value = decoded_extent
    run_value_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        run_shape,
        len(run_shape),
        TileIRDataType(DTYPE_KINDS[value_dtype]),
    )
    run_length_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        run_shape,
        len(run_shape),
        TileIRDataType(DTYPE_KINDS[length_dtype]),
    )
    window_offset_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        None,
        0,
        TileIRDataType(DTYPE_KINDS[length_dtype]),
    )
    decoded_value_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        decoded_shape,
        len(decoded_shape),
        TileIRDataType(DTYPE_KINDS[value_dtype]),
    )
    decoded_offset_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        decoded_shape,
        len(decoded_shape),
        TileIRDataType(DTYPE_KINDS[length_dtype]),
    )
    if family == "items":
        arguments = (TileIRForeignType * 4)(
            run_value_type,
            run_length_type,
            window_offset_type,
            decoded_value_type,
        )
        results = (TileIRForeignType * 1)(decoded_value_type)
    else:
        assert family == "items_and_offsets"
        arguments = (TileIRForeignType * 5)(
            run_value_type,
            run_length_type,
            window_offset_type,
            decoded_value_type,
            decoded_offset_type,
        )
        results = (TileIRForeignType * 2)(
            decoded_value_type,
            decoded_offset_type,
        )
    return run_shape, decoded_shape, arguments, results


def _run_length_decode_decl(arguments, results, family: str):
    names_and_kinds = [
        ("RunN", TILEIR_CONSTANT_PARAMETER_INT),
        ("DecodedN", TILEIR_CONSTANT_PARAMETER_INT),
        ("ItemT", TILEIR_CONSTANT_PARAMETER_DTYPE),
        ("LengthT", TILEIR_CONSTANT_PARAMETER_DTYPE),
        ("ItemDTypeToken", TILEIR_CONSTANT_PARAMETER_STRING),
        ("LengthDTypeToken", TILEIR_CONSTANT_PARAMETER_STRING),
    ]
    const_params = (TileIRConstantParameter * len(names_and_kinds))(
        *(
            TileIRConstantParameter(_string_view(name), kind)
            for name, kind in names_and_kinds
        )
    )
    access_values = [
        TILEIR_LP_ACCESS_IN,
        TILEIR_LP_ACCESS_IN,
        TILEIR_LP_ACCESS_IN,
        TILEIR_LP_ACCESS_OUT,
    ]
    if family == "items_and_offsets":
        access_values.append(TILEIR_LP_ACCESS_OUT)
    access = (ctypes.c_int * len(access_values))(*access_values)
    decl = TileIRForeignDecl(
        _string_view(RUN_LENGTH_DECODE_DECLS[family]),
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


def _run_length_decode_const_values(
    run_extent: int,
    decoded_extent: int,
    value_dtype: str,
    length_dtype: str,
    block_threads: int,
    runs_per_thread: int,
    decoded_items_per_thread: int,
):
    return (
        run_extent,
        decoded_extent,
        value_dtype,
        length_dtype,
        value_dtype,
        length_dtype,
    )


def _run_length_decode_const_arguments(
    *,
    symbolic: bool,
    run_extent: int,
    decoded_extent: int,
    value_dtype: str,
    length_dtype: str,
    block_threads: int,
    runs_per_thread: int,
    decoded_items_per_thread: int,
):
    values = _run_length_decode_const_values(
        run_extent,
        decoded_extent,
        value_dtype,
        length_dtype,
        block_threads,
        runs_per_thread,
        decoded_items_per_thread,
    )
    argument_type = (
        TileIRSymbolicExpression if symbolic else TileIRConstantArgument
    )
    arguments = (argument_type * len(values))()
    for index, value in enumerate(values):
        if index in (2, 3):
            arguments[index].kind = (
                TILEIR_SYMBOLIC_EXPRESSION_DTYPE
                if symbolic
                else TILEIR_CONSTANT_ARGUMENT_DTYPE
            )
            arguments[index].dtype_val = TileIRDataType(DTYPE_KINDS[value])
        elif isinstance(value, int):
            arguments[index].kind = (
                TILEIR_SYMBOLIC_EXPRESSION_INT
                if symbolic
                else TILEIR_CONSTANT_ARGUMENT_INT
            )
            arguments[index].int_val = value
        else:
            arguments[index].kind = (
                TILEIR_SYMBOLIC_EXPRESSION_STRING
                if symbolic
                else TILEIR_CONSTANT_ARGUMENT_STRING
            )
            arguments[index].string_val = _string_view(value)
    return arguments


def _run_length_decode_callsite(
    family: str,
    *,
    run_extent: int = 256,
    decoded_extent: int = 384,
    value_dtype: str = "i32",
    length_dtype: str = "u32",
    block_threads: int = 128,
    runs_per_thread: int = 2,
    decoded_items_per_thread: int = 3,
):
    run_shape, decoded_shape, arguments, results = _run_length_decode_types(
        family,
        run_extent,
        decoded_extent,
        value_dtype,
        length_dtype,
    )
    decl, const_params, access = _run_length_decode_decl(
        arguments, results, family
    )
    const_arguments = _run_length_decode_const_arguments(
        symbolic=True,
        run_extent=run_extent,
        decoded_extent=decoded_extent,
        value_dtype=value_dtype,
        length_dtype=length_dtype,
        block_threads=block_threads,
        runs_per_thread=runs_per_thread,
        decoded_items_per_thread=decoded_items_per_thread,
    )
    callsite = TileIRLPCallsite(
        71 if family == "items" else 72,
        101,
        ctypes.pointer(decl),
        const_arguments,
        len(const_arguments),
        arguments,
        len(arguments),
        results,
        len(results),
    )
    return callsite, (
        run_shape,
        decoded_shape,
        arguments,
        results,
        decl,
        const_params,
        access,
        const_arguments,
    )


def _resolved_run_length_decode_call(
    family: str,
    *,
    run_extent: int = 256,
    decoded_extent: int = 384,
    value_dtype: str = "i32",
    length_dtype: str = "u32",
    block_threads: int = 128,
    runs_per_thread: int = 2,
    decoded_items_per_thread: int = 3,
    impl_symbol: str | None = None,
    architecture: str = "sm_80",
    concrete_suffix: str = DEFAULT_LATE_CALLBACK_SUFFIX,
):
    run_shape, decoded_shape, arguments, results = _run_length_decode_types(
        family,
        run_extent,
        decoded_extent,
        value_dtype,
        length_dtype,
    )
    decl, const_params, access = _run_length_decode_decl(
        arguments, results, family
    )
    const_arguments = _run_length_decode_const_arguments(
        symbolic=False,
        run_extent=run_extent,
        decoded_extent=decoded_extent,
        value_dtype=value_dtype,
        length_dtype=length_dtype,
        block_threads=block_threads,
        runs_per_thread=runs_per_thread,
        decoded_items_per_thread=decoded_items_per_thread,
    )
    context = _execution_context(architecture)
    context.num_warps_per_cta = block_threads // 32
    impl = TileIRForeignImpl(
        _string_view(
            impl_symbol
            or _run_length_decode_impl_symbol(
                family,
                block_threads,
                runs_per_thread,
                decoded_items_per_thread,
                value_dtype,
                length_dtype,
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
        run_shape,
        decoded_shape,
        arguments,
        results,
        decl,
        const_params,
        access,
        const_arguments,
        context,
        impl,
    )
