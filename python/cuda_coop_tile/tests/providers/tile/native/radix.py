# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes

from .harness import (
    DEFAULT_LATE_CALLBACK_SUFFIX,
    DTYPE_KINDS,
    RADIX_RANK_DECL,
    RADIX_SORT_DECLS,
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


def _constant_arguments(values, dtype_indices: set[int], *, resolved: bool):
    value_type = (
        TileIRConstantArgument if resolved else TileIRSymbolicExpression
    )
    arguments = (value_type * len(values))()
    for index, value in enumerate(values):
        if index in dtype_indices:
            arguments[index].kind = (
                TILEIR_CONSTANT_ARGUMENT_DTYPE
                if resolved
                else TILEIR_SYMBOLIC_EXPRESSION_DTYPE
            )
            arguments[index].dtype_val = TileIRDataType(DTYPE_KINDS[value])
        elif isinstance(value, str):
            arguments[index].kind = (
                TILEIR_CONSTANT_ARGUMENT_STRING
                if resolved
                else TILEIR_SYMBOLIC_EXPRESSION_STRING
            )
            arguments[index].string_val = _string_view(value)
        else:
            arguments[index].kind = (
                TILEIR_CONSTANT_ARGUMENT_INT
                if resolved
                else TILEIR_SYMBOLIC_EXPRESSION_INT
            )
            arguments[index].int_val = value
    return arguments


def _radix_sort_const_values(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool,
    radix_bits: int,
    memoize_outer_scan: bool,
    inner_scan_algorithm: str,
    smem_config: str,
):
    if family == "keys":
        return (
            128 * items_per_thread,
            key_dtype,
            int(descending),
            radix_bits,
            int(memoize_outer_scan),
            inner_scan_algorithm,
            smem_config,
            key_dtype,
        )
    assert family == "pairs"
    assert value_dtype is not None
    return (
        128 * items_per_thread,
        key_dtype,
        value_dtype,
        int(descending),
        radix_bits,
        int(memoize_outer_scan),
        inner_scan_algorithm,
        smem_config,
        key_dtype,
        value_dtype,
    )


def _radix_sort_primitive_meta(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool,
    radix_bits: int,
    memoize_outer_scan: bool,
    inner_scan_algorithm: str,
    smem_config: str,
) -> str:
    if family == "keys":
        return (
            f"block_threads=128;descending={int(descending)};dtype={key_dtype};"
            f"inner_scan_algorithm={inner_scan_algorithm};"
            f"items_per_thread={items_per_thread};"
            f"memoize_outer_scan={int(memoize_outer_scan)};"
            f"radix_bits={radix_bits};smem_config={smem_config}"
        )
    assert family == "pairs"
    assert value_dtype is not None
    return (
        f"block_threads=128;descending={int(descending)};"
        f"inner_scan_algorithm={inner_scan_algorithm};"
        f"items_per_thread={items_per_thread};key_dtype={key_dtype};"
        f"memoize_outer_scan={int(memoize_outer_scan)};"
        f"radix_bits={radix_bits};smem_config={smem_config};"
        f"value_dtype={value_dtype}"
    )


def _radix_sort_semantic_key(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool,
    radix_bits: int,
    memoize_outer_scan: bool,
    inner_scan_algorithm: str,
    smem_config: str,
) -> str:
    metadata = _radix_sort_primitive_meta(
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
        descending,
        radix_bits,
        memoize_outer_scan,
        inner_scan_algorithm,
        smem_config,
    )
    return f"coop::{RADIX_SORT_DECLS[family]}|{metadata}"


def _radix_sort_impl_symbol(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool,
    radix_bits: int,
    memoize_outer_scan: bool,
    inner_scan_algorithm: str,
    smem_config: str,
    *,
    architecture: str = "sm_80",
) -> str:
    identity = _radix_sort_semantic_key(
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
        descending,
        radix_bits,
        memoize_outer_scan,
        inner_scan_algorithm,
        smem_config,
    )
    return (
        f"{RADIX_SORT_DECLS[family]}_impl__lpv4_"
        f"{_fnv1a(_artifact_identity(identity, architecture)):x}"
    )


def _radix_sort_types(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
):
    tile_shape = (TileIRDim * 1)()
    tile_shape[0].kind = TILEIR_DIM_CONCRETE
    tile_shape[0].value = 128 * items_per_thread
    key_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        tile_shape,
        len(tile_shape),
        TileIRDataType(DTYPE_KINDS[key_dtype]),
    )
    scalar_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        None,
        0,
        TileIRDataType(TILEIR_DTYPE_I32),
    )
    if family == "keys":
        arguments = (TileIRForeignType * 4)(
            key_type,
            scalar_type,
            scalar_type,
            key_type,
        )
        results = (TileIRForeignType * 1)(key_type)
        return tile_shape, None, arguments, results

    assert family == "pairs"
    assert value_dtype is not None
    value_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        tile_shape,
        len(tile_shape),
        TileIRDataType(DTYPE_KINDS[value_dtype]),
    )
    arguments = (TileIRForeignType * 6)(
        key_type,
        value_type,
        scalar_type,
        scalar_type,
        key_type,
        value_type,
    )
    results = (TileIRForeignType * 2)(key_type, value_type)
    return tile_shape, None, arguments, results


def _radix_sort_decl(arguments, results, family: str):
    if family == "keys":
        names_and_kinds = (
            ("N", TILEIR_CONSTANT_PARAMETER_INT),
            ("T", TILEIR_CONSTANT_PARAMETER_DTYPE),
            ("Descending", TILEIR_CONSTANT_PARAMETER_INT),
            ("RadixBits", TILEIR_CONSTANT_PARAMETER_INT),
            ("MemoizeOuterScan", TILEIR_CONSTANT_PARAMETER_INT),
            ("InnerScanAlgorithm", TILEIR_CONSTANT_PARAMETER_STRING),
            ("SmemConfig", TILEIR_CONSTANT_PARAMETER_STRING),
            ("DTypeToken", TILEIR_CONSTANT_PARAMETER_STRING),
        )
        access_values = (
            TILEIR_LP_ACCESS_IN,
            TILEIR_LP_ACCESS_IN,
            TILEIR_LP_ACCESS_IN,
            TILEIR_LP_ACCESS_OUT,
        )
    else:
        assert family == "pairs"
        names_and_kinds = (
            ("N", TILEIR_CONSTANT_PARAMETER_INT),
            ("KeyT", TILEIR_CONSTANT_PARAMETER_DTYPE),
            ("ValueT", TILEIR_CONSTANT_PARAMETER_DTYPE),
            ("Descending", TILEIR_CONSTANT_PARAMETER_INT),
            ("RadixBits", TILEIR_CONSTANT_PARAMETER_INT),
            ("MemoizeOuterScan", TILEIR_CONSTANT_PARAMETER_INT),
            ("InnerScanAlgorithm", TILEIR_CONSTANT_PARAMETER_STRING),
            ("SmemConfig", TILEIR_CONSTANT_PARAMETER_STRING),
            ("KeyDTypeToken", TILEIR_CONSTANT_PARAMETER_STRING),
            ("ValueDTypeToken", TILEIR_CONSTANT_PARAMETER_STRING),
        )
        access_values = (
            TILEIR_LP_ACCESS_IN,
            TILEIR_LP_ACCESS_IN,
            TILEIR_LP_ACCESS_IN,
            TILEIR_LP_ACCESS_IN,
            TILEIR_LP_ACCESS_OUT,
            TILEIR_LP_ACCESS_OUT,
        )
    const_params = (TileIRConstantParameter * len(names_and_kinds))(
        *(
            TileIRConstantParameter(_string_view(name), kind)
            for name, kind in names_and_kinds
        )
    )
    access = (ctypes.c_int * len(access_values))(*access_values)
    decl = TileIRForeignDecl(
        _string_view(RADIX_SORT_DECLS[family]),
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


def _radix_sort_call_parts(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool,
    radix_bits: int,
    memoize_outer_scan: bool,
    inner_scan_algorithm: str,
    smem_config: str,
    *,
    resolved: bool,
):
    tile_shape, scalar_shape, arguments, results = _radix_sort_types(
        family, items_per_thread, key_dtype, value_dtype
    )
    decl, const_params, access = _radix_sort_decl(arguments, results, family)
    values = _radix_sort_const_values(
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
        descending,
        radix_bits,
        memoize_outer_scan,
        inner_scan_algorithm,
        smem_config,
    )
    dtype_indices = {1} if family == "keys" else {1, 2}
    const_arguments = _constant_arguments(
        values, dtype_indices, resolved=resolved
    )
    return (
        tile_shape,
        scalar_shape,
        arguments,
        results,
        decl,
        const_params,
        access,
        const_arguments,
    )


def _radix_sort_callsite(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool = False,
    radix_bits: int = 4,
    memoize_outer_scan: bool = True,
    inner_scan_algorithm: str = "warp_scans",
    smem_config: str = "four_byte",
):
    parts = _radix_sort_call_parts(
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
        descending,
        radix_bits,
        memoize_outer_scan,
        inner_scan_algorithm,
        smem_config,
        resolved=False,
    )
    _, _, arguments, results, decl, _, _, const_arguments = parts
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
    return callsite, parts


def _resolved_radix_sort_call(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool = False,
    radix_bits: int = 4,
    memoize_outer_scan: bool = True,
    inner_scan_algorithm: str = "warp_scans",
    smem_config: str = "four_byte",
    impl_symbol: str | None = None,
    architecture: str = "sm_80",
    concrete_suffix: str = DEFAULT_LATE_CALLBACK_SUFFIX,
):
    parts = _radix_sort_call_parts(
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
        descending,
        radix_bits,
        memoize_outer_scan,
        inner_scan_algorithm,
        smem_config,
        resolved=True,
    )
    _, _, arguments, _, decl, _, _, const_arguments = parts
    context = _execution_context(architecture)
    impl = TileIRForeignImpl(
        _string_view(
            impl_symbol
            or _radix_sort_impl_symbol(
                family,
                items_per_thread,
                key_dtype,
                value_dtype,
                descending,
                radix_bits,
                memoize_outer_scan,
                inner_scan_algorithm,
                smem_config,
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


def _radix_rank_semantic_key(
    dtype: str,
    block_threads: int,
    items_per_thread: int,
    radix_bits: int,
    descending: bool,
    memoize_outer_scan: bool,
    inner_scan_algorithm: str,
    smem_config: str,
) -> str:
    metadata = (
        f"block_threads={block_threads};descending={int(descending)};"
        f"dtype={dtype};inner_scan_algorithm={inner_scan_algorithm};"
        f"items_per_thread={items_per_thread};"
        f"memoize_outer_scan={int(memoize_outer_scan)};"
        f"radix_bits={radix_bits};smem_config={smem_config}"
    )
    return f"coop::{RADIX_RANK_DECL}|{metadata}"


def _radix_rank_impl_symbol(
    dtype: str,
    block_threads: int,
    items_per_thread: int,
    radix_bits: int = 4,
    descending: bool = False,
    memoize_outer_scan: bool = True,
    inner_scan_algorithm: str = "warp_scans",
    smem_config: str = "four_byte",
    *,
    architecture: str = "sm_80",
) -> str:
    identity = _radix_rank_semantic_key(
        dtype,
        block_threads,
        items_per_thread,
        radix_bits,
        descending,
        memoize_outer_scan,
        inner_scan_algorithm,
        smem_config,
    )
    return (
        f"{RADIX_RANK_DECL}_impl__lpv4_"
        f"{_fnv1a(_artifact_identity(identity, architecture)):x}"
    )


def _radix_rank_const_arguments(
    *,
    resolved: bool,
    num_items: int,
    dtype: str,
    block_threads: int,
    items_per_thread: int,
    radix_bits: int,
    descending: bool,
    memoize_outer_scan: bool,
    inner_scan_algorithm: str,
    smem_config: str,
    dtype_token: str,
):
    values = (
        num_items,
        dtype,
        radix_bits,
        int(descending),
        int(memoize_outer_scan),
        inner_scan_algorithm,
        smem_config,
        dtype_token,
    )
    return _constant_arguments(values, {1}, resolved=resolved)


def _radix_rank_call_parts(
    *,
    resolved: bool,
    num_items: int,
    dtype: str,
    block_threads: int,
    items_per_thread: int,
    radix_bits: int,
    descending: bool,
    memoize_outer_scan: bool,
    inner_scan_algorithm: str,
    smem_config: str,
    dtype_token: str,
):
    tile_shape = (TileIRDim * 1)()
    tile_shape[0].kind = TILEIR_DIM_CONCRETE
    tile_shape[0].value = num_items
    key_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        tile_shape,
        len(tile_shape),
        TileIRDataType(DTYPE_KINDS[dtype]),
    )
    scalar_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        None,
        0,
        TileIRDataType(TILEIR_DTYPE_I32),
    )
    rank_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        tile_shape,
        len(tile_shape),
        TileIRDataType(TILEIR_DTYPE_I32),
    )
    arguments = (TileIRForeignType * 4)(
        key_type,
        scalar_type,
        scalar_type,
        rank_type,
    )
    results = (TileIRForeignType * 1)(rank_type)
    names_and_kinds = (
        ("N", TILEIR_CONSTANT_PARAMETER_INT),
        ("T", TILEIR_CONSTANT_PARAMETER_DTYPE),
        ("RadixBits", TILEIR_CONSTANT_PARAMETER_INT),
        ("Descending", TILEIR_CONSTANT_PARAMETER_INT),
        ("MemoizeOuterScan", TILEIR_CONSTANT_PARAMETER_INT),
        ("InnerScanAlgorithm", TILEIR_CONSTANT_PARAMETER_STRING),
        ("SmemConfig", TILEIR_CONSTANT_PARAMETER_STRING),
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
        _string_view(RADIX_RANK_DECL),
        const_params,
        len(const_params),
        arguments,
        len(arguments),
        results,
        len(results),
        access,
        len(access),
    )
    const_arguments = _radix_rank_const_arguments(
        resolved=resolved,
        num_items=num_items,
        dtype=dtype,
        block_threads=block_threads,
        items_per_thread=items_per_thread,
        radix_bits=radix_bits,
        descending=descending,
        memoize_outer_scan=memoize_outer_scan,
        inner_scan_algorithm=inner_scan_algorithm,
        smem_config=smem_config,
        dtype_token=dtype_token,
    )
    return (
        tile_shape,
        arguments,
        results,
        decl,
        const_params,
        access,
        const_arguments,
    )


def _radix_rank_callsite(
    *,
    num_items: int = 256,
    dtype: str = "u32",
    block_threads: int = 128,
    items_per_thread: int | None = None,
    radix_bits: int = 4,
    descending: bool = False,
    memoize_outer_scan: bool = True,
    inner_scan_algorithm: str = "warp_scans",
    smem_config: str = "four_byte",
    dtype_token: str | None = None,
):
    if items_per_thread is None:
        items_per_thread = (
            0 if block_threads == 0 else num_items // block_threads
        )
    parts = _radix_rank_call_parts(
        resolved=False,
        num_items=num_items,
        dtype=dtype,
        block_threads=block_threads,
        items_per_thread=items_per_thread,
        radix_bits=radix_bits,
        descending=descending,
        memoize_outer_scan=memoize_outer_scan,
        inner_scan_algorithm=inner_scan_algorithm,
        smem_config=smem_config,
        dtype_token=dtype_token or dtype,
    )
    _, arguments, results, decl, _, _, const_arguments = parts
    callsite = TileIRLPCallsite(
        59,
        61,
        ctypes.pointer(decl),
        const_arguments,
        len(const_arguments),
        arguments,
        len(arguments),
        results,
        len(results),
    )
    return callsite, parts


def _resolved_radix_rank_call(
    *,
    num_items: int = 256,
    dtype: str = "u32",
    block_threads: int = 128,
    items_per_thread: int | None = None,
    resolved_block_threads: int | None = None,
    radix_bits: int = 4,
    descending: bool = False,
    memoize_outer_scan: bool = True,
    inner_scan_algorithm: str = "warp_scans",
    smem_config: str = "four_byte",
    dtype_token: str | None = None,
    impl_symbol: str | None = None,
    architecture: str = "sm_80",
    concrete_suffix: str = DEFAULT_LATE_CALLBACK_SUFFIX,
):
    if items_per_thread is None:
        items_per_thread = (
            0 if block_threads == 0 else num_items // block_threads
        )
    parts = _radix_rank_call_parts(
        resolved=True,
        num_items=num_items,
        dtype=dtype,
        block_threads=block_threads,
        items_per_thread=items_per_thread,
        radix_bits=radix_bits,
        descending=descending,
        memoize_outer_scan=memoize_outer_scan,
        inner_scan_algorithm=inner_scan_algorithm,
        smem_config=smem_config,
        dtype_token=dtype_token or dtype,
    )
    _, arguments, _, decl, _, _, const_arguments = parts
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
            or _radix_rank_impl_symbol(
                dtype_token or dtype,
                resolved_threads,
                resolved_items,
                radix_bits,
                descending,
                memoize_outer_scan,
                inner_scan_algorithm,
                smem_config,
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
