# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import ctypes

from .harness import (
    DEFAULT_LATE_CALLBACK_SUFFIX,
    DTYPE_KINDS,
    MERGE_SORT_DECLS,
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
    TOPK_DECLS,
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


def _topk_const_values(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool,
    full_tile: bool,
):
    common = (
        128 * items_per_thread,
        key_dtype,
    )
    if family == "keys":
        values = (
            *common,
            int(descending),
            int(full_tile),
        )
        return (*values, key_dtype)
    assert family == "pairs"
    assert value_dtype is not None
    values = (
        *common,
        value_dtype,
        int(descending),
        int(full_tile),
    )
    return (*values, key_dtype, value_dtype)


def _topk_primitive_meta(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool,
    full_tile: bool,
) -> str:
    if family == "keys":
        return (
            f"block_threads=128;descending={int(descending)};dtype={key_dtype};"
            f"full_tile={int(full_tile)};items_per_thread={items_per_thread}"
        )
    assert family == "pairs"
    assert value_dtype is not None
    return (
        f"block_threads=128;descending={int(descending)};"
        f"full_tile={int(full_tile)};items_per_thread={items_per_thread};"
        f"key_dtype={key_dtype};value_dtype={value_dtype}"
    )


def _topk_semantic_key(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool,
    full_tile: bool,
) -> str:
    primitive_meta = _topk_primitive_meta(
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
        descending,
        full_tile,
    )
    return f"coop::cuda_coop_tile_topk_{family}|{primitive_meta}"


def _topk_impl_symbol(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool,
    full_tile: bool,
    *,
    architecture: str = "sm_80",
) -> str:
    semantic_key = _topk_semantic_key(
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
        descending,
        full_tile,
    )
    prefix = f"{TOPK_DECLS[family]}_impl"
    digest = _fnv1a(_artifact_identity(semantic_key, architecture))
    return f"{prefix}__lpv4_{digest:x}"


def _topk_types(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
):
    tile_shape = (TileIRDim * 1)()
    tile_shape[0].kind = TILEIR_DIM_CONCRETE
    tile_shape[0].value = 128 * items_per_thread
    scalar_shape = None

    key_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        tile_shape,
        len(tile_shape),
        TileIRDataType(DTYPE_KINDS[key_dtype]),
    )
    scalar_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        scalar_shape,
        0 if scalar_shape is None else len(scalar_shape),
        TileIRDataType(TILEIR_DTYPE_I32),
    )
    if family == "keys":
        arguments = (TileIRForeignType * 6)(
            key_type,
            scalar_type,
            scalar_type,
            scalar_type,
            scalar_type,
            key_type,
        )
        results = (TileIRForeignType * 1)(key_type)
        return tile_shape, scalar_shape, arguments, results

    assert family == "pairs"
    assert value_dtype is not None
    value_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        tile_shape,
        len(tile_shape),
        TileIRDataType(DTYPE_KINDS[value_dtype]),
    )
    arguments = (TileIRForeignType * 8)(
        key_type,
        value_type,
        scalar_type,
        scalar_type,
        scalar_type,
        scalar_type,
        key_type,
        value_type,
    )
    results = (TileIRForeignType * 2)(key_type, value_type)
    return tile_shape, scalar_shape, arguments, results


def _topk_decl(
    arguments,
    results,
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool,
    full_tile: bool,
):
    del items_per_thread, key_dtype, value_dtype, descending, full_tile
    if family == "keys":
        names_and_kinds = [
            ("N", TILEIR_CONSTANT_PARAMETER_INT),
            ("T", TILEIR_CONSTANT_PARAMETER_DTYPE),
            ("Descending", TILEIR_CONSTANT_PARAMETER_INT),
            ("FullTile", TILEIR_CONSTANT_PARAMETER_INT),
        ]
        names_and_kinds.append(("DTypeToken", TILEIR_CONSTANT_PARAMETER_STRING))
        access_values = (
            TILEIR_LP_ACCESS_IN,
            TILEIR_LP_ACCESS_IN,
            TILEIR_LP_ACCESS_IN,
            TILEIR_LP_ACCESS_IN,
            TILEIR_LP_ACCESS_IN,
            TILEIR_LP_ACCESS_OUT,
        )
    else:
        assert family == "pairs"
        names_and_kinds = [
            ("N", TILEIR_CONSTANT_PARAMETER_INT),
            ("KeyT", TILEIR_CONSTANT_PARAMETER_DTYPE),
            ("ValueT", TILEIR_CONSTANT_PARAMETER_DTYPE),
            ("Descending", TILEIR_CONSTANT_PARAMETER_INT),
            ("FullTile", TILEIR_CONSTANT_PARAMETER_INT),
        ]
        names_and_kinds.extend(
            [
                ("KeyDTypeToken", TILEIR_CONSTANT_PARAMETER_STRING),
                ("ValueDTypeToken", TILEIR_CONSTANT_PARAMETER_STRING),
            ]
        )
        access_values = (
            TILEIR_LP_ACCESS_IN,
            TILEIR_LP_ACCESS_IN,
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
        _string_view(TOPK_DECLS[family]),
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


def _symbolic_topk_arguments(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool,
    full_tile: bool,
):
    const_values = _topk_const_values(
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
        descending,
        full_tile,
    )
    dtype_indices = {1} if family == "keys" else {1, 2}
    values = (TileIRSymbolicExpression * len(const_values))()
    for index, value in enumerate(const_values):
        if index in dtype_indices:
            values[index].kind = TILEIR_SYMBOLIC_EXPRESSION_DTYPE
            values[index].dtype_val = TileIRDataType(DTYPE_KINDS[value])
        elif isinstance(value, str):
            values[index].kind = TILEIR_SYMBOLIC_EXPRESSION_STRING
            values[index].string_val = _string_view(value)
        else:
            values[index].kind = TILEIR_SYMBOLIC_EXPRESSION_INT
            values[index].int_val = value
    return values


def _concrete_topk_arguments(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool,
    full_tile: bool,
):
    const_values = _topk_const_values(
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
        descending,
        full_tile,
    )
    dtype_indices = {1} if family == "keys" else {1, 2}
    values = (TileIRConstantArgument * len(const_values))()
    for index, value in enumerate(const_values):
        if index in dtype_indices:
            values[index].kind = TILEIR_CONSTANT_ARGUMENT_DTYPE
            values[index].dtype_val = TileIRDataType(DTYPE_KINDS[value])
        elif isinstance(value, str):
            values[index].kind = TILEIR_CONSTANT_ARGUMENT_STRING
            values[index].string_val = _string_view(value)
        else:
            values[index].kind = TILEIR_CONSTANT_ARGUMENT_INT
            values[index].int_val = value
    return values


def _topk_callsite(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool,
    full_tile: bool,
):
    tile_shape, scalar_shape, arguments, results = _topk_types(
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
    )
    decl, const_params, access = _topk_decl(
        arguments,
        results,
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
        descending,
        full_tile,
    )
    const_arguments = _symbolic_topk_arguments(
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
        descending,
        full_tile,
    )
    callsite = TileIRLPCallsite(
        17,
        19,
        ctypes.pointer(decl),
        const_arguments,
        len(const_arguments),
        arguments,
        len(arguments),
        results,
        len(results),
    )
    return callsite, (
        tile_shape,
        scalar_shape,
        arguments,
        results,
        decl,
        const_params,
        access,
        const_arguments,
    )


def _resolved_topk_call(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool,
    full_tile: bool,
    impl_symbol: str | None = None,
    architecture: str = "sm_80",
    concrete_suffix: str = DEFAULT_LATE_CALLBACK_SUFFIX,
):
    tile_shape, scalar_shape, arguments, results = _topk_types(
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
    )
    decl, const_params, access = _topk_decl(
        arguments,
        results,
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
        descending,
        full_tile,
    )
    const_arguments = _concrete_topk_arguments(
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
        descending,
        full_tile,
    )
    context = _execution_context(architecture)
    impl = TileIRForeignImpl(
        _string_view(
            impl_symbol
            or _topk_impl_symbol(
                family,
                items_per_thread,
                key_dtype,
                value_dtype,
                descending,
                full_tile,
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
        tile_shape,
        scalar_shape,
        arguments,
        results,
        decl,
        const_params,
        access,
        const_arguments,
        context,
        impl,
    )


def _merge_sort_const_values(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool,
    partial_tile: bool,
):
    common = (128 * items_per_thread, key_dtype)
    if family == "keys":
        values = (
            *common,
            int(descending),
            int(partial_tile),
        )
        return (*values, key_dtype)
    assert family == "pairs"
    assert value_dtype is not None
    values = (
        *common,
        value_dtype,
        int(descending),
        int(partial_tile),
    )
    return (*values, key_dtype, value_dtype)


def _merge_sort_primitive_meta(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool,
    partial_tile: bool,
) -> str:
    if family == "keys":
        return (
            f"block_threads=128;descending={int(descending)};dtype={key_dtype};"
            f"items_per_thread={items_per_thread};"
            f"partial_tile={int(partial_tile)}"
        )
    assert family == "pairs"
    assert value_dtype is not None
    return (
        f"block_threads=128;descending={int(descending)};"
        f"items_per_thread={items_per_thread};key_dtype={key_dtype};"
        f"partial_tile={int(partial_tile)};value_dtype={value_dtype}"
    )


def _merge_sort_semantic_key(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool,
    partial_tile: bool,
) -> str:
    primitive_meta = _merge_sort_primitive_meta(
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
        descending,
        partial_tile,
    )
    return f"coop::cuda_coop_tile_merge_sort_{family}|{primitive_meta}"


def _merge_sort_impl_symbol(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool,
    partial_tile: bool,
    *,
    architecture: str = "sm_80",
) -> str:
    semantic_key = _merge_sort_semantic_key(
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
        descending,
        partial_tile,
    )
    prefix = f"{MERGE_SORT_DECLS[family]}_impl"
    digest = _fnv1a(_artifact_identity(semantic_key, architecture))
    return f"{prefix}__lpv4_{digest:x}"


def _merge_sort_types(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
):
    tile_shape = (TileIRDim * 1)()
    tile_shape[0].kind = TILEIR_DIM_CONCRETE
    tile_shape[0].value = 128 * items_per_thread
    scalar_shape = None

    key_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        tile_shape,
        len(tile_shape),
        TileIRDataType(DTYPE_KINDS[key_dtype]),
    )
    valid_items_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        scalar_shape,
        0 if scalar_shape is None else len(scalar_shape),
        TileIRDataType(TILEIR_DTYPE_I32),
    )
    oob_default_type = TileIRForeignType(
        TILEIR_TYPE_TILE,
        scalar_shape,
        0 if scalar_shape is None else len(scalar_shape),
        TileIRDataType(DTYPE_KINDS[key_dtype]),
    )
    if family == "keys":
        arguments = (TileIRForeignType * 4)(
            key_type,
            valid_items_type,
            oob_default_type,
            key_type,
        )
        results = (TileIRForeignType * 1)(key_type)
        return tile_shape, scalar_shape, arguments, results

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
        valid_items_type,
        oob_default_type,
        key_type,
        value_type,
    )
    results = (TileIRForeignType * 2)(key_type, value_type)
    return tile_shape, scalar_shape, arguments, results


def _merge_sort_decl(
    arguments,
    results,
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool,
    partial_tile: bool,
):
    del items_per_thread, key_dtype, value_dtype, descending, partial_tile
    if family == "keys":
        names_and_kinds = [
            ("N", TILEIR_CONSTANT_PARAMETER_INT),
            ("T", TILEIR_CONSTANT_PARAMETER_DTYPE),
            ("Descending", TILEIR_CONSTANT_PARAMETER_INT),
            ("PartialTile", TILEIR_CONSTANT_PARAMETER_INT),
        ]
        names_and_kinds.append(("DTypeToken", TILEIR_CONSTANT_PARAMETER_STRING))
        access_values = (
            TILEIR_LP_ACCESS_IN,
            TILEIR_LP_ACCESS_IN,
            TILEIR_LP_ACCESS_IN,
            TILEIR_LP_ACCESS_OUT,
        )
    else:
        assert family == "pairs"
        names_and_kinds = [
            ("N", TILEIR_CONSTANT_PARAMETER_INT),
            ("KeyT", TILEIR_CONSTANT_PARAMETER_DTYPE),
            ("ValueT", TILEIR_CONSTANT_PARAMETER_DTYPE),
            ("Descending", TILEIR_CONSTANT_PARAMETER_INT),
            ("PartialTile", TILEIR_CONSTANT_PARAMETER_INT),
        ]
        names_and_kinds.extend(
            [
                ("KeyDTypeToken", TILEIR_CONSTANT_PARAMETER_STRING),
                ("ValueDTypeToken", TILEIR_CONSTANT_PARAMETER_STRING),
            ]
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
        _string_view(MERGE_SORT_DECLS[family]),
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


def _merge_sort_constant_arguments(
    expression_type,
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool,
    partial_tile: bool,
):
    values = _merge_sort_const_values(
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
        descending,
        partial_tile,
    )
    dtype_indices = {1} if family == "keys" else {1, 2}
    arguments = (expression_type * len(values))()
    symbolic = expression_type is TileIRSymbolicExpression
    for index, value in enumerate(values):
        if index in dtype_indices:
            arguments[index].kind = (
                TILEIR_SYMBOLIC_EXPRESSION_DTYPE
                if symbolic
                else TILEIR_CONSTANT_ARGUMENT_DTYPE
            )
            arguments[index].dtype_val = TileIRDataType(DTYPE_KINDS[value])
        elif isinstance(value, str):
            arguments[index].kind = (
                TILEIR_SYMBOLIC_EXPRESSION_STRING
                if symbolic
                else TILEIR_CONSTANT_ARGUMENT_STRING
            )
            arguments[index].string_val = _string_view(value)
        else:
            arguments[index].kind = (
                TILEIR_SYMBOLIC_EXPRESSION_INT
                if symbolic
                else TILEIR_CONSTANT_ARGUMENT_INT
            )
            arguments[index].int_val = value
    return arguments


def _merge_sort_callsite(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool,
    partial_tile: bool,
):
    tile_shape, scalar_shape, arguments, results = _merge_sort_types(
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
    )
    decl, const_params, access = _merge_sort_decl(
        arguments,
        results,
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
        descending,
        partial_tile,
    )
    const_arguments = _merge_sort_constant_arguments(
        TileIRSymbolicExpression,
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
        descending,
        partial_tile,
    )
    callsite = TileIRLPCallsite(
        29,
        31,
        ctypes.pointer(decl),
        const_arguments,
        len(const_arguments),
        arguments,
        len(arguments),
        results,
        len(results),
    )
    return callsite, (
        tile_shape,
        scalar_shape,
        arguments,
        results,
        decl,
        const_params,
        access,
        const_arguments,
    )


def _resolved_merge_sort_call(
    family: str,
    items_per_thread: int,
    key_dtype: str,
    value_dtype: str | None,
    descending: bool,
    partial_tile: bool,
    impl_symbol: str | None = None,
    architecture: str = "sm_80",
    concrete_suffix: str = DEFAULT_LATE_CALLBACK_SUFFIX,
):
    tile_shape, scalar_shape, arguments, results = _merge_sort_types(
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
    )
    decl, const_params, access = _merge_sort_decl(
        arguments,
        results,
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
        descending,
        partial_tile,
    )
    const_arguments = _merge_sort_constant_arguments(
        TileIRConstantArgument,
        family,
        items_per_thread,
        key_dtype,
        value_dtype,
        descending,
        partial_tile,
    )
    context = _execution_context(architecture)
    impl = TileIRForeignImpl(
        _string_view(
            impl_symbol
            or _merge_sort_impl_symbol(
                family,
                items_per_thread,
                key_dtype,
                value_dtype,
                descending,
                partial_tile,
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
        tile_shape,
        scalar_shape,
        arguments,
        results,
        decl,
        const_params,
        access,
        const_arguments,
        context,
        impl,
    )
