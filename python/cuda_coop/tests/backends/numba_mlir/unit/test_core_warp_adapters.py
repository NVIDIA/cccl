# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


def test_numba_mlir_warp_adapters_lower_shared_specs():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop._core import INT8, CxxFunction, CxxOperator, Dependency
    from cuda.coop._core.warp import (
        make_warp_exchange_spec,
        make_warp_load_spec,
        make_warp_merge_sort_spec,
        make_warp_reduce_spec,
        make_warp_scan_spec,
        make_warp_store_spec,
    )
    from cuda.coop.numba_mlir._core_adapter import NumbaMlirCoreAdapter
    from cuda.coop.numba_mlir._types import algo_coalesce_key

    adapter = NumbaMlirCoreAdapter()
    assert adapter.normalize_dtype(INT8) is types.int8

    reduce_spec = make_warp_reduce_spec(
        dtype=types.int32,
        threads_in_warp=16,
        operation="min",
        valid_items=True,
        include_full_warp=True,
    )
    reduce_specialization = NumbaMlirCoreAdapter().materialize(
        reduce_spec.specialization
    )
    assert [
        [type(parameter).__name__ for parameter in method]
        for method in reduce_specialization.parameters
    ] == [
        ["Pointer", "Reference", "Reference"],
        ["Pointer", "Reference", "Reference", "Value"],
    ]
    assert reduce_specialization.parameters[0][2].is_output

    load_spec = make_warp_load_spec(
        dtype=types.int32,
        items_per_thread=2,
        threads_in_warp=16,
        algorithm="transpose",
        valid_items=True,
        oob_default=True,
        include_full_tile=True,
    )
    load_specialization = NumbaMlirCoreAdapter().materialize(load_spec.specialization)
    assert [
        [type(parameter).__name__ for parameter in method]
        for method in load_specialization.parameters
    ] == [
        ["Pointer", "Pointer", "Array"],
        ["Pointer", "Pointer", "Array", "Value", "Value"],
    ]
    mlir_load_pointer = load_specialization.parameters[0][1]
    assert not mlir_load_pointer.is_output
    assert not hasattr(mlir_load_pointer, "restrict")
    assert not hasattr(mlir_load_pointer, "is_array_pointer")

    store_spec = make_warp_store_spec(
        dtype=types.int32,
        items_per_thread=2,
        threads_in_warp=16,
        algorithm="direct",
        valid_items=True,
        include_full_tile=True,
        include_pointer_offset=True,
    )
    store_specialization = NumbaMlirCoreAdapter().materialize(store_spec.specialization)
    assert [
        [type(parameter).__name__ for parameter in method]
        for method in store_specialization.parameters
    ] == [
        ["Pointer", "Pointer", "Array"],
        ["Pointer", "Pointer", "Array", "Value"],
        ["Pointer", "Pointer", "Array", "PointerOffset"],
        ["Pointer", "Pointer", "Array", "Value", "PointerOffset"],
    ]
    mlir_store_pointer = store_specialization.parameters[0][1]
    assert not mlir_store_pointer.is_output
    assert not hasattr(mlir_store_pointer, "restrict")
    assert not hasattr(mlir_store_pointer, "is_array_pointer")

    exchange_spec = make_warp_exchange_spec(
        dtype=types.int32,
        items_per_thread=2,
        threads_in_warp=16,
        mode="scatter_to_striped",
        value_form="both",
        rank_dtype=types.int32,
    )
    exchange_specialization = NumbaMlirCoreAdapter().materialize(
        exchange_spec.specialization
    )
    assert exchange_specialization._symbol_base_name().endswith("_ScatterToStriped")
    assert [
        [type(parameter).__name__ for parameter in method]
        for method in exchange_specialization.parameters
    ] == [
        ["Pointer", "Array", "Array"],
        ["Pointer", "Array", "Array", "Array"],
    ]
    assert not exchange_specialization.parameters[0][1].is_output
    assert not exchange_specialization.parameters[1][2].is_output

    blocked_exchange = NumbaMlirCoreAdapter().materialize(
        make_warp_exchange_spec(
            dtype=types.int32,
            items_per_thread=2,
            threads_in_warp=16,
            mode="blocked_to_striped",
        ).specialization
    )
    assert exchange_specialization.c_name == blocked_exchange.c_name
    assert algo_coalesce_key(exchange_specialization) != algo_coalesce_key(
        blocked_exchange
    )

    merge_sort_spec = make_warp_merge_sort_spec(
        key_dtype=types.int32,
        value_dtype=types.float32,
        items_per_thread=2,
        threads_in_warp=16,
        compare_operator=CxxOperator(
            "::cuda::std::less<KeyT>",
            Dependency("KeyT"),
            name="compare_op",
        ),
    )
    merge_sort_specialization = adapter.materialize(merge_sort_spec.specialization)
    assert [
        type(parameter).__name__
        for parameter in merge_sort_specialization.parameters[0]
    ] == ["Pointer", "Array", "Array", "CxxFunction"]
    assert not merge_sort_specialization.parameters[0][1].is_output
    assert not merge_sort_specialization.parameters[0][2].is_output

    partial_merge_sort_spec = make_warp_merge_sort_spec(
        key_dtype=types.int32,
        value_dtype=types.float32,
        items_per_thread=2,
        threads_in_warp=16,
        compare_operator=CxxOperator(
            "::cuda::std::less<KeyT>",
            Dependency("KeyT"),
            name="compare_op",
        ),
        valid_items=31,
        oob_default=0,
    )
    partial_merge_sort_specialization = adapter.materialize(
        partial_merge_sort_spec.specialization
    )
    assert [
        type(parameter).__name__
        for parameter in partial_merge_sort_specialization.parameters[0]
    ] == ["Pointer", "Array", "Array", "CxxFunction", "Value", "Reference"]
    assert algo_coalesce_key(partial_merge_sort_specialization) != algo_coalesce_key(
        merge_sort_specialization
    )

    scan_spec = make_warp_scan_spec(
        dtype=types.int32,
        threads_in_warp=16,
        mode="exclusive",
        scan_operator=CxxOperator(
            "::cuda::maximum<T>",
            Dependency("T"),
            name="scan_op",
        ),
        initial_value=CxxFunction("0", types.int32, name="initial_value"),
        valid_items=True,
        warp_aggregate=True,
    )
    scan_specialization = NumbaMlirCoreAdapter().materialize(scan_spec.specialization)
    assert scan_specialization.fake_return
    assert [
        type(parameter).__name__ for parameter in scan_specialization.parameters[0]
    ] == [
        "Pointer",
        "Reference",
        "Reference",
        "CxxFunction",
        "CxxFunction",
        "Value",
        "PointerReference",
    ]
    assert scan_specialization.parameters[0][2].is_output
    assert not scan_specialization.parameters[0][-1].is_output


def test_numba_mlir_warp_provider_symbols_include_physical_block_size():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop._core.warp import make_warp_store_spec
    from cuda.coop.numba_mlir._core_adapter import NumbaMlirCoreAdapter

    def materialize():
        spec = make_warp_store_spec(
            dtype=types.int32,
            items_per_thread=2,
            threads_in_warp=32,
            algorithm="transpose",
        )
        return NumbaMlirCoreAdapter().materialize(spec.specialization)

    block_64 = materialize()
    block_64_tuple = materialize()
    block_128 = materialize()
    block_default = materialize()
    block_1024 = materialize()
    block_64._source_code(threads=32, block_threads=64)
    block_64_tuple._source_code(threads=32, block_threads=(8, 8))
    block_128._source_code(threads=32, block_threads=128)
    block_default._source_code(threads=32)
    block_1024._source_code(threads=32, block_threads=1024)

    assert block_64.c_name == block_128.c_name
    assert block_64._private_symbol_digest == block_64_tuple._private_symbol_digest
    assert block_64._private_symbol_digest != block_128._private_symbol_digest
    assert block_default._private_symbol_digest == block_1024._private_symbol_digest
    assert block_64._temp_storage_symbol_names() != (
        block_128._temp_storage_symbol_names()
    )

    # Equivalent spellings are safe when the same specialization is reused.
    block_default._qualify_private_symbols(threads=32, block_threads=1024)
