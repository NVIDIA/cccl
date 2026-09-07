# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from ..support.compile import compile_for_launch

pytestmark = pytest.mark.gpu

_THREADS = 64
_ITEMS_PER_THREAD = 2


@pytest.mark.evidence_for(
    "group.radix_sort_keys", backend="numba_mlir", evidence="compile"
)
def test_common_and_qualified_radix_sort_compile_to_shared_provider_plans(
    numba_mlir_cuda_available,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    @cuda.jit
    def kernel(source, output, begin_only, subrange_begin, subrange_end):
        tid = cuda.threadIdx.x
        common_keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
        qualified_keys = numba_coop.ThreadData(
            _ITEMS_PER_THREAD,
            dtype=types.int32,
        )
        for index in range(_ITEMS_PER_THREAD):
            value = source[tid * _ITEMS_PER_THREAD + index]
            common_keys[index] = value
            qualified_keys[index] = value

        common_group = coop.this_block()
        qualified_group = numba_coop.this_block()
        common_storage = coop.TempStorage()
        qualified_storage = numba_coop.TempStorage()
        common_full = coop.radix_sort_keys(
            common_group,
            common_keys,
            temp_storage=common_storage,
        )
        qualified_full = numba_coop.radix_sort_keys(
            qualified_group,
            qualified_keys,
            temp_storage=qualified_storage,
        )
        common_begin_only = coop.radix_sort_keys(
            common_group,
            common_keys,
            begin_bit=begin_only,
            descending=True,
        )
        qualified_begin_only = numba_coop.radix_sort_keys(
            qualified_group,
            qualified_keys,
            begin_bit=begin_only,
            descending=True,
        )
        common_subrange = coop.radix_sort_keys(
            common_group,
            common_keys,
            begin_bit=subrange_begin,
            end_bit=subrange_end,
        )
        qualified_subrange = numba_coop.radix_sort_keys(
            qualified_group,
            qualified_keys,
            begin_bit=subrange_begin,
            end_bit=subrange_end,
        )
        output[tid] = (
            common_full[0]
            + qualified_full[0]
            + common_begin_only[0]
            + qualified_begin_only[0]
            + common_subrange[0]
            + qualified_subrange[0]
        )

    array_type = types.Array(types.int32, 1, "C")
    signature = cuda_typing.signature(
        types.none,
        array_type,
        array_type,
        types.int32,
        types.int32,
        types.int32,
    )
    inspect_key, result = compile_for_launch(kernel, signature, block=_THREADS)

    cubin = result.metadata.get("cubin")
    assert isinstance(cubin, bytes)
    assert cubin.startswith(b"\x7fELF")
    link_plan = result.metadata["link_plan"]
    assert link_plan.has_external_link_items
    assert link_plan.has_ltoir_link_items
    assert result.metadata["linked_external_link_items"]

    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    radix_records = [
        record for record in records if record[0].split("<", 1)[0] == "BlockRadixSort"
    ]
    assert len(radix_records) == 2
    assert {record[1] for record in radix_records} == {"Sort", "SortDescending"}
    assert all(len(record[5]) == 2 for record in radix_records)
    assert len({record[2] for record in radix_records}) == 1
    assert kernel.get_metadata(inspect_key)["cubin"] == cubin


def test_qualified_scalar_radix_sort_compiles_with_scalar_results(
    numba_mlir_cuda_available,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    import cuda.coop.numba_mlir as numba_coop

    @cuda.jit
    def kernel(source, output):
        tid = cuda.threadIdx.x
        value = source[tid]
        ascending = numba_coop.radix_sort_keys(
            numba_coop.this_block(),
            value,
        )
        descending = numba_coop.radix_sort_keys(
            numba_coop.this_block(),
            value,
            begin_bit=8,
            descending=True,
        )
        output[tid] = ascending + descending

    array_type = types.Array(types.int32, 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type)
    _inspect_key, result = compile_for_launch(kernel, signature, block=_THREADS)

    cubin = result.metadata.get("cubin")
    assert isinstance(cubin, bytes)
    assert cubin.startswith(b"\x7fELF")
    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    assert {
        record[1]
        for record in records
        if record[0].split("<", 1)[0] == "BlockRadixSort"
    } >= {"Sort", "SortDescending"}


@pytest.mark.evidence_for(
    "group.radix_sort_pairs",
    backend="numba_mlir",
    evidence="compile",
)
def test_common_radix_sort_pairs_compile_with_distinct_value_dtype(
    numba_mlir_cuda_available,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    from cuda import coop

    @cuda.jit
    def kernel(keys, values, key_output, value_output):
        tid = cuda.threadIdx.x
        thread_keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.uint32)
        thread_values = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.float32)
        for item in range(_ITEMS_PER_THREAD):
            index = tid * _ITEMS_PER_THREAD + item
            thread_keys[item] = keys[index]
            thread_values[item] = values[index]
        key, value = coop.radix_sort_pairs(
            coop.this_block(),
            thread_keys,
            thread_values,
            begin_bit=4,
            end_bit=10,
        )
        key_output[tid] = key[0]
        value_output[tid] = value[0]

    key_array_type = types.Array(types.uint32, 1, "C")
    value_array_type = types.Array(types.float32, 1, "C")
    signature = cuda_typing.signature(
        types.none,
        key_array_type,
        value_array_type,
        key_array_type,
        value_array_type,
    )
    _inspect_key, result = compile_for_launch(kernel, signature, block=_THREADS)

    cubin = result.metadata.get("cubin")
    assert isinstance(cubin, bytes)
    assert cubin.startswith(b"\x7fELF")
    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    radix_records = [
        record for record in records if record[0].split("<", 1)[0] == "BlockRadixSort"
    ]
    assert {record[1] for record in radix_records} == {"Sort"}
    assert all(len(record[5]) == 2 for record in radix_records)
