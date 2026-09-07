# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Final-link evidence for common keys-only Numba-CUDA-MLIR Radix Sort."""

import re
import shutil

import pytest

from ....backends.numba_mlir.support.compile import compile_for_launch

pytestmark = [pytest.mark.gpu, pytest.mark.link, pytest.mark.qualification]

_THREADS = 64
_ITEMS_PER_THREAD = 2


@pytest.mark.evidence_for(
    "group.radix_sort_keys", backend="numba_mlir", evidence="link"
)
def test_common_and_qualified_radix_sort_providers_are_eliminated(
    backend_prerequisite,
):
    if shutil.which("nvdisasm") is None:
        pytest.skip("requires nvdisasm to inspect the final cubin")

    cuda = pytest.importorskip("numba_cuda_mlir.cuda")
    backend_prerequisite(
        "numba_mlir",
        cuda.is_available(),
        "CUDA GPU required for Numba-CUDA-MLIR tests",
    )

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

    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    radix_records = [
        record for record in records if record[0].split("<", 1)[0] == "BlockRadixSort"
    ]
    assert len(radix_records) == 2
    assert {record[1] for record in radix_records} == {"Sort", "SortDescending"}
    assert all(len(record[5]) == 2 for record in radix_records)
    symbols = tuple(record[2] for record in radix_records)
    assert len(set(symbols)) == 1

    sass = kernel.inspect_sass(inspect_key)
    assert re.search(r"\bCALL(?:\.[A-Z0-9_]+)*\b", sass) is None
    for symbol in set(symbols):
        assert (
            re.search(
                rf"(?im)^\s*(?:Function\s*:|\.section\s+\.text\.)"
                rf"[^\n]*{re.escape(symbol)}",
                sass,
            )
            is None
        )


@pytest.mark.evidence_for(
    "group.radix_sort_pairs", backend="numba_mlir", evidence="link"
)
def test_common_and_qualified_radix_sort_pair_providers_are_eliminated(
    backend_prerequisite,
):
    if shutil.which("nvdisasm") is None:
        pytest.skip("requires nvdisasm to inspect the final cubin")

    cuda = pytest.importorskip("numba_cuda_mlir.cuda")
    backend_prerequisite(
        "numba_mlir",
        cuda.is_available(),
        "CUDA GPU required for Numba-CUDA-MLIR tests",
    )
    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    @cuda.jit
    def kernel(key_source, value_source, key_output, value_output):
        tid = cuda.threadIdx.x
        common_keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.uint32)
        common_values = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.float64)
        qualified_keys = numba_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.uint32)
        qualified_values = numba_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.float64)
        for index in range(_ITEMS_PER_THREAD):
            offset = tid * _ITEMS_PER_THREAD + index
            common_keys[index] = key_source[offset]
            common_values[index] = value_source[offset]
            qualified_keys[index] = key_source[offset]
            qualified_values[index] = value_source[offset]

        common_keys_out, common_values_out = coop.radix_sort_pairs(
            coop.this_block(),
            common_keys,
            common_values,
            begin_bit=4,
            end_bit=16,
            descending=True,
        )
        qualified_keys_out, qualified_values_out = numba_coop.radix_sort_pairs(
            numba_coop.this_block(),
            qualified_keys,
            qualified_values,
            begin_bit=4,
            end_bit=16,
            descending=True,
        )
        key_output[tid] = common_keys_out[0] + qualified_keys_out[0]
        value_output[tid] = common_values_out[0] + qualified_values_out[0]

    key_array = types.Array(types.uint32, 1, "C")
    value_array = types.Array(types.float64, 1, "C")
    signature = cuda_typing.signature(
        types.none, key_array, value_array, key_array, value_array
    )
    inspect_key, result = compile_for_launch(kernel, signature, block=_THREADS)
    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    pair_records = [
        record for record in records if record[0].split("<", 1)[0] == "BlockRadixSort"
    ]
    assert {record[1] for record in pair_records} == {"SortDescending"}
    assert all(len(record[5]) == 2 for record in pair_records)
    symbols = {record[2] for record in pair_records}
    sass = kernel.inspect_sass(inspect_key)
    assert re.search(r"\bCALL(?:\.[A-Z0-9_]+)*\b", sass) is None
    for symbol in symbols:
        assert (
            re.search(
                rf"(?im)^\s*(?:Function\s*:|\.section\s+\.text\.)"
                rf"[^\n]*{re.escape(symbol)}",
                sass,
            )
            is None
        )
