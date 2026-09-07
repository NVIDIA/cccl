# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
import shutil

import pytest

from ....backends.numba_mlir.support.compile import compile_for_launch

pytestmark = [pytest.mark.gpu, pytest.mark.link, pytest.mark.qualification]

_THREADS = 64
_ITEMS_PER_THREAD = 2


@pytest.mark.evidence_for(
    "group.merge_sort_keys", backend="numba_mlir", evidence="link"
)
def test_common_and_qualified_merge_sort_providers_are_eliminated(
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
    def kernel(source, output):
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

        common_block = coop.this_block()
        qualified_block = numba_coop.this_block()
        common_block_full = coop.merge_sort_keys(common_block, common_keys)
        qualified_block_full = numba_coop.merge_sort_keys(
            qualified_block,
            qualified_keys,
        )
        common_block_partial = coop.merge_sort_keys(
            common_block,
            common_keys,
            descending=True,
            valid_items=117,
            oob_default=-2_147_483_648,
        )
        qualified_block_partial = numba_coop.merge_sort_keys(
            qualified_block,
            qualified_keys,
            descending=True,
            valid_items=117,
            oob_default=-2_147_483_648,
        )

        common_warp = coop.this_warp()
        qualified_warp = numba_coop.this_warp()
        common_warp_full = coop.merge_sort_keys(
            common_warp,
            common_keys,
            descending=True,
        )
        qualified_warp_full = numba_coop.merge_sort_keys(
            qualified_warp,
            qualified_keys,
            descending=True,
        )
        common_warp_partial = coop.merge_sort_keys(
            common_warp,
            common_keys,
            valid_items=53,
            oob_default=2_147_483_647,
        )
        qualified_warp_partial = numba_coop.merge_sort_keys(
            qualified_warp,
            qualified_keys,
            valid_items=53,
            oob_default=2_147_483_647,
        )
        output[tid] = (
            common_block_full[0]
            + qualified_block_full[0]
            + common_block_partial[0]
            + qualified_block_partial[0]
            + common_warp_full[0]
            + qualified_warp_full[0]
            + common_warp_partial[0]
            + qualified_warp_partial[0]
        )

    array_type = types.Array(types.int32, 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type)
    inspect_key, result = compile_for_launch(kernel, signature, block=_THREADS)

    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    merge_sort_records = [
        record
        for record in records
        if record[0].split("<", 1)[0] in {"BlockMergeSort", "WarpMergeSort"}
    ]
    assert len(merge_sort_records) == 4
    assert {record[0].split("<", 1)[0] for record in merge_sort_records} == {
        "BlockMergeSort",
        "WarpMergeSort",
    }
    assert {record[1] for record in merge_sort_records} == {"Sort"}
    symbols = tuple(record[2] for record in merge_sort_records)
    assert len(set(symbols)) == 2
    for class_name in ("BlockMergeSort", "WarpMergeSort"):
        class_records = [
            record
            for record in merge_sort_records
            if record[0].split("<", 1)[0] == class_name
        ]
        assert len({record[2] for record in class_records}) == 1
        assert len({repr(record[5]) for record in class_records}) == 2

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
    "group.merge_sort_pairs", backend="numba_mlir", evidence="link"
)
def test_common_and_qualified_merge_sort_pair_providers_are_eliminated(
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
        common_keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
        common_values = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.float32)
        qualified_keys = numba_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
        qualified_values = numba_coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.float32)
        for index in range(_ITEMS_PER_THREAD):
            offset = tid * _ITEMS_PER_THREAD + index
            common_keys[index] = key_source[offset]
            common_values[index] = value_source[offset]
            qualified_keys[index] = key_source[offset]
            qualified_values[index] = value_source[offset]

        common_block_keys, common_block_values = coop.merge_sort_pairs(
            coop.this_block(), common_keys, common_values
        )
        qualified_block_keys, qualified_block_values = numba_coop.merge_sort_pairs(
            numba_coop.this_block(), qualified_keys, qualified_values
        )
        common_warp_keys, common_warp_values = coop.merge_sort_pairs(
            coop.this_warp(),
            common_keys,
            common_values,
            descending=True,
            valid_items=53,
            oob_default=2_147_483_647,
        )
        qualified_warp_keys, qualified_warp_values = numba_coop.merge_sort_pairs(
            numba_coop.this_warp(),
            qualified_keys,
            qualified_values,
            descending=True,
            valid_items=53,
            oob_default=2_147_483_647,
        )
        key_output[tid] = (
            common_block_keys[0]
            + qualified_block_keys[0]
            + common_warp_keys[0]
            + qualified_warp_keys[0]
        )
        value_output[tid] = (
            common_block_values[0]
            + qualified_block_values[0]
            + common_warp_values[0]
            + qualified_warp_values[0]
        )

    key_array = types.Array(types.int32, 1, "C")
    value_array = types.Array(types.float32, 1, "C")
    signature = cuda_typing.signature(
        types.none, key_array, value_array, key_array, value_array
    )
    inspect_key, result = compile_for_launch(kernel, signature, block=_THREADS)
    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    pair_records = [
        record
        for record in records
        if record[0].split("<", 1)[0] in {"BlockMergeSort", "WarpMergeSort"}
    ]
    assert {record[0].split("<", 1)[0] for record in pair_records} == {
        "BlockMergeSort",
        "WarpMergeSort",
    }
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
