# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from ..support.compile import compile_for_launch

pytestmark = pytest.mark.gpu

_THREADS = 64
_ITEMS_PER_THREAD = 2


@pytest.mark.evidence_for(
    "group.merge_sort_keys", backend="numba_mlir", evidence="compile"
)
def test_common_and_qualified_merge_sort_compile_to_shared_provider_plans(
    numba_mlir_cuda_available,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    @cuda.jit
    def kernel(source, output):
        tid = cuda.threadIdx.x
        common_full_keys = coop.ThreadData(_ITEMS_PER_THREAD)
        qualified_full_keys = numba_coop.ThreadData(_ITEMS_PER_THREAD)
        common_partial_keys = coop.ThreadData(_ITEMS_PER_THREAD)
        qualified_partial_keys = numba_coop.ThreadData(_ITEMS_PER_THREAD)
        for index in range(_ITEMS_PER_THREAD):
            value = source[tid * _ITEMS_PER_THREAD + index]
            common_full_keys[index] = value
            qualified_full_keys[index] = value
            common_partial_keys[index] = value
            qualified_partial_keys[index] = value

        common_block = coop.this_block()
        qualified_block = numba_coop.this_block()
        common_block_full = coop.merge_sort_keys(common_block, common_full_keys)
        qualified_block_full = numba_coop.merge_sort_keys(
            qualified_block,
            qualified_full_keys,
        )
        common_block_partial = coop.merge_sort_keys(
            common_block,
            common_partial_keys,
            descending=True,
            valid_items=117,
            oob_default=-2_147_483_648,
        )
        qualified_block_partial = numba_coop.merge_sort_keys(
            qualified_block,
            qualified_partial_keys,
            descending=True,
            valid_items=117,
            oob_default=-2_147_483_648,
        )

        common_warp = coop.this_warp()
        qualified_warp = numba_coop.this_warp()
        common_warp_full = coop.merge_sort_keys(
            common_warp,
            common_full_keys,
            descending=True,
        )
        qualified_warp_full = numba_coop.merge_sort_keys(
            qualified_warp,
            qualified_full_keys,
            descending=True,
        )
        common_warp_partial = coop.merge_sort_keys(
            common_warp,
            common_partial_keys,
            valid_items=53,
            oob_default=2_147_483_647,
        )
        qualified_warp_partial = numba_coop.merge_sort_keys(
            qualified_warp,
            qualified_partial_keys,
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

    cubin = result.metadata.get("cubin")
    assert isinstance(cubin, bytes)
    assert cubin.startswith(b"\x7fELF")
    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    merge_sort_records = {
        class_name: [
            record for record in records if record[0].split("<", 1)[0] == class_name
        ]
        for class_name in ("BlockMergeSort", "WarpMergeSort")
    }
    assert all(len(class_records) == 2 for class_records in merge_sort_records.values())
    assert {
        record[1]
        for class_records in merge_sort_records.values()
        for record in class_records
    } == {"Sort"}
    assert all(
        len({record[2] for record in class_records}) == 1
        and len({repr(record[5]) for record in class_records}) == 2
        for class_records in merge_sort_records.values()
    )
    assert kernel.get_metadata(inspect_key)["cubin"] == cubin


def test_qualified_scalar_merge_sort_compiles_with_scalar_results(
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
        block_result = numba_coop.merge_sort_keys(
            numba_coop.this_block(),
            value,
        )
        warp_result = numba_coop.merge_sort_keys(
            numba_coop.this_warp(),
            value,
            descending=True,
            valid_items=31,
            oob_default=-2_147_483_648,
        )
        output[tid] = block_result + warp_result

    array_type = types.Array(types.int32, 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type)
    _inspect_key, result = compile_for_launch(kernel, signature, block=_THREADS)

    cubin = result.metadata.get("cubin")
    assert isinstance(cubin, bytes)
    assert cubin.startswith(b"\x7fELF")
    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    assert {(record[0].split("<", 1)[0], record[1]) for record in records} >= {
        ("BlockMergeSort", "Sort"),
        ("WarpMergeSort", "Sort"),
    }


@pytest.mark.evidence_for(
    "group.merge_sort_pairs",
    backend="numba_mlir",
    evidence="compile",
)
def test_common_merge_sort_pairs_compile_for_block_and_physical_warp(
    numba_mlir_cuda_available,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    import cuda.coop.numba_mlir as _numba_coop  # noqa: F401
    from cuda import coop

    @cuda.jit
    def kernel(keys, values, block_keys, block_values, warp_keys, warp_values):
        tid = cuda.threadIdx.x
        thread_keys = coop.ThreadData(_ITEMS_PER_THREAD)
        thread_values = coop.ThreadData(_ITEMS_PER_THREAD)
        for item in range(_ITEMS_PER_THREAD):
            index = tid * _ITEMS_PER_THREAD + item
            thread_keys[item] = keys[index]
            thread_values[item] = values[index]
        block_key, block_value = coop.merge_sort_pairs(
            coop.this_block(),
            thread_keys,
            thread_values,
        )
        warp_key, warp_value = coop.merge_sort_pairs(
            coop.this_warp(),
            thread_keys,
            thread_values,
            descending=True,
        )
        block_keys[tid] = block_key[0]
        block_values[tid] = block_value[0]
        warp_keys[tid] = warp_key[0]
        warp_values[tid] = warp_value[0]

    key_array_type = types.Array(types.int32, 1, "C")
    value_array_type = types.Array(types.float32, 1, "C")
    signature = cuda_typing.signature(
        types.none,
        key_array_type,
        value_array_type,
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
    assert {(record[0].split("<", 1)[0], record[1]) for record in records} >= {
        ("BlockMergeSort", "Sort"),
        ("WarpMergeSort", "Sort"),
    }


def test_common_merge_sort_rejects_nonportable_key_dtype_with_actionable_error(
    numba_mlir_cuda_available,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    from cuda import coop
    from cuda.coop.numba_mlir._single_phase_rewrites import (
        CoopSinglePhaseRewriteError,
    )

    @cuda.jit
    def kernel(source, output):
        tid = cuda.threadIdx.x
        keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.float32)
        keys[0] = source[tid * _ITEMS_PER_THREAD]
        keys[1] = source[tid * _ITEMS_PER_THREAD + 1]
        output[tid] = coop.merge_sort_keys(coop.this_warp(), keys)[0]

    array_type = types.Array(types.float32, 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type)
    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=(
            r"cuda\.coop\.merge_sort_keys common V1 supports key dtypes int32, "
            r"uint32, int64, uint64"
        ),
    ):
        compile_for_launch(kernel, signature, block=_THREADS)


@pytest.mark.parametrize(
    ("sentinel", "diagnostic"),
    [
        (
            1.5,
            r"oob_default must have the same integer dtype as keys \(int32\); got float",
        ),
        (
            2_147_483_648,
            r"oob_default=2147483648 is not representable in keys dtype int32",
        ),
    ],
)
def test_common_merge_sort_rejects_lossy_static_sentinel_before_materialization(
    numba_mlir_cuda_available,
    sentinel,
    diagnostic,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    from cuda import coop
    from cuda.coop.numba_mlir._single_phase_rewrites import (
        CoopSinglePhaseRewriteError,
    )

    @cuda.jit
    def kernel(source, output):
        tid = cuda.threadIdx.x
        keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
        keys[0] = source[tid * _ITEMS_PER_THREAD]
        keys[1] = source[tid * _ITEMS_PER_THREAD + 1]
        output[tid] = coop.merge_sort_keys(
            coop.this_warp(),
            keys,
            valid_items=63,
            oob_default=sentinel,
        )[0]

    array_type = types.Array(types.int32, 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type)
    with pytest.raises(CoopSinglePhaseRewriteError, match=diagnostic):
        compile_for_launch(kernel, signature, block=_THREADS)


@pytest.mark.parametrize(
    ("sentinel_dtype_name", "diagnostic"),
    [
        ("int32", None),
        (
            "int64",
            r"oob_default must have the same integer dtype as keys \(int32\); got int64",
        ),
    ],
)
def test_common_merge_sort_validates_dynamic_compiler_integer_sentinel_dtype(
    numba_mlir_cuda_available,
    sentinel_dtype_name,
    diagnostic,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    from cuda import coop
    from cuda.coop.numba_mlir._single_phase_rewrites import (
        CoopSinglePhaseRewriteError,
    )

    @cuda.jit
    def kernel(source, sentinels, output):
        tid = cuda.threadIdx.x
        keys = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
        keys[0] = source[tid * _ITEMS_PER_THREAD]
        keys[1] = source[tid * _ITEMS_PER_THREAD + 1]
        output[tid] = coop.merge_sort_keys(
            coop.this_warp(),
            keys,
            valid_items=63,
            oob_default=sentinels[0],
        )[0]

    int32_array = types.Array(types.int32, 1, "C")
    sentinel_array = types.Array(getattr(types, sentinel_dtype_name), 1, "C")
    signature = cuda_typing.signature(
        types.none,
        int32_array,
        sentinel_array,
        int32_array,
    )
    if diagnostic is not None:
        with pytest.raises(CoopSinglePhaseRewriteError, match=diagnostic):
            compile_for_launch(kernel, signature, block=_THREADS)
        return

    _inspect_key, result = compile_for_launch(kernel, signature, block=_THREADS)
    cubin = result.metadata.get("cubin")
    assert isinstance(cubin, bytes)
    assert cubin.startswith(b"\x7fELF")
