# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
import shutil

import pytest

from ....backends.numba_mlir.support.compile import compile_for_launch

pytestmark = [pytest.mark.gpu, pytest.mark.link, pytest.mark.qualification]

_THREADS = 32
_ITEMS_PER_THREAD = 2


@pytest.mark.evidence_for(
    "group.adjacent_difference", backend="numba_mlir", evidence="link"
)
@pytest.mark.evidence_for("group.discontinuity", backend="numba_mlir", evidence="link")
def test_common_and_qualified_adjacent_discontinuity_providers_are_eliminated(
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
    def kernel(source, output, valid_items, predecessor, successor):
        tid = cuda.threadIdx.x
        common_items = coop.ThreadData(_ITEMS_PER_THREAD, dtype=types.int32)
        qualified_items = numba_coop.ThreadData(
            _ITEMS_PER_THREAD,
            dtype=types.int32,
        )
        for index in range(_ITEMS_PER_THREAD):
            value = source[tid * _ITEMS_PER_THREAD + index]
            common_items[index] = value
            qualified_items[index] = value

        common_group = coop.this_block()
        common_storage = coop.TempStorage()
        common_left = coop.adjacent_difference(
            common_group,
            common_items,
            valid_items=valid_items,
            tile_predecessor_item=predecessor,
            temp_storage=common_storage,
        )
        common_right = coop.adjacent_difference(
            common_group,
            common_items,
            direction="right",
            tile_successor_item=successor,
            temp_storage=common_storage,
        )
        common_heads = coop.discontinuity(
            common_group,
            common_items,
            mode="heads",
            tile_predecessor_item=predecessor,
            temp_storage=common_storage,
        )
        common_tails = coop.discontinuity(
            common_group,
            common_items,
            mode="tails",
            tile_successor_item=successor,
            temp_storage=common_storage,
        )

        qualified_group = numba_coop.this_block()
        qualified_storage = numba_coop.TempStorage()
        qualified_left = numba_coop.adjacent_difference(
            qualified_group,
            qualified_items,
            valid_items=valid_items,
            tile_predecessor_item=predecessor,
            temp_storage=qualified_storage,
        )
        qualified_right = numba_coop.adjacent_difference(
            qualified_group,
            qualified_items,
            direction="right",
            tile_successor_item=successor,
            temp_storage=qualified_storage,
        )
        qualified_heads = numba_coop.discontinuity(
            qualified_group,
            qualified_items,
            mode="heads",
            tile_predecessor_item=predecessor,
            temp_storage=qualified_storage,
        )
        qualified_tails = numba_coop.discontinuity(
            qualified_group,
            qualified_items,
            mode="tails",
            tile_successor_item=successor,
            temp_storage=qualified_storage,
        )
        pair_heads, pair_tails = numba_coop.discontinuity(
            qualified_group,
            qualified_items,
            mode="heads_and_tails",
            tile_predecessor_item=predecessor,
            tile_successor_item=successor,
            temp_storage=qualified_storage,
        )

        output[tid] = (
            common_left[0]
            + common_right[0]
            + common_heads[0]
            + common_tails[0]
            + qualified_left[0]
            + qualified_right[0]
            + qualified_heads[0]
            + qualified_tails[0]
            + pair_heads[0]
            + pair_tails[0]
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
    segmentation_records = [
        record
        for record in records
        if record[0].split("<", 1)[0]
        in {"BlockAdjacentDifference", "BlockDiscontinuity"}
    ]
    assert {
        (record[0].split("<", 1)[0], record[1]) for record in segmentation_records
    } == {
        ("BlockAdjacentDifference", "SubtractLeftPartialTile"),
        ("BlockAdjacentDifference", "SubtractRight"),
        ("BlockDiscontinuity", "FlagHeads"),
        ("BlockDiscontinuity", "FlagTails"),
        ("BlockDiscontinuity", "FlagHeadsAndTails"),
    }
    assert len(segmentation_records) == 5
    symbols = {record[2] for record in segmentation_records}
    assert len(symbols) == 2

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


def test_qualified_complex128_segmentation_providers_are_eliminated(
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

    @cuda.jit(device=True)
    def subtract(left, right):
        return left - right

    @cuda.jit(device=True)
    def not_equal(left, right):
        return left.real != right.real or left.imag != right.imag

    @cuda.jit
    def kernel(source, value_output, flag_output, predecessor, successor):
        tid = cuda.threadIdx.x
        items = numba_coop.ThreadData(2, dtype=types.complex128)
        items[0] = source[tid * 2]
        items[1] = source[tid * 2 + 1]
        group = numba_coop.this_block()
        storage = numba_coop.TempStorage()
        left = numba_coop.adjacent_difference(
            group,
            items,
            tile_predecessor_item=predecessor,
            temp_storage=storage,
            difference_op=subtract,
        )
        right = numba_coop.adjacent_difference(
            group,
            items,
            direction="right",
            tile_successor_item=successor,
            temp_storage=storage,
            difference_op=subtract,
        )
        heads = numba_coop.discontinuity(
            group,
            items,
            mode="heads",
            tile_predecessor_item=predecessor,
            temp_storage=storage,
            flag_op=not_equal,
        )
        tails = numba_coop.discontinuity(
            group,
            items,
            mode="tails",
            tile_successor_item=successor,
            temp_storage=storage,
            flag_op=not_equal,
        )
        pair_heads, pair_tails = numba_coop.discontinuity(
            group,
            items,
            mode="heads_and_tails",
            tile_predecessor_item=predecessor,
            tile_successor_item=successor,
            temp_storage=storage,
            flag_op=not_equal,
        )
        value_output[tid] = left[0] + right[0]
        flag_output[tid] = heads[0] + tails[0] + pair_heads[0] + pair_tails[0]

    complex_array = types.Array(types.complex128, 1, "C")
    flag_array = types.Array(types.int32, 1, "C")
    signature = cuda_typing.signature(
        types.none,
        complex_array,
        complex_array,
        flag_array,
        types.complex128,
        types.complex128,
    )
    inspect_key, result = compile_for_launch(kernel, signature, block=_THREADS)
    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    segmentation_records = [
        record
        for record in records
        if record[0].split("<", 1)[0]
        in {"BlockAdjacentDifference", "BlockDiscontinuity"}
    ]
    assert {
        (record[0].split("<", 1)[0], record[1]) for record in segmentation_records
    } == {
        ("BlockAdjacentDifference", "SubtractLeft"),
        ("BlockAdjacentDifference", "SubtractRight"),
        ("BlockDiscontinuity", "FlagHeads"),
        ("BlockDiscontinuity", "FlagTails"),
        ("BlockDiscontinuity", "FlagHeadsAndTails"),
    }
    assert len(segmentation_records) == 5
    symbols = {record[2] for record in segmentation_records}
    assert len(symbols) == 2

    # User-defined Python operators may remain as device calls. The provider
    # qualification boundary is that the generated CUB entrypoints themselves
    # have disappeared from the final cubin.
    sass = kernel.inspect_sass(inspect_key)
    for symbol in symbols:
        assert (
            re.search(
                rf"(?im)^\s*(?:Function\s*:|\.section\s+\.text\.)"
                rf"[^\n]*{re.escape(symbol)}",
                sass,
            )
            is None
        )
