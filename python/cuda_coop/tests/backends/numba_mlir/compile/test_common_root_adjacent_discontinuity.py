# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from ..support.compile import compile_for_launch

pytestmark = pytest.mark.gpu

_THREADS = 32
_ITEMS_PER_THREAD = 2


@pytest.mark.evidence_for(
    "group.adjacent_difference", backend="numba_mlir", evidence="compile"
)
@pytest.mark.evidence_for(
    "group.discontinuity", backend="numba_mlir", evidence="compile"
)
def test_common_and_qualified_adjacent_discontinuity_compile_to_same_provider_plan(
    numba_mlir_cuda_available,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

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
            common_items[0]
            + common_left[0]
            + common_right[0]
            + common_heads[0]
            + common_tails[0]
            + qualified_items[0]
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

    cubin = result.metadata.get("cubin")
    assert isinstance(cubin, bytes)
    assert cubin.startswith(b"\x7fELF")
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
    assert kernel.get_metadata(inspect_key)["cubin"] == cubin


def test_common_scalar_comparison_rejection_leaves_qualified_compile_usable(
    numba_mlir_cuda_available,
):
    del numba_mlir_cuda_available
    cuda = pytest.importorskip("numba_cuda_mlir.cuda")

    from numba_cuda_mlir import types
    from numba_cuda_mlir.numba_cuda import typing as cuda_typing

    import cuda.coop.numba_mlir as numba_coop
    from cuda import coop

    @cuda.jit
    def common_adjacent_difference(source, output):
        tid = cuda.threadIdx.x
        output[tid] = coop.adjacent_difference(
            coop.this_block(),
            source[tid],
        )

    @cuda.jit
    def common_discontinuity(source, output):
        tid = cuda.threadIdx.x
        output[tid] = coop.discontinuity(
            coop.this_block(),
            source[tid],
        )

    @cuda.jit
    def qualified_comparisons(source, output):
        tid = cuda.threadIdx.x
        value = source[tid]
        group = numba_coop.this_block()
        difference = numba_coop.adjacent_difference(group, value)
        flag = numba_coop.discontinuity(group, value)
        output[tid] = difference + flag

    array_type = types.Array(types.int32, 1, "C")
    signature = cuda_typing.signature(types.none, array_type, array_type)
    rejected = (
        (common_adjacent_difference, "adjacent_difference"),
        (common_discontinuity, "discontinuity"),
    )
    for dispatcher, operation_name in rejected:
        with pytest.raises(
            TypeError,
            match=rf"cuda\.coop\.{operation_name} requires a fixed-size ThreadData",
        ):
            compile_for_launch(dispatcher, signature, block=_THREADS)

    inspect_key, result = compile_for_launch(
        qualified_comparisons,
        signature,
        block=_THREADS,
    )
    cubin = result.metadata.get("cubin")
    assert isinstance(cubin, bytes)
    assert cubin.startswith(b"\x7fELF")
    assert qualified_comparisons.get_metadata(inspect_key)["cubin"] == cubin
