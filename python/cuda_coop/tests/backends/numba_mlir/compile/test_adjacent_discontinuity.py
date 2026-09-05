# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import warnings

import pytest

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.compile]

_THREADS = 32
_ITEMS_PER_THREAD = 2


def _compile_for_launch(dispatcher, signature):
    from numba_cuda_mlir import descriptor
    from numba_cuda_mlir.numba_cuda.core.errors import NumbaPerformanceWarning

    launch_key = descriptor._launch_config_key(
        {
            "grid": (1, 1, 1),
            "block": (_THREADS, 1, 1),
            "sharedmem": 0,
            "cluster": None,
        }
    )
    compiler = getattr(dispatcher, "_compile_launch_config_signature", None)
    if not callable(compiler):
        raise RuntimeError(
            "Numba-CUDA-MLIR runtime lacks launch-qualified compile support"
        )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=(
                "Persistent disk cache is disabled for "
                "launch-config-specialized compiles"
            ),
            category=NumbaPerformanceWarning,
        )
        result = compiler(signature, launch_key)
    matching = tuple(
        key
        for key in dispatcher.signatures
        if getattr(key, "launch_config_key", None) == launch_key
    )
    assert len(matching) == 1
    return matching[0], result


def test_common_and_qualified_comparison_cohort_compiles_to_cub():
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

        common_left = coop.adjacent_difference(
            coop.this_block(),
            common_items,
            valid_items=valid_items,
            tile_predecessor_item=predecessor,
            temp_storage=coop.TempStorage(),
        )
        common_heads = coop.discontinuity(
            coop.this_block(),
            common_items,
            tile_predecessor_item=predecessor,
            temp_storage=coop.TempStorage(),
        )

        qualified_group = numba_coop.this_block()
        qualified_storage = numba_coop.TempStorage()
        qualified_right = numba_coop.adjacent_difference(
            qualified_group,
            qualified_items,
            direction="right",
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
            + common_heads[0]
            + qualified_items[0]
            + qualified_right[0]
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
    inspect_key, result = _compile_for_launch(kernel, signature)

    cubin = result.metadata.get("cubin")
    assert isinstance(cubin, bytes)
    assert cubin.startswith(b"\x7fELF")
    records = result.metadata["__cuda_coop_numba_mlir_materialized_specializations__"]
    comparison_records = [
        record
        for record in records
        if record[0].split("<", 1)[0]
        in {"BlockAdjacentDifference", "BlockDiscontinuity"}
    ]
    assert {
        (record[0].split("<", 1)[0], record[1]) for record in comparison_records
    } == {
        ("BlockAdjacentDifference", "SubtractLeftPartialTile"),
        ("BlockAdjacentDifference", "SubtractRight"),
        ("BlockDiscontinuity", "FlagHeads"),
        ("BlockDiscontinuity", "FlagHeadsAndTails"),
    }
    assert len(comparison_records) == 4
    assert kernel.get_metadata(inspect_key)["cubin"] == cubin
