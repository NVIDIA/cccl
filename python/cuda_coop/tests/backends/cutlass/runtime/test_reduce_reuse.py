# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Reuse CUB reduction scratch across calls and runtime loop iterations."""

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from tests.backends.cutlass.support import device_array, values_for

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize(
    "api", (coop, cutlass_coop), ids=("common", "qualified")
)
@pytest.mark.parametrize("width", (8, 32, 128))
def test_full_reductions_reuse_scratch_in_a_loop(api, width):
    @cute.kernel
    def kernel(
        source: cute.Pointer, observed: cute.Pointer, iterations: cutlass.Int32
    ):
        thread = cute.arch.thread_idx()[0]
        inputs = cute.make_tensor(source, cute.make_layout(128))
        outputs = cute.make_tensor(observed, cute.make_layout(128 // width))
        if cutlass.const_expr(width == 128):
            group = api.this_block()
        elif cutlass.const_expr(width == 32):
            group = api.this_warp()
        else:
            group = api.this_warp().group_by(width)
        accumulated = cutlass.Int32(0)
        for iteration in range(iterations):
            total = api.sum(group, inputs[thread] + iteration)
            largest = api.reduce(group, inputs[thread], binary_op="max")
            if thread % width == 0:
                accumulated += total + largest
        if thread % width == 0:
            outputs[thread // width] = accumulated

    @cute.jit
    def launch(
        source: cute.Pointer, observed: cute.Pointer, iterations: cutlass.Int32
    ):
        kernel(source, observed, iterations).launch(grid=1, block=128)

    iterations = 5
    source = values_for(np.int32, 128, shift=79)
    observed = np.zeros(128 // width, dtype=np.int32)
    rows = source.reshape(-1, width)
    expected = iterations * (rows.sum(axis=1) + rows.max(axis=1)) + width * sum(
        range(iterations)
    )
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out, iterations)
    np.testing.assert_array_equal(observed, expected)


@pytest.mark.parametrize(
    "api", (coop, cutlass_coop), ids=("common", "qualified")
)
@pytest.mark.parametrize("items_per_thread", (1, 4))
@pytest.mark.parametrize(
    "sharing,auto_sync,size_in_bytes",
    (
        ("shared", True, None),
        ("shared", False, None),
        ("exclusive", False, None),
        ("shared", True, 64 * 1024),
    ),
    ids=("shared-auto", "shared-manual", "exclusive-manual", "dynamic"),
)
def test_explicit_scratch_reused_by_load_reduce_store(
    api, items_per_thread, sharing, auto_sync, size_in_bytes
):
    tile_size = 128 * items_per_thread

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        preserved: cute.Pointer,
        observed: cute.Pointer,
        tiles: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        block = api.this_block()
        thread = block.rank()
        output = cute.make_tensor(observed, cute.make_layout(4))
        scratch = api.TempStorage(
            size_in_bytes, sharing=sharing, auto_sync=auto_sync
        )
        payload = api.ThreadData(items_per_thread)
        for tile in range(tiles):
            api.load(
                block,
                source,
                payload,
                offset=tile * tile_size,
                algorithm="transpose",
                temp_storage=scratch,
            )
            if cutlass.const_expr(not auto_sync):
                scratch.sync()
            total = api.sum(block, payload, temp_storage=scratch)
            if cutlass.const_expr(not auto_sync):
                scratch.sync()
            maximum = api.reduce(
                block, payload, binary_op="max", temp_storage=scratch
            )
            if cutlass.const_expr(not auto_sync):
                scratch.sync()
            api.store(
                block,
                preserved,
                payload,
                offset=tile * tile_size,
                algorithm="transpose",
                temp_storage=scratch,
            )
            if cutlass.const_expr(not auto_sync):
                scratch.sync()
            if thread == 0:
                output[tile * 2] = total
                output[tile * 2 + 1] = maximum

    @cute.jit
    def launch(
        source: cute.Pointer,
        preserved: cute.Pointer,
        observed: cute.Pointer,
        tiles: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(source, preserved, observed, tiles, items_per_thread).launch(
            grid=1, block=128
        )

    source = values_for(np.int32, 2 * tile_size, shift=97)
    preserved = np.zeros_like(source)
    observed = np.zeros(4, dtype=np.int32)
    with (
        device_array(source) as src,
        device_array(preserved) as copied,
        device_array(observed) as out,
    ):
        launch(src, copied, out, 2, items_per_thread)
    tiles = source.reshape(2, tile_size)
    np.testing.assert_array_equal(observed[::2], tiles.sum(axis=1))
    np.testing.assert_array_equal(observed[1::2], tiles.max(axis=1))
    np.testing.assert_array_equal(preserved, source)
