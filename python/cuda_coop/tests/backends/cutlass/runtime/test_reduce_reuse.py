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


@pytest.mark.parametrize(
    "api", (coop, cutlass_coop), ids=("common", "qualified")
)
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize(
    "width,physical",
    [(1, False), (8, False), (17, False), (31, False), (32, False), (32, True)],
    ids=[
        "logical1",
        "logical8",
        "logical17",
        "logical31",
        "logical32",
        "physical",
    ],
)
def test_warp_payload_reductions_reuse_scratch(
    api, items_per_thread, width, physical
):
    """Reduce every item, preserving payloads across independent warp groups."""

    tile_size = 64 * items_per_thread
    groups_per_warp = 32 // width
    group_count = 2 * groups_per_warp

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        preserved: cute.Pointer,
        observed: cute.Pointer,
        iterations: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        thread = api.this_block().rank()
        inputs = cute.make_tensor(source, cute.make_layout(tile_size))
        copied = cute.make_tensor(preserved, cute.make_layout(tile_size))
        outputs = cute.make_tensor(observed, cute.make_layout(group_count * 2))
        group = api.this_warp()
        if cutlass.const_expr(not physical):
            group = group.group_by(width, exhaustive=False)
        accumulated_sum = cutlass.Int32(0)
        accumulated_max = cutlass.Int32(0)
        for iteration in range(iterations):
            payload = api.ThreadData(items_per_thread)
            for item in cutlass.range_constexpr(items_per_thread):
                payload[item] = inputs[thread * items_per_thread + item]
                payload[item] += iteration
            if group.is_member():
                total = api.sum(group, payload)
                largest = api.reduce(group, payload, binary_op="max")
                if group.rank() == 0:
                    accumulated_sum += total
                    accumulated_max += largest
            for item in cutlass.range_constexpr(items_per_thread):
                copied[thread * items_per_thread + item] = payload[item]
        if group.is_member():  # noqa: SIM102 - query ranks only for members
            if group.rank() == 0:
                group_index = (thread // 32) * groups_per_warp + (
                    thread % 32
                ) // width
                outputs[group_index * 2] = accumulated_sum
                outputs[group_index * 2 + 1] = accumulated_max

    @cute.jit
    def launch(
        source: cute.Pointer,
        preserved: cute.Pointer,
        observed: cute.Pointer,
        iterations: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(
            source, preserved, observed, iterations, items_per_thread
        ).launch(grid=1, block=(8, 4, 2))

    iterations = 5
    source = values_for(np.int32, tile_size, shift=107)
    preserved = np.zeros_like(source)
    observed = np.zeros(group_count * 2, dtype=np.int32)
    with (
        device_array(source) as src,
        device_array(preserved) as copied,
        device_array(observed) as out,
    ):
        launch(src, copied, out, iterations, items_per_thread)
    groups = source.reshape(2, 32, items_per_thread)[
        :, : groups_per_warp * width
    ].reshape(group_count, width * items_per_thread)
    increments = sum(range(iterations))
    np.testing.assert_array_equal(
        observed[::2],
        iterations * groups.sum(axis=1) + width * items_per_thread * increments,
    )
    np.testing.assert_array_equal(
        observed[1::2], iterations * groups.max(axis=1) + increments
    )
    np.testing.assert_array_equal(preserved, source + iterations - 1)


@pytest.mark.parametrize(
    "api", (coop, cutlass_coop), ids=("common", "qualified")
)
@pytest.mark.parametrize("width", [17, 31])
def test_non_power_of_two_warp_prefixes_reuse_scratch(api, width):
    """Partial logical-warp partitions keep scratch local to each group."""

    groups_per_warp = 32 // width
    group_count = 2 * groups_per_warp

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        observed: cute.Pointer,
        valid_items: cutlass.Int32,
        iterations: cutlass.Int32,
    ):
        thread = api.this_block().rank()
        inputs = cute.make_tensor(source, cute.make_layout(64))
        outputs = cute.make_tensor(observed, cute.make_layout(group_count * 2))
        group = api.this_warp().group_by(width, exhaustive=False)
        if group.is_member():
            accumulated_sum = cutlass.Int32(0)
            accumulated_max = cutlass.Int32(0)
            for iteration in range(iterations):
                total = api.sum(
                    group, inputs[thread] + iteration, valid_items=valid_items
                )
                largest = api.reduce(
                    group,
                    inputs[thread] + iteration,
                    binary_op="max",
                    valid_items=width // 2,
                )
                if group.rank() == 0:
                    accumulated_sum += total
                    accumulated_max += largest
            if group.rank() == 0:
                group_index = (thread // 32) * groups_per_warp + (
                    thread % 32
                ) // width
                outputs[group_index * 2] = accumulated_sum
                outputs[group_index * 2 + 1] = accumulated_max

    @cute.jit
    def launch(
        source: cute.Pointer,
        observed: cute.Pointer,
        valid_items: cutlass.Int32,
        iterations: cutlass.Int32,
    ):
        kernel(source, observed, valid_items, iterations).launch(
            grid=1, block=(8, 4, 2)
        )

    iterations = 5
    valid_items = width - 1
    source = values_for(np.int32, 64, shift=109)
    observed = np.zeros(group_count * 2, dtype=np.int32)
    with device_array(source) as src, device_array(observed) as out:
        launch(src, out, valid_items, iterations)
    groups = source.reshape(2, 32)[:, : groups_per_warp * width].reshape(
        group_count, width
    )
    increments = sum(range(iterations))
    np.testing.assert_array_equal(
        observed[::2],
        iterations * groups[:, :valid_items].sum(axis=1)
        + valid_items * increments,
    )
    np.testing.assert_array_equal(
        observed[1::2],
        iterations * groups[:, : width // 2].max(axis=1) + increments,
    )
