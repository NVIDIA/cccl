# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compare decoded windows and complete streams with a host repeat oracle.

Each block owns independent runs and output space. Window results have a
separate extent and zero-filled tails. Bulk results preserve destination
sentinels outside the decoded stream and return its total to every thread.
Original values and run lengths must survive both forms unchanged.
"""

from contextlib import ExitStack

import numpy as np
import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass import cute

from cuda import coop
from cuda.coop import cutlass as cutlass_coop
from tests.backends.cutlass.support import (
    NUMPY_DTYPES,
    cutlass_dtype,
    device_array,
    values_for,
)

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


class _Readonly:
    """Expose run items and metadata without a writable payload interface.

    Both values and lengths can use this tuple snapshot. Original payloads
    remain available for preservation checks after decoding.
    """

    def __init__(self, source):
        self.items_per_thread = source.items_per_thread
        self.dtype = source.dtype
        self.alignment = source.alignment
        self._items = tuple(source)

    def __len__(self):
        return self.items_per_thread

    def __getitem__(self, index):
        return self._items[index]


def _run(
    api=coop,
    *,
    bulk=False,
    dtype=np.int32,
    length_dtype=np.uint64,
    offset=11,
    control_type=cutlass.Uint64,
    static=False,
    threads=32,
    items_per_thread=2,
    decoded=3,
    blocks=2,
    capacity=512,
    empty=False,
    readonly=False,
    inferred=False,
    sharing=None,
    alignment=64,
    auto_sync=True,
    repeats=1,
    storage_bytes=None,
    invalid=None,
    compile_options=(),
    dynamic_capacity=False,
):
    """Check each block with independent run and output extents.

    The host expands positive lengths followed by trailing zero padding.
    Window output is compared bitwise, including zero-filled positions beyond
    the stream. Bulk output retains sentinels before and after the stream;
    every thread must report the same total length.

    The kernel decodes the same inputs ``repeats`` times. Explicit barriers
    protect shared or exclusive scratch when automatic synchronization is
    disabled. Invalid variants serve child-process trap tests and must fail
    before reaching the host expansion checks.
    """

    value_type, length_type = cutlass_dtype(dtype), cutlass_dtype(length_dtype)
    run_tile, window = threads * items_per_thread, threads * decoded
    output_size = capacity if bulk else window

    @cute.kernel
    def kernel(
        source: cute.Pointer,
        counts: cute.Pointer,
        output: cute.Pointer,
        value_check: cute.Pointer,
        length_check: cute.Pointer,
        totals: cute.Pointer,
        dynamic_offset: control_type,
        dynamic_capacity_arg: cutlass.Int64,
        iterations: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        thread, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        sources = cute.recast_tensor(
            cute.make_tensor(source, cute.make_layout(blocks * run_tile)),
            value_type,
        )
        counts_tensor = cute.recast_tensor(
            cute.make_tensor(counts, cute.make_layout(blocks * run_tile)),
            length_type,
        )
        outputs = cute.recast_tensor(
            cute.make_tensor(output, cute.make_layout(blocks * output_size)),
            value_type,
        )
        values_check = cute.recast_tensor(
            cute.make_tensor(value_check, cute.make_layout(blocks * run_tile)),
            value_type,
        )
        lengths_check = cute.recast_tensor(
            cute.make_tensor(length_check, cute.make_layout(blocks * run_tile)),
            length_type,
        )
        total_output = cute.recast_tensor(
            cute.make_tensor(totals, cute.make_layout(blocks * threads)),
            cutlass.Uint32,
        )
        group = api.this_block()
        values = api.ThreadData(
            items_per_thread,
            dtype=None if inferred else value_type,
            alignment=alignment,
        )
        lengths = api.ThreadData(
            items_per_thread, dtype=None if inferred else length_type
        )
        for item in cutlass.range_constexpr(items_per_thread):
            index = block * run_tile + thread * items_per_thread + item
            values[item] = sources[index]
            lengths[item] = counts_tensor[index]
        if cutlass.const_expr(readonly):
            value_input, length_input = _Readonly(values), _Readonly(lengths)
        else:
            value_input, length_input = values, lengths
        storage = (
            None
            if sharing is None
            else api.TempStorage(
                storage_bytes,
                alignment=alignment,
                sharing=sharing,
                auto_sync=auto_sync,
            )
        )
        start = offset if static else dynamic_offset
        for iteration in range(iterations):
            if cutlass.const_expr(bulk):
                extent = dynamic_capacity_arg if dynamic_capacity else capacity
                destination = cute.recast_tensor(
                    cute.make_tensor(
                        outputs.iterator + block * capacity,
                        cute.make_layout(extent),
                    ),
                    value_type,
                )
                total = api.run_length_decode_into(
                    group,
                    value_input,
                    length_input,
                    destination,
                    decoded_items_per_thread=decoded,
                    destination_offset=start,
                    temp_storage=storage,
                )
                total_output[block * threads + thread] = total
            else:
                result = api.run_length_decode(
                    group,
                    value_input,
                    length_input,
                    decoded_items_per_thread=decoded,
                    decoded_window_offset=start,
                    temp_storage=storage,
                )
                assert isinstance(result, cutlass_coop.ThreadData)
                assert result is not values
                assert result.dtype is value_type
                assert result.items_per_thread == decoded
                assert result.alignment == alignment
                for item in cutlass.range_constexpr(decoded):
                    outputs[block * window + thread * decoded + item] = result[
                        item
                    ]
            if cutlass.const_expr(sharing is not None and not auto_sync):
                storage.sync()
        for item in cutlass.range_constexpr(items_per_thread):
            index = block * run_tile + thread * items_per_thread + item
            values_check[index] = values[item]
            lengths_check[index] = lengths[item]

    @cute.jit
    def launch(
        source: cute.Pointer,
        counts: cute.Pointer,
        output: cute.Pointer,
        value_check: cute.Pointer,
        length_check: cute.Pointer,
        totals: cute.Pointer,
        dynamic_offset: control_type,
        dynamic_capacity_arg: cutlass.Int64,
        iterations: cutlass.Int32,
        items_per_thread: cutlass.Constexpr,
    ):
        kernel(
            source,
            counts,
            output,
            value_check,
            length_check,
            totals,
            dynamic_offset,
            dynamic_capacity_arg,
            iterations,
            items_per_thread,
        ).launch(grid=blocks, block=threads)

    source = values_for(dtype, blocks * run_tile)
    counts = np.tile((np.arange(run_tile) % 9 + 1).astype(length_dtype), blocks)
    for block in range(blocks):
        counts[block * run_tile + run_tile - 5 : (block + 1) * run_tile] = 0
    if empty:
        counts.fill(0)
    if invalid == "negative":
        counts[0] = -1
    elif invalid == "padding":
        counts[0] = 0
    elif invalid == "overflow":
        counts[0] = 1 << 32
    elif invalid == "wide-overflow":
        counts[0] = (1 << 64) - 1
    observed = np.full(blocks * output_size, 42, dtype=dtype)
    preserved_values, preserved_lengths = (
        np.empty_like(source),
        np.empty_like(counts),
    )
    totals = np.full(blocks * threads, (1 << 32) - 1, dtype=np.uint32)
    arrays = (
        source,
        counts,
        observed,
        preserved_values,
        preserved_lengths,
        totals,
    )
    with ExitStack() as stack:
        pointers = [
            stack.enter_context(device_array(array)) for array in arrays
        ]
        args = (
            *pointers,
            control_type(offset),
            cutlass.Int64(capacity),
            cutlass.Int32(repeats),
        )
        compiled = cute.compile[compile_options](
            launch, *args, items_per_thread
        )
        compiled(*args)
    assert invalid is None, "invalid run lengths did not trap"
    np.testing.assert_array_equal(
        preserved_values.view(np.uint8), source.view(np.uint8)
    )
    np.testing.assert_array_equal(preserved_lengths, counts)
    for block in range(blocks):
        start = block * run_tile
        expected_stream = np.repeat(
            source[start : start + run_tile],
            counts[start : start + run_tile].astype(np.int64),
        )
        if bulk:
            expected = np.full(capacity, 42, dtype=dtype)
            expected[offset : offset + len(expected_stream)] = expected_stream
            np.testing.assert_array_equal(
                totals[block * threads : (block + 1) * threads],
                len(expected_stream),
            )
        else:
            expected = np.zeros(window, dtype=dtype)
            chunk = (
                expected_stream[offset : offset + window]
                if offset < len(expected_stream)
                else expected_stream[:0]
            )
            expected[: len(chunk)] = chunk
        np.testing.assert_array_equal(
            observed[block * output_size : (block + 1) * output_size].view(
                np.uint8
            ),
            expected.view(np.uint8),
        )
    return compiled


@pytest.mark.parametrize(
    "api", (coop, cutlass_coop), ids=("common", "qualified")
)
@pytest.mark.parametrize("bulk", (False, True))
@pytest.mark.parametrize("static", (False, True))
@pytest.mark.parametrize("items_per_thread", (1, 4))
def test_entrypoints(api, bulk, static, items_per_thread):
    _run(
        api,
        bulk=bulk,
        static=static,
        items_per_thread=items_per_thread,
        capacity=256 * items_per_thread,
    )


@pytest.mark.parametrize("bulk", (False, True))
@pytest.mark.parametrize("dtype", NUMPY_DTYPES)
def test_value_types(dtype, bulk):
    _run(dtype=dtype, bulk=bulk)


@pytest.mark.parametrize("length_dtype", NUMPY_DTYPES[:8])
def test_length_types(length_dtype):
    _run(length_dtype=length_dtype, bulk=True)


@pytest.mark.parametrize("offset", (0, 1, 17, 275, 289, 1000, (1 << 64) - 1))
def test_window_offsets_and_tails(offset):
    """Keep wide window offsets intact through range checks and decoding.

    Offsets cover stream starts, boundaries, partial tails, and positions far
    beyond the end. Even the largest Uint64 offset must yield a zero window
    instead of wrapping to a valid position.
    """

    _run(offset=offset)


@pytest.mark.parametrize("bulk", (False, True))
def test_empty_runs(bulk):
    """Reuse scratch when every run length is zero.

    Windows must be all zero. Bulk calls must return zero to every thread and
    leave the destination sentinel intact across repeated calls.
    """

    _run(bulk=bulk, empty=True, repeats=3, sharing="shared")


@pytest.mark.parametrize("bulk", (False, True))
def test_readonly_and_inferred(bulk):
    _run(bulk=bulk, readonly=True, inferred=True)


@pytest.mark.parametrize("sharing", ("shared", "exclusive"))
@pytest.mark.parametrize("auto_sync", (False, True))
@pytest.mark.parametrize("bulk", (False, True))
def test_scratch_reuse(sharing, auto_sync, bulk):
    """Repeat decoding with both scratch ownership and reuse-barrier policies.

    The same call site executes three times. Manual barriers follow each call
    when automatic synchronization is disabled, including the final iteration.
    """

    _run(
        bulk=bulk,
        repeats=3,
        sharing=sharing,
        auto_sync=auto_sync,
        alignment=128,
    )


@pytest.mark.parametrize("bulk", (False, True))
def test_alignment_minimum(bulk):
    """Allow native scratch alignment to exceed a one-byte request.

    The 32 KiB scratch size keeps size limits from masking alignment errors.
    Repeated window and bulk calls still pass their complete output and
    input-preservation checks.
    """

    _run(
        bulk=bulk, repeats=3, sharing="shared", alignment=1, storage_bytes=32768
    )


def test_dynamic_capacity():
    _run(bulk=True, dynamic_capacity=True)


@pytest.mark.parametrize("bulk", (False, True))
@pytest.mark.parametrize(
    "threads,items_per_thread,decoded", ((17, 1, 4), (64, 1, 1))
)
def test_block_and_window_extents(bulk, threads, items_per_thread, decoded):
    _run(
        bulk=bulk,
        threads=threads,
        items_per_thread=items_per_thread,
        decoded=decoded,
    )
