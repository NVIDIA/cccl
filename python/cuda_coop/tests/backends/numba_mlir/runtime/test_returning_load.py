# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check fresh Load ownership alongside the supported destination form."""

import numpy as np
import pytest

cuda = pytest.importorskip("numba_cuda_mlir.cuda")
if not cuda.is_available():
    pytest.skip("requires a CUDA-capable runtime", allow_module_level=True)

import cuda.coop.numba_mlir as numba_coop
from cuda import coop

pytestmark = [
    pytest.mark.backend_numba_mlir,
    pytest.mark.runtime,
    pytest.mark.gpu,
    pytest.mark.filterwarnings(
        "ignore::numba_cuda_mlir.numba_cuda.core.errors.NumbaPerformanceWarning"
    ),
]


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_returning_load_keeps_old_aliases_and_destination_load_updates_them(
    api, items_per_thread
):
    @cuda.jit
    def kernel(source, first, second, updated, flags, count):
        block = api.this_block()
        values = api.load(block, source, items_per_thread=count)
        alias = values
        values = api.load(
            block, source, items_per_thread=count, offset=32 * count
        )
        api.store(block, first, alias)
        api.store(block, second, values)
        result = api.load(block, source, alias, offset=64 * count)
        api.store(block, updated, alias)
        flags[cuda.threadIdx.x] = result is None

    size = 32 * items_per_thread
    source = np.arange(3 * size, dtype=np.int32)
    outputs = [cuda.device_array(size, dtype=np.int32) for _ in range(3)]
    flags = cuda.device_array(32, dtype=np.bool_)
    kernel[1, 32](cuda.to_device(source), *outputs, flags, items_per_thread)
    for index, output in enumerate(outputs):
        np.testing.assert_array_equal(
            output.copy_to_host(), source[index * size : (index + 1) * size]
        )
    assert flags.copy_to_host().all()


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("choose_second", [False, True])
def test_returning_load_through_helper_tuple_branch_and_loop(
    api, items_per_thread, choose_second
):
    @cuda.jit(device=True, inline="always")
    def pair(group, source, count):
        return (
            api.load(group, source, items_per_thread=count),
            api.load(group, source, items_per_thread=count, offset=32 * count),
        )

    @cuda.jit
    def kernel(source, output, saved_output, count, choose_second):
        block = api.this_block()
        payloads = pair(block, source, count)
        saved = payloads[0]
        if choose_second:
            values = payloads[1]
        else:
            values = payloads[0]
        for iteration in range(2):
            for item in range(count):
                values[item] += 1
            api.store(block, output, values, offset=iteration * 32 * count)
            values = api.load(block, source, items_per_thread=count)
        api.store(block, saved_output, saved)

    size = 32 * items_per_thread
    source = np.arange(2 * size, dtype=np.int32)
    output = cuda.device_array(2 * size, dtype=np.int32)
    saved = cuda.device_array(size, dtype=np.int32)
    kernel[1, 32](
        cuda.to_device(source), output, saved, items_per_thread, choose_second
    )
    first = source[size:] if choose_second else source[:size]
    np.testing.assert_array_equal(
        output.copy_to_host(), np.concatenate([first + 1, source[:size] + 1])
    )
    np.testing.assert_array_equal(
        saved.copy_to_host(), source[:size] + (0 if choose_second else 1)
    )


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
@pytest.mark.parametrize("items_per_thread", [1, 4])
@pytest.mark.parametrize("valid", [0, 17])
def test_returning_partial_load_explicit_padding(api, items_per_thread, valid):
    @cuda.jit
    def kernel(source, output, count, valid):
        block = api.this_block()
        values = api.load(
            block,
            source,
            items_per_thread=count,
            valid_items=valid,
            oob_default=-7,
            offset=3,
            algorithm="transpose",
            temp_storage=api.TempStorage(),
        )
        api.store(block, output, values)

    size = 32 * items_per_thread
    source = np.arange(size + 3, dtype=np.int16)
    output = cuda.device_array(size, dtype=np.int16)
    kernel[1, 32](cuda.to_device(source), output, items_per_thread, valid)
    expected = np.full(size, -7, dtype=np.int16)
    expected[:valid] = source[3 : 3 + valid]
    np.testing.assert_array_equal(output.copy_to_host(), expected)


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_returning_partial_load_without_padding_reads_only_valid_prefix(
    api, items_per_thread
):
    @cuda.jit
    def kernel(source, output, count):
        block = api.this_block()
        values = api.load(block, source, items_per_thread=count, valid_items=17)
        api.store(block, output, values, valid_items=17)

    size = 32 * items_per_thread
    source = np.arange(size, dtype=np.uint16)
    output = cuda.to_device(np.full(size, 999, dtype=np.uint16))
    kernel[1, 32](cuda.to_device(source), output, items_per_thread)
    expected = np.full(size, 999, dtype=np.uint16)
    expected[:17] = source[:17]
    np.testing.assert_array_equal(output.copy_to_host(), expected)


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
@pytest.mark.parametrize("items_per_thread", [1, 4])
def test_explicit_double_buffer_keeps_previous_iteration(api, items_per_thread):
    @cuda.jit
    def kernel(source, output, count, tiles):
        block = api.this_block()
        previous = api.ThreadData(count)
        current = api.ThreadData(count)
        api.load(block, source, previous)
        differences = api.ThreadData(count)
        for tile in range(1, tiles):
            api.load(block, source, current, offset=tile * 32 * count)
            for item in range(count):
                differences[item] = np.int32(current[item] - previous[item])
            api.store(
                block, output, differences, offset=(tile - 1) * 32 * count
            )
            previous, current = current, previous

    size = 32 * items_per_thread
    source = np.arange(4 * size, dtype=np.int32) ** 2
    output = cuda.device_array(3 * size, dtype=np.int32)
    kernel[1, 32](cuda.to_device(source), output, items_per_thread, 4)
    np.testing.assert_array_equal(
        output.copy_to_host(), source[size:] - source[:-size]
    )


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
def test_loop_result_consumed_before_replacement_and_after_loop(api):
    @cuda.jit
    def kernel(source, output, final_output, scalar_output, tiles):
        block = api.this_block()
        values = api.load(block, source, items_per_thread=2)
        scalar = 0
        for tile in range(1, tiles):
            old = values
            api.store(block, output, old, offset=(tile - 1) * 64)
            values = api.load(
                block, source, items_per_thread=2, offset=tile * 64
            )
            pair = ((values, values), 7)
            for item in range(2):
                scalar += pair[0][1][item]
        api.store(block, final_output, values)
        scalar_output[cuda.threadIdx.x] = scalar

    source = np.arange(4 * 64, dtype=np.int32)
    output = cuda.device_array(3 * 64, dtype=np.int32)
    final_output = cuda.device_array(64, dtype=np.int32)
    scalar_output = cuda.device_array(32, dtype=np.int64)
    kernel[1, 32](
        cuda.to_device(source), output, final_output, scalar_output, 4
    )
    np.testing.assert_array_equal(output.copy_to_host(), source[: 3 * 64])
    np.testing.assert_array_equal(final_output.copy_to_host(), source[3 * 64 :])
    np.testing.assert_array_equal(
        scalar_output.copy_to_host(),
        source[64:].reshape(3, 32, 2).sum(axis=(0, 2)),
    )


def test_returning_load_allows_an_alias_overwritten_before_the_next_call():
    @cuda.jit
    def kernel(source, output, tiles):
        block = coop.this_block()
        original = coop.load(block, source, items_per_thread=1)
        saved = original
        for tile in range(1, tiles):
            saved = original
            current = coop.load(
                block, source, items_per_thread=1, offset=tile * 32
            )
            output[(tile - 1) * 32 + cuda.threadIdx.x] = current[0] - saved[0]
            saved = current

    source = np.arange(4 * 32, dtype=np.int32) ** 2
    output = cuda.device_array(3 * 32, dtype=np.int32)
    kernel[1, 32](cuda.to_device(source), output, 4)
    np.testing.assert_array_equal(
        output.copy_to_host(),
        (source[32:].reshape(3, 32) - source[:32]).ravel(),
    )


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
def test_returning_load_allows_retained_tuple_metadata_and_array_shape(api):
    @cuda.jit
    def kernel(source, output, tiles, step):
        block = api.this_block()
        initial = api.load(block, source, items_per_thread=1)
        # Runtime strides keep tuple members in the same native array layout.
        # Numba does not cast static local arrays into loop-carried tuple slots.
        pair = (initial[::step], 7)
        scalar = 0
        for tile in range(tiles):
            current = api.load(
                block, source, items_per_thread=1, offset=tile * 32
            )
            view = current[::step]
            scalar += view[0]
            output[tile * 32 + cuda.threadIdx.x] = (
                scalar + pair[1] + pair[0].shape[0] + len(pair) + len(pair[0])
            )
            pair = (view, tile)

    source = np.arange(4 * 32, dtype=np.int32)
    output = cuda.device_array(source.size, dtype=np.int64)
    kernel[1, 32](cuda.to_device(source), output, 4, 1)
    expected = source.reshape(4, 32).cumsum(axis=0)
    expected += np.array([11, 4, 5, 6])[:, None]
    np.testing.assert_array_equal(output.copy_to_host(), expected.ravel())


@pytest.mark.parametrize("api", [coop, numba_coop], ids=["common", "qualified"])
def test_returning_load_allows_conditional_reload(api):
    @cuda.jit
    def kernel(source, output, tiles):
        block = api.this_block()
        values = api.load(block, source, items_per_thread=1)
        tile = 0
        while tile < tiles:
            if tile % 2 == 0:
                values = api.load(
                    block, source, items_per_thread=1, offset=tile * 32
                )
            output[tile * 32 + cuda.threadIdx.x] = values[0]
            tile += 1

    source = np.arange(4 * 32, dtype=np.int32)
    output = cuda.device_array(source.size, dtype=np.int32)
    kernel[1, 32](cuda.to_device(source), output, 4)
    expected = source.reshape(4, 32)[[0, 0, 2, 2]]
    np.testing.assert_array_equal(output.copy_to_host(), expected.ravel())
