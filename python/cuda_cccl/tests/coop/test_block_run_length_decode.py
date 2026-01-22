import numba
import numpy as np
from numba import cuda

import cuda.coop as coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def _expected_decode(run_values, run_lengths):
    decoded_items = []
    relative_offsets = []
    for value, length in zip(run_values, run_lengths):
        for offset in range(length):
            decoded_items.append(value)
            relative_offsets.append(offset)
    return (
        np.asarray(decoded_items, dtype=run_values.dtype),
        np.asarray(relative_offsets, dtype=run_lengths.dtype),
    )


def test_block_run_length_decode_single_phase():
    item_dtype = np.uint32
    length_dtype = np.uint32
    decoded_offset_dtype = np.uint32

    threads_per_block = 32
    runs_per_thread = 2
    decoded_items_per_thread = 4

    total_runs = threads_per_block * runs_per_thread
    window_size = threads_per_block * decoded_items_per_thread

    h_run_values = np.arange(total_runs, dtype=item_dtype)
    h_run_lengths = (np.arange(total_runs, dtype=length_dtype) % 3) + 1
    h_run_lengths[-1] += window_size - int(h_run_lengths.sum())

    expected_items, expected_offsets = _expected_decode(h_run_values, h_run_lengths)
    expected_items = expected_items[:window_size]
    expected_offsets = expected_offsets[:window_size]

    d_run_values = cuda.to_device(h_run_values)
    d_run_lengths = cuda.to_device(h_run_lengths)
    d_decoded_items = cuda.device_array(window_size, dtype=item_dtype)
    d_relative_offsets = cuda.device_array(window_size, dtype=length_dtype)
    d_total_decoded_size = cuda.device_array(1, dtype=decoded_offset_dtype)

    @cuda.jit
    def kernel(
        run_values,
        run_lengths,
        decoded_items_out,
        relative_offsets_out,
        total_decoded_size_out,
    ):
        runs_per_thread = 2
        decoded_items_per_thread = 4

        run_values_local = coop.local.array(runs_per_thread, dtype=run_values.dtype)
        run_lengths_local = coop.local.array(runs_per_thread, dtype=run_lengths.dtype)

        block_offset = cuda.blockIdx.x * runs_per_thread * cuda.blockDim.x

        coop.block.load(
            run_values[block_offset:],
            run_values_local,
            items_per_thread=runs_per_thread,
            algorithm=coop.BlockLoadAlgorithm.DIRECT,
        )
        coop.block.load(
            run_lengths[block_offset:],
            run_lengths_local,
            items_per_thread=runs_per_thread,
            algorithm=coop.BlockLoadAlgorithm.DIRECT,
        )

        decoded_offset_dtype = total_decoded_size_out.dtype
        total_decoded_size = coop.local.array(1, dtype=decoded_offset_dtype)
        total_decoded_size[0] = 0

        run_length = coop.block.run_length(
            run_values_local,
            run_lengths_local,
            runs_per_thread,
            decoded_items_per_thread,
            total_decoded_size,
            decoded_offset_dtype=decoded_offset_dtype,
        )

        decoded_items = coop.local.array(
            decoded_items_per_thread, dtype=run_values.dtype
        )
        relative_offsets = coop.local.array(
            decoded_items_per_thread, dtype=run_lengths.dtype
        )
        decoded_window_offset = 0

        run_length.decode(decoded_items, decoded_window_offset, relative_offsets)

        base = cuda.threadIdx.x * decoded_items_per_thread
        for i in range(decoded_items_per_thread):
            decoded_items_out[base + i] = decoded_items[i]
            relative_offsets_out[base + i] = relative_offsets[i]

        if cuda.threadIdx.x == 0:
            total_decoded_size_out[cuda.blockIdx.x] = total_decoded_size[0]

    kernel[1, threads_per_block](
        d_run_values,
        d_run_lengths,
        d_decoded_items,
        d_relative_offsets,
        d_total_decoded_size,
    )
    cuda.synchronize()

    h_decoded_items = d_decoded_items.copy_to_host()
    h_relative_offsets = d_relative_offsets.copy_to_host()
    h_total_decoded_size = d_total_decoded_size.copy_to_host()

    np.testing.assert_array_equal(h_decoded_items, expected_items)
    np.testing.assert_array_equal(h_relative_offsets, expected_offsets)
    np.testing.assert_array_equal(h_total_decoded_size, [window_size])


def test_block_run_length_decode_two_phase():
    item_dtype = np.uint32
    length_dtype = np.uint32
    decoded_offset_dtype = np.uint32

    threads_per_block = 32
    runs_per_thread = 2
    decoded_items_per_thread = 4

    total_runs = threads_per_block * runs_per_thread
    window_size = threads_per_block * decoded_items_per_thread

    h_run_values = np.arange(total_runs, dtype=item_dtype)
    h_run_lengths = (np.arange(total_runs, dtype=length_dtype) % 3) + 1
    h_run_lengths[-1] += window_size - int(h_run_lengths.sum())

    expected_items, expected_offsets = _expected_decode(h_run_values, h_run_lengths)
    expected_items = expected_items[:window_size]
    expected_offsets = expected_offsets[:window_size]

    d_run_values = cuda.to_device(h_run_values)
    d_run_lengths = cuda.to_device(h_run_lengths)
    d_decoded_items = cuda.device_array(window_size, dtype=item_dtype)
    d_relative_offsets = cuda.device_array(window_size, dtype=length_dtype)
    d_total_decoded_size = cuda.device_array(1, dtype=decoded_offset_dtype)

    run_length_instance = coop.block.run_length(
        item_dtype,
        threads_per_block,
        runs_per_thread,
        decoded_items_per_thread,
        decoded_offset_dtype=decoded_offset_dtype,
    )

    @cuda.jit
    def kernel(
        run_values,
        run_lengths,
        decoded_items_out,
        relative_offsets_out,
        total_decoded_size_out,
    ):
        runs_per_thread = 2
        decoded_items_per_thread = 4

        run_values_local = coop.local.array(runs_per_thread, dtype=run_values.dtype)
        run_lengths_local = coop.local.array(runs_per_thread, dtype=run_lengths.dtype)

        block_offset = cuda.blockIdx.x * runs_per_thread * cuda.blockDim.x

        coop.block.load(
            run_values[block_offset:],
            run_values_local,
            items_per_thread=runs_per_thread,
            algorithm=coop.BlockLoadAlgorithm.DIRECT,
        )
        coop.block.load(
            run_lengths[block_offset:],
            run_lengths_local,
            items_per_thread=runs_per_thread,
            algorithm=coop.BlockLoadAlgorithm.DIRECT,
        )

        total_decoded_size = coop.local.array(1, dtype=decoded_offset_dtype)
        total_decoded_size[0] = 0

        run_length = run_length_instance(
            run_values_local,
            run_lengths_local,
            runs_per_thread,
            decoded_items_per_thread,
            total_decoded_size,
            decoded_offset_dtype=decoded_offset_dtype,
        )

        decoded_items = coop.local.array(
            decoded_items_per_thread, dtype=run_values.dtype
        )
        relative_offsets = coop.local.array(
            decoded_items_per_thread, dtype=run_lengths.dtype
        )
        decoded_window_offset = 0

        run_length.decode(decoded_items, decoded_window_offset, relative_offsets)

        base = cuda.threadIdx.x * decoded_items_per_thread
        for i in range(decoded_items_per_thread):
            decoded_items_out[base + i] = decoded_items[i]
            relative_offsets_out[base + i] = relative_offsets[i]

        if cuda.threadIdx.x == 0:
            total_decoded_size_out[cuda.blockIdx.x] = total_decoded_size[0]

    kernel[1, threads_per_block](
        d_run_values,
        d_run_lengths,
        d_decoded_items,
        d_relative_offsets,
        d_total_decoded_size,
    )
    cuda.synchronize()

    h_decoded_items = d_decoded_items.copy_to_host()
    h_relative_offsets = d_relative_offsets.copy_to_host()
    h_total_decoded_size = d_total_decoded_size.copy_to_host()

    np.testing.assert_array_equal(h_decoded_items, expected_items)
    np.testing.assert_array_equal(h_relative_offsets, expected_offsets)
    np.testing.assert_array_equal(h_total_decoded_size, [window_size])
