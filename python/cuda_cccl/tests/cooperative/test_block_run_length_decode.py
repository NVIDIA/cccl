import numba
import numpy as np
from numba import cuda

import cuda.cccl.cooperative.experimental as coop


def test_block_run_length_decode_single_phase0():
    dtype = np.int32
    dim = 128
    runs_per_thread = 16
    decoded_items_per_thread = 16
    decoded_offset_dtype = np.uint32

    # Can only use this instance for temp_storage.
    run_length = coop.block.run_length(
        dtype,
        dim,
        runs_per_thread,
        decoded_items_per_thread,
        decoded_offset_dtype,
    )
    temp_storage_bytes = run_length.temp_storage_bytes
    temp_storage_alignment = run_length.temp_storage_alignment

    @cuda.jit
    def kernel1(
        run_values, run_lengths, run_items, runs_per_thread, decoded_items_per_thread
    ):
        total_decoded_size = numba.int32(0)

        temp_storage = coop.shared.array(
            temp_storage_bytes,
            dtype=np.uint8,
            alignment=temp_storage_alignment,
        )

        run_length = coop.block.run_length(
            run_values,
            run_lengths,
            runs_per_thread,
            decoded_items_per_thread,
            total_decoded_size,
            temp_storage=temp_storage,
        )

        stride = cuda.blockDim.x * decoded_items_per_thread
        decoded_window_offset = 0

        relative_offsets = coop.local.array(
            decoded_items_per_thread,
            dtype=run_lengths.dtype,
        )

        decoded_items = coop.local.array(
            decoded_items_per_thread,
            dtype=run_items.dtype,
        )

        while decoded_window_offset < total_decoded_size:
            # num_valid_items = total_decoded_size - decoded_window_offset

            run_length.decode(
                decoded_items,
                decoded_window_offset,
                relative_offsets,
            )

            # Or, without relative_offsets:
            # run_length.decode(decoded_items, decoded_window_offset)

            decoded_window_offset += stride

            global_idx = (
                cuda.threadIdx.x * decoded_items_per_thread + decoded_window_offset
            )

            if global_idx < total_decoded_size:
                start = global_idx
                end = min(global_idx + decoded_items_per_thread, total_decoded_size)
                run_items[start:end] = decoded_items[: end - start]

    h_run_values = np.random.randint(0, dim, dim, dtype=dtype)
    h_run_lengths = np.random.randint(0, dim, dim, dtype=dtype)
    h_run_items = np.random.randint(0, dim, dim, dtype=dtype)
    d_run_values = cuda.to_device(h_run_values)
    d_run_lengths = cuda.to_device(h_run_lengths)
    d_run_items = cuda.device_array(dim, dtype=dtype)
    d_output = cuda.device_array(dim, dtype=decoded_offset_dtype)
    num_blocks = 1
    k = kernel1[num_blocks, dim]
    k(
        d_run_values,
        d_run_lengths,
        d_run_items,
        runs_per_thread,
        decoded_items_per_thread,
    )

    # Generate a reference output.
    ref_output = []
    for i in range(dim):
        length = h_run_lengths[i]
        value = h_run_items[i]
        ref_output.extend([value] * length)
    ref_output = np.array(ref_output[:dim])

    h_output = d_output.copy_to_host()
    h_run_items = h_run_items.copy_to_host()

    h_run_items_result = d_run_items.copy_to_host()
    np.testing.assert_array_equal(
        h_run_items_result[: len(ref_output)],
        ref_output,
    )

    # Verify the output.
    np.testing.assert_array_equal(h_output, h_run_items)
