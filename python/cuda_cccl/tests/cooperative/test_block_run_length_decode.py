import numba
import numba.types
import numpy as np
from numba import cuda

import cuda.cccl.cooperative.experimental as coop
from cuda.cccl.cooperative.experimental._numba_extension import (
    _set_source_code_rewriter,
)

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


def source_code_rewriter(source, algorithm) -> str:
    print(f"Rewriting source code for algorithm: {algorithm}")
    print(f"Source code before rewriting:\n{source}")
    return source


_set_source_code_rewriter(source_code_rewriter)


def test_block_run_length_decode_single_phase0():
    min_length = 1
    max_length = 5
    max_value = 1000
    item_dtype = np.uint32
    length_dtype = np.uint32
    threads_per_block = 128
    total_num_runs = 10000
    runs_per_thread = np.uint32(4)
    decoded_items_per_thread = np.uint32(4)
    decoded_size_dtype = np.uint32
    decoded_offset_dtype = np.uint32
    # total_decoded_size_dtype = np.uint32
    # relative_offset_dtype = np.int32

    num_blocks = (total_num_runs + (threads_per_block - 1)) // threads_per_block

    # num_block_threads = threads_per_block * num_blocks
    # runs_per_block = runs_per_thread * num_block_threads

    # dim = (num_blocks, threads_per_block)

    @cuda.jit
    def get_sizes_kernel(
        d_run_values,
        d_run_lengths,
        decoded_sizes,
        num_runs,
        runs_per_thread,
        decoded_items_per_thread,
    ):
        # total_decoded_size = total_decoded_size_dtype(0)

        num_block_threads = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        runs_per_block = runs_per_thread * num_block_threads

        block_offset = cuda.blockIdx.x * runs_per_block
        if block_offset + runs_per_block >= num_runs:
            num_valid_runs = num_runs - block_offset
        else:
            num_valid_runs = runs_per_block

        unique_items = coop.local.array(
            runs_per_thread,
            dtype=d_run_values.dtype,
        )
        src_run_values = d_run_values[block_offset:]

        coop.block.load(
            src_run_values,
            unique_items,
            items_per_thread=runs_per_thread,
            algorithm=coop.BlockLoadAlgorithm.WARP_TRANSPOSE,
            num_valid_items=num_valid_runs,
        )

        # block_load_items(
        #    src_run_values,
        #    unique_items,
        #    num_valid_items=num_valid_runs,
        #    temp_storage=None,
        # )

        run_lengths = coop.local.array(runs_per_thread, dtype=length_dtype)
        src_run_lengths = d_run_lengths[block_offset:]

        coop.block.load(
            src_run_lengths,
            run_lengths,
            items_per_thread=runs_per_thread,
            algorithm=coop.BlockLoadAlgorithm.WARP_TRANSPOSE,
            num_valid_items=num_valid_runs,
        )

        # block_load_lengths(
        #    src_run_lengths,
        #    run_lengths,
        #    num_valid_items=num_valid_runs,
        # )

        decoded_size = decoded_size_dtype(0)

        run_length = coop.block.run_length(  # noqa: F841
            unique_items,
            run_lengths,
            runs_per_thread,
            decoded_items_per_thread,
            decoded_size,
            decoded_offset_dtype,
        )

        decoded_sizes[cuda.blockIdx.x] = decoded_size

    h_run_values = np.random.randint(
        low=0,
        high=max_value,
        size=total_num_runs,
        dtype=item_dtype,
    )

    h_run_lengths = np.random.randint(
        low=min_length,
        high=max_length,
        size=total_num_runs,
        dtype=length_dtype,
    )

    h_decoded_sizes = np.zeros(
        num_blocks,
        dtype=decoded_size_dtype,
    )

    d_run_values = cuda.to_device(h_run_values)
    d_run_lengths = cuda.to_device(h_run_lengths)
    d_decoded_sizes = cuda.device_array(num_blocks, dtype=decoded_size_dtype)

    k = get_sizes_kernel[num_blocks, threads_per_block]
    k(
        d_run_values,
        d_run_lengths,
        d_decoded_sizes,
        total_num_runs,
        runs_per_thread,
        decoded_items_per_thread,
    )
    cuda.synchronize()

    h_decoded_sizes = d_decoded_sizes.copy_to_host()

    all_zeros = np.all(h_decoded_sizes == 0)
    assert not all_zeros, "All decoded sizes are zero, check input data."


def test_block_run_length_decode_single_phase1():
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
            # decoded_offset_dtype=np.uint16,
            # total_decoded_size=total_decoded_size,
            temp_storage=temp_storage,
        )

        stride = cuda.blockDim.x * decoded_items_per_thread
        decoded_window_offset = decoded_offset_dtype(0)

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


if __name__ == "__main__":
    test_block_run_length_decode_single_phase0()
    print("Test passed successfully.")
