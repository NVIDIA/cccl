import numba
import numba.types
import numpy as np
from numba import cuda

import cuda.cccl.cooperative.experimental as coop

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


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

        unique_items = coop.local.array(runs_per_thread, dtype=item_dtype)
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
