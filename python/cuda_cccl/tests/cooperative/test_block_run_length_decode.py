import numba
import numpy as np
from numba import cuda

import cuda.cccl.cooperative.experimental as coop

def test_block_run_length_decode_two_phase():

    dtype = np.int32
    dim = 128
    runs_per_thread = 16
    decoded_items_per_thread = 16
    decoded_offset_dtype = np.uint32

    block_rld = coop.block.run_length_decode(
        dtype,
        dim,
        runs_per_thread,
        decoded_items_per_thread,
        decoded_offset_dtype,
    )

    @cuda.jit
    def kernel1(run_values,
                run_lengths,
                run_items,
                runs_per_thread,
                decoded_items_per_thread):

        total_decoded_size = 0

        stride = cuda.blockDim.x * decoded_items_per_thread
        decoded_window_offset = 0
        while decoded_window_offset < total_decoded_size:
            relative_offsets = cuda.local.array(
                decoded_items_per_thread,
                dtype=run_lengths.dtype,
            )

            decoded_items = cuda.local.array(
                decoded_items_per_thread,
                dtype=run_items.dtype,
            )

            num_valid_items = total_decoded_size - decoded_window_offset

            block_rld.decode(
                decoded_items,
                relative_offsets,
                decoded_window_offset,
            )

            decoded_window_offset += stride

def test_block_run_length_decode_two_phase_alternate():

    dtype = np.int32
    dim = 128
    runs_per_thread = 16
    decoded_items_per_thread = 16
    decoded_offset_dtype = np.uint32

    block_rld = coop.block.run_length_decode(
        dtype,
        dim,
        runs_per_thread,
        decoded_items_per_thread,
        decoded_offset_dtype,
    )

    @cuda.jit
    def kernel1(run_values,
                run_lengths,
                run_items,
                runs_per_thread,
                decoded_items_per_thread):

        total_decoded_size = 0

        stride = cuda.blockDim.x * decoded_items_per_thread
        decoded_window_offset = 0
        while decoded_window_offset < total_decoded_size:
            relative_offsets = cuda.local.array(
                decoded_items_per_thread,
                dtype=run_lengths.dtype,
            )

            decoded_items = cuda.local.array(
                decoded_items_per_thread,
                dtype=run_items.dtype,
            )

            num_valid_items = total_decoded_size - decoded_window_offset

            block_rld(
                decoded_items,
                relative_offsets,
                decoded_window_offset,
            )

            decoded_window_offset += stride


def test_block_run_length_decode_single_phase():

    @cuda.jit
    def kernel1(run_values,
                run_lengths,
                run_items,
                runs_per_thread,
                decoded_items_per_thread):

        total_decoded_size = 0

        block_rld = coop.block.run_length_decode(
            run_values,
            run_lengths,
            total_decoded_size,
        )

        stride = cuda.blockDim.x * decoded_items_per_thread
        decoded_window_offset = 0
        while decoded_window_offset < total_decoded_size:
            relative_offsets = cuda.local.array(
                decoded_items_per_thread,
                dtype=run_lengths.dtype,
            )

            decoded_items = cuda.local.array(
                decoded_items_per_thread,
                dtype=run_items.dtype,
            )

            num_valid_items = total_decoded_size - decoded_window_offset

            block_rld(
                decoded_items,
                relative_offsets,
                decoded_window_offset,
            )

            decoded_window_offset += stride

