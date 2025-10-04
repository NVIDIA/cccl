from functools import reduce
from operator import mul

import numpy as np
import pytest
from numba import cuda

import cuda.cccl.cooperative.experimental as coop


def get_histogram_bins_for_type(np_type):
    dtype = np.dtype(np_type)
    bins = 1 << (dtype.itemsize * 8)
    return bins if dtype.kind == "u" else bins >> 1


@pytest.mark.parametrize("item_dtype", [np.uint8, np.int8])
@pytest.mark.parametrize("counter_dtype", [np.int32, np.uint32])
@pytest.mark.parametrize("threads_per_block", [32, 128, (4, 16), (4, 8, 16)])
@pytest.mark.parametrize("items_per_thread", [2, 4, 8])
@pytest.mark.parametrize(
    "num_total_items",
    [
        1 << 10,  # 1KB
        1 << 12,  # 4KB
        1 << 15,  # 32KB
        1 << 19,  # 512KB
        1 << 23,  # 8MB
        1 << 28,  # 256MB
    ],
)
@pytest.mark.parametrize(
    "algorithm",
    [
        coop.BlockHistogramAlgorithm.ATOMIC,
        coop.BlockHistogramAlgorithm.SORT,
    ],
)
def test_block_histogram_histo_single_phase_2(
    item_dtype,
    counter_dtype,
    threads_per_block,
    items_per_thread,
    num_total_items,
    algorithm,
):
    # Example of what a two-phase histogram instance creation would look like,
    # purely for the purpose of demonstrating how much simpler the single-phase
    # construction is because we can infer so many of the other parameters
    # on the fly.
    histo_two_phase = coop.block.histogram(
        item_dtype=np.int8,
        counter_dtype=np.int32,
        dim=128,
        items_per_thread=4,
        algorithm=coop.BlockHistogramAlgorithm.ATOMIC,
        bins=256,
    )

    # N.B. The example above is still valid; the single-phase functionality
    #      does not remove the ability to construct these primitives outside
    #      of kernels--which you would do if you needed to access temp storage
    #      or alignment ahead of time.
    print(f"temp storage bytes: {histo_two_phase.temp_storage_bytes}")
    print(f"temp storage alignment: {histo_two_phase.temp_storage_alignment}")

    @cuda.jit
    def kernel(d_in, d_out, items_per_thread, num_total_items):
        tid = cuda.grid(1)
        if tid >= num_total_items:
            return

        threads_per_block = cuda.blockDim.x * cuda.blockDim.y * cuda.blockDim.z
        items_per_block = items_per_thread * threads_per_block

        block_offset = cuda.blockIdx.x * items_per_block

        # Shared per-block histogram bin counts.
        smem_histogram = coop.shared.array(
            d_out.shape,
            d_out.dtype,
            alignment=128,
        )
        thread_samples = coop.local.array(
            items_per_thread,
            d_in.dtype,
            alignment=16,
        )

        # Create the "parent" histogram instance.  Note that we only need
        # to provide two parameters--the local register array from which we
        # plan to read, and the corresponding shared-memory bin count array.
        #
        # We can infer all of the other arguments required by the two-phase
        # constructor (see `histo_two_phase` above) automatically, as follows:
        #
        #   item_dtype: inferred from thread_samples.dtype (which was inferred
        #               from d_in.dtype).
        #
        #   counter_dtype: inferred from smem_histogram.dtype (which was inferred
        #                  from d_out.dtype).
        #
        #   dim: inferred from the grid launch dimensions of the current kernel.
        #
        #   items_per_thread: inferred from the thread_samples.shape (which was
        #                     obtained via the items_per_thread kernel param).
        #
        #   algorithm: defaults to ATOMIC.
        #
        #   bins: inferred from smem_histogram.shape (which was inferred from
        #         d_out.shape).
        histo = coop.block.histogram(thread_samples, smem_histogram)

        # Initialize the histogram.  This corresponds to the CUB instance
        # method BlockHistogram::InitHistogram.  It is a "child" call of a
        # parent instance.  (Our shared-memory array also comes pre-zeroed
        # so this technically isn't necessary.)
        histo.init()

        cuda.syncthreads()

        # Loop through tiles of the input data, loading chunks optimally via
        # block load, then updating the histogram accordingly.  To simplify
        # the example, we don't have any corner-case handling for undesirable
        # shapes (e.g. items per block (and thus, block offset) not a perfect
        # multiple of total items, etc.).
        while block_offset < num_total_items:
            coop.block.load(
                d_in[block_offset:],
                thread_samples,
                algorithm=coop.BlockLoadAlgorithm.WARP_TRANSPOSE,
            )

            # This is the second "child" call against the parent histogram
            # instance, corresponding to the the CUB C++ instance method:
            #
            #   BlockHistogram::Compute(
            #       T& thread_samples[ITEMS_PER_THREAD],
            #       CounterT& histogram[BINS]
            #   )
            #
            # Note that we don't need to furnish the smem_histogram as the
            # second parameter here--that was already provided to the
            # histogram's constructor, and thus, is accessible to us behind
            # the scenes when we need to wire up this composite call.
            histo.composite(thread_samples)

            cuda.syncthreads()

            block_offset += items_per_block * cuda.gridDim.x

        # Final block atomic update to merge our block counts into the user's
        # output results array (d_out).
        for bin_idx in range(cuda.threadIdx.x, bins, threads_per_block):
            cuda.atomic.add(d_out, bin_idx, smem_histogram[bin_idx])

    # Kernel ends.  Remaining code is test scaffolding.

    bins = get_histogram_bins_for_type(item_dtype)

    threads_per_block = (
        threads_per_block
        if isinstance(threads_per_block, int)
        else reduce(mul, threads_per_block)
    )

    items_per_block = threads_per_block * items_per_thread
    blocks_per_grid = (num_total_items + items_per_block - 1) // items_per_block

    h_input = np.random.randint(0, bins, num_total_items, dtype=item_dtype)
    d_input = cuda.to_device(h_input)
    d_output = cuda.device_array(bins, dtype=counter_dtype)
    num_blocks = blocks_per_grid

    k = kernel[num_blocks, threads_per_block]
    k(d_input, d_output, items_per_thread, num_total_items)

    actual = d_output.copy_to_host()
    expected = np.bincount(h_input, minlength=bins).astype(counter_dtype)

    # Sanity check sum of histo bins matches total items.
    assert np.sum(actual) == num_total_items
    assert np.sum(expected) == num_total_items

    # Verify arrays match.
    np.testing.assert_array_equal(actual, expected)
