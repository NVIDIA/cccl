/******************************************************************************
 * Copyright (c) 2011-2020, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * \file
 * agent_radix_sort_histogram.cuh implements a stateful abstraction of CUDA
 * thread blocks for participating in the device histogram kernel used for
 * one-sweep radix sorting.
 */

#pragma once

#include "../config.cuh"

_CCCL_IMPLICIT_SYSTEM_HEADER

#include "../block/block_load.cuh"
#include "../block/radix_rank_sort_operations.cuh"
#include "../thread/thread_reduce.cuh"
#include "../util_math.cuh"
#include "../util_type.cuh"


CUB_NAMESPACE_BEGIN

template <
  int _BLOCK_THREADS,
  int _ITEMS_PER_THREAD,
  int NOMINAL_4B_NUM_PARTS,
  typename ComputeT,
  int _RADIX_BITS>
struct AgentRadixSortHistogramPolicy
{
    enum
    {
        BLOCK_THREADS = _BLOCK_THREADS,
        ITEMS_PER_THREAD = _ITEMS_PER_THREAD,
        /** NUM_PARTS is the number of private histograms (parts) each histogram is split
         * into. Each warp lane is assigned to a specific part based on the lane
         * ID. However, lanes with the same ID in different warp use the same private
         * histogram. This arrangement helps reduce the degree of conflicts in atomic
         * operations. */
        NUM_PARTS = CUB_MAX(1, NOMINAL_4B_NUM_PARTS * 4 / CUB_MAX(sizeof(ComputeT), 4)),
        RADIX_BITS = _RADIX_BITS,
    };
};

template <
    int _BLOCK_THREADS,
    int _RADIX_BITS>
struct AgentRadixSortExclusiveSumPolicy
{
    enum
    {
        BLOCK_THREADS = _BLOCK_THREADS,
        RADIX_BITS = _RADIX_BITS,
    };
};

template <
    typename AgentRadixSortHistogramPolicy,
    bool IS_DESCENDING,
    typename KeyT,
    typename OffsetT,
    typename DecomposerT = detail::identity_decomposer_t>
struct AgentRadixSortHistogram
{
    // constants
    enum
    {
        ITEMS_PER_THREAD = AgentRadixSortHistogramPolicy::ITEMS_PER_THREAD,
        BLOCK_THREADS = AgentRadixSortHistogramPolicy::BLOCK_THREADS,
        TILE_ITEMS = BLOCK_THREADS * ITEMS_PER_THREAD,
        RADIX_BITS = AgentRadixSortHistogramPolicy::RADIX_BITS,
        RADIX_DIGITS = 1 << RADIX_BITS,
        MAX_NUM_PASSES = (sizeof(KeyT) * 8 + RADIX_BITS - 1) / RADIX_BITS,
        NUM_PARTS = AgentRadixSortHistogramPolicy::NUM_PARTS,
    };

    using traits = detail::radix::traits_t<KeyT>;
    using bit_ordered_type = typename traits::bit_ordered_type;
    using bit_ordered_conversion = typename traits::bit_ordered_conversion_policy;

    typedef RadixSortTwiddle<IS_DESCENDING, KeyT> Twiddle;
    typedef std::uint32_t ShmemCounterT;
    typedef ShmemCounterT ShmemAtomicCounterT;

    using fundamental_digit_extractor_t = ShiftDigitExtractor<KeyT>;
    using digit_extractor_t =
      typename traits::template digit_extractor_t<fundamental_digit_extractor_t, DecomposerT>;

    struct _TempStorage
    {
        ShmemAtomicCounterT bins[MAX_NUM_PASSES][RADIX_DIGITS][NUM_PARTS];
    };

    struct TempStorage : Uninitialized<_TempStorage> {};

    // thread fields
    // shared memory storage
    _TempStorage& s;

    // bins for the histogram
    OffsetT* d_bins_out;

    // data to compute the histogram
    const bit_ordered_type* d_keys_in;

    // number of data items
    OffsetT num_items;

    // begin and end bits for sorting
    int begin_bit, end_bit;

    // number of sorting passes
    int num_passes;

    DecomposerT decomposer;

    __device__ __forceinline__ AgentRadixSortHistogram(TempStorage &temp_storage,
                                                       OffsetT *d_bins_out,
                                                       const KeyT *d_keys_in,
                                                       OffsetT num_items,
                                                       int begin_bit,
                                                       int end_bit,
                                                       DecomposerT decomposer = {})
        : s(temp_storage.Alias())
        , d_bins_out(d_bins_out)
        , d_keys_in(reinterpret_cast<const bit_ordered_type *>(d_keys_in))
        , num_items(num_items)
        , begin_bit(begin_bit)
        , end_bit(end_bit)
        , num_passes((end_bit - begin_bit + RADIX_BITS - 1) / RADIX_BITS)
        , decomposer(decomposer)
    {}

    __device__ __forceinline__ void Init()
    {
        // Initialize bins to 0.
        #pragma unroll
        for (int bin = threadIdx.x; bin < RADIX_DIGITS; bin += BLOCK_THREADS)
        {
            #pragma unroll
            for (int pass = 0; pass < num_passes; ++pass)
            {
                #pragma unroll
                for (int part = 0; part < NUM_PARTS; ++part)
                {
                    s.bins[pass][bin][part] = 0;
                }
            }
        }
        CTA_SYNC();
    }

    __device__ __forceinline__
    void LoadTileKeys(OffsetT tile_offset, bit_ordered_type (&keys)[ITEMS_PER_THREAD])
    {
        // tile_offset < num_items always, hence the line below works
        bool full_tile = num_items - tile_offset >= TILE_ITEMS;
        if (full_tile)
        {
            LoadDirectStriped<BLOCK_THREADS>(
                threadIdx.x, d_keys_in + tile_offset, keys);
        }
        else
        {
            LoadDirectStriped<BLOCK_THREADS>(
                threadIdx.x, d_keys_in + tile_offset, keys,
                num_items - tile_offset, Twiddle::DefaultKey(decomposer));
        }

        #pragma unroll
        for (int u = 0; u < ITEMS_PER_THREAD; ++u)
        {
            keys[u] = Twiddle::In(keys[u], decomposer);
        }
    }

    __device__ __forceinline__
    void AccumulateSharedHistograms(OffsetT tile_offset, bit_ordered_type (&keys)[ITEMS_PER_THREAD])
    {
        int part = LaneId() % NUM_PARTS;
        #pragma unroll
        for (int current_bit = begin_bit, pass = 0;
             current_bit < end_bit; current_bit += RADIX_BITS, ++pass)
        {
            int num_bits = CUB_MIN(RADIX_BITS, end_bit - current_bit);
            #pragma unroll
            for (int u = 0; u < ITEMS_PER_THREAD; ++u)
            {
                std::uint32_t bin = digit_extractor(current_bit, num_bits).Digit(keys[u]);
                // Using cuda::atomic<> results in lower performance on GP100,
                // so atomicAdd() is used instead.
                atomicAdd(&s.bins[pass][bin][part], 1);
            }
        }
    }

    __device__ __forceinline__ void AccumulateGlobalHistograms()
    {
        #pragma unroll
        for (int bin = threadIdx.x; bin < RADIX_DIGITS; bin += BLOCK_THREADS)
        {
            #pragma unroll
            for (int pass = 0; pass < num_passes; ++pass)
            {
                OffsetT count = internal::ThreadReduce(s.bins[pass][bin], Sum());
                if (count > 0)
                {
                    // Using cuda::atomic<> here would also require using it in
                    // other kernels. However, other kernels of onesweep sorting
                    // (ExclusiveSum, Onesweep) don't need atomic
                    // access. Therefore, atomicAdd() is used, until
                    // cuda::atomic_ref<> becomes available.
                    atomicAdd(&d_bins_out[pass * RADIX_DIGITS + bin], count);
                }
            }
        }
    }

    __device__ __forceinline__ void Process()
    {
        // Within a portion, avoid overflowing (u)int32 counters.
        // Between portions, accumulate results in global memory.
        constexpr OffsetT MAX_PORTION_SIZE = 1 << 30;
        OffsetT num_portions = cub::DivideAndRoundUp(num_items, MAX_PORTION_SIZE);
        for (OffsetT portion = 0; portion < num_portions; ++portion)
        {
            // Reset the counters.
            Init();
            CTA_SYNC();

            // Process the tiles.
            OffsetT portion_offset = portion * MAX_PORTION_SIZE;
            OffsetT portion_size = CUB_MIN(MAX_PORTION_SIZE, num_items - portion_offset);
            for (OffsetT offset = blockIdx.x * TILE_ITEMS; offset < portion_size;
                 offset += TILE_ITEMS * gridDim.x)
            {
                OffsetT tile_offset = portion_offset + offset;
                bit_ordered_type keys[ITEMS_PER_THREAD];
                LoadTileKeys(tile_offset, keys);
                AccumulateSharedHistograms(tile_offset, keys);
            }
            CTA_SYNC();

            // Accumulate the result in global memory.
            AccumulateGlobalHistograms();
            CTA_SYNC();
        }
    }

    __device__ __forceinline__ digit_extractor_t digit_extractor(int current_bit, int num_bits)
    {
        return traits::template digit_extractor<fundamental_digit_extractor_t>(current_bit,
                                                                               num_bits,
                                                                               decomposer);
    }
};

CUB_NAMESPACE_END
