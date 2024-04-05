/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2018, NVIDIA CORPORATION.  All rights reserved.
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

//! @file The cub::BlockExchange class provides :ref:`collective <collective-primitives>` methods for
//!       rearranging data partitioned across a CUDA thread block.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/uninitialized_copy.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/warp_exchange.cuh>

CUB_NAMESPACE_BEGIN

//! @rst
//! The BlockExchange class provides :ref:`collective <collective-primitives>` methods for rearranging data partitioned
//! across a CUDA thread block.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! - It is commonplace for blocks of threads to rearrange data items between
//!   threads.  For example, the device-accessible memory subsystem prefers access patterns
//!   where data items are "striped" across threads (where consecutive threads access consecutive items),
//!   yet most block-wide operations prefer a "blocked" partitioning of items across threads
//!   (where consecutive items belong to a single thread).
//! - BlockExchange supports the following types of data exchanges:
//!
//!   - Transposing between :ref:`blocked <flexible-data-arrangement>` and :ref:`striped <flexible-data-arrangement>`
//!     arrangements
//!   - Transposing between :ref:`blocked <flexible-data-arrangement>` and
//!     :ref:`warp-striped <flexible-data-arrangement>`  arrangements
//!   - Scattering ranked items to a :ref:`blocked arrangement <flexible-data-arrangement>`
//!   - Scattering ranked items to a :ref:`striped arrangement <flexible-data-arrangement>`
//!
//! - @rowmajor
//!
//! A Simple Example
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! @blockcollective{BlockExchange}
//!
//! The code snippet below illustrates the conversion from a "blocked" to a "striped" arrangement
//! of 512 integer items partitioned across 128 threads where each thread owns 4 items.
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_exchange.cuh>
//!
//!    __global__ void ExampleKernel(int *d_data, ...)
//!    {
//!        // Specialize BlockExchange for a 1D block of 128 threads owning 4 integer items each
//!        typedef cub::BlockExchange<int, 128, 4> BlockExchange;
//!
//!        // Allocate shared memory for BlockExchange
//!        __shared__ typename BlockExchange::TempStorage temp_storage;
//!
//!        // Load a tile of data striped across threads
//!        int thread_data[4];
//!        cub::LoadDirectStriped<128>(threadIdx.x, d_data, thread_data);
//!
//!        // Collectively exchange data into a blocked arrangement across threads
//!        BlockExchange(temp_storage).StripedToBlocked(thread_data);
//!
//! Suppose the set of striped input ``thread_data`` across the block of threads is
//! ``{ [0,128,256,384], [1,129,257,385], ..., [127,255,383,511] }``.
//! The corresponding output ``thread_data`` in those threads will be
//! ``{ [0,1,2,3], [4,5,6,7], [8,9,10,11], ..., [508,509,510,511] }``.
//!
//! Performance Considerations
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! - Proper device-specific padding ensures zero bank conflicts for most types.
//!
//! Re-using dynamically allocating shared memory
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! The ``block/example_block_reduce_dyn_smem.cu`` example illustrates usage of dynamically shared memory with
//! BlockReduce and how to re-purpose the same memory region. This example can be easily adapted to
//! the storage required by BlockExchange.
//! @endrst
//!
//! @tparam T
//!   The data type to be exchanged
//!
//! @tparam BLOCK_DIM_X
//!   The thread block length in threads along the X dimension
//!
//! @tparam ITEMS_PER_THREAD
//!   The number of items partitioned onto each thread.
//!
//! @tparam WARP_TIME_SLICING
//!   **[optional]** When `true`, only use enough shared memory for a single warp's worth of tile data,
//!   time-slicing the block-wide exchange over multiple synchronized rounds.
//!   Yields a smaller memory footprint at the expense of decreased parallelism. (Default: false)
//!
//! @tparam BLOCK_DIM_Y
//!   **[optional]** The thread block length in threads along the Y dimension (default: 1)
//!
//! @tparam BLOCK_DIM_Z
//!   **[optional]** The thread block length in threads along the Z dimension (default: 1)
//!
//! @tparam LEGACY_PTX_ARCH
//!   <b>[optional]</b> Unused.
template <typename InputT,
          int BLOCK_DIM_X,
          int ITEMS_PER_THREAD,
          bool WARP_TIME_SLICING = false,
          int BLOCK_DIM_Y        = 1,
          int BLOCK_DIM_Z        = 1,
          int LEGACY_PTX_ARCH    = 0>
class BlockExchange
{
private:
    /// Constants
    enum
    {
        /// The thread block size in threads
        BLOCK_THREADS               = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z,

        LOG_WARP_THREADS            = CUB_LOG_WARP_THREADS(0),
        WARP_THREADS                = 1 << LOG_WARP_THREADS,
        WARPS                       = (BLOCK_THREADS + WARP_THREADS - 1) / WARP_THREADS,

        LOG_SMEM_BANKS              = CUB_LOG_SMEM_BANKS(0),
        SMEM_BANKS                  = 1 << LOG_SMEM_BANKS,

        TILE_ITEMS                  = BLOCK_THREADS * ITEMS_PER_THREAD,

        TIME_SLICES                 = (WARP_TIME_SLICING) ? WARPS : 1,

        TIME_SLICED_THREADS         = (WARP_TIME_SLICING) ? CUB_MIN(BLOCK_THREADS, WARP_THREADS) : BLOCK_THREADS,
        TIME_SLICED_ITEMS           = TIME_SLICED_THREADS * ITEMS_PER_THREAD,

        WARP_TIME_SLICED_THREADS    = CUB_MIN(BLOCK_THREADS, WARP_THREADS),
        WARP_TIME_SLICED_ITEMS      = WARP_TIME_SLICED_THREADS * ITEMS_PER_THREAD,

        // Insert padding to avoid bank conflicts during raking when items per thread is a power of two and > 4 (otherwise we can typically use 128b loads)
        INSERT_PADDING              = (ITEMS_PER_THREAD > 4) && (PowerOfTwo<ITEMS_PER_THREAD>::VALUE),
        PADDING_ITEMS               = (INSERT_PADDING) ? (TIME_SLICED_ITEMS >> LOG_SMEM_BANKS) : 0,
    };

    /// Shared memory storage layout type
    struct __align__(16) _TempStorage
    {
        InputT buff[TIME_SLICED_ITEMS + PADDING_ITEMS];
    };

public:

    /// @smemstorage{BlockExchange}
    struct TempStorage : Uninitialized<_TempStorage> {};

private:

    /// Shared storage reference
    _TempStorage &temp_storage;

    /// Linear thread-id
    unsigned int linear_tid;
    unsigned int lane_id;
    unsigned int warp_id;
    unsigned int warp_offset;

    /// Internal storage allocator
    _CCCL_DEVICE _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
    {
        __shared__ _TempStorage private_storage;
        return private_storage;
    }

    //! @brief Transposes data items from **blocked** arrangement to **striped** arrangement.
    //!        Specialized for no timeslicing.
    //!
    //! @param[in] input_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    //!
    //! @param[out] output_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    template <typename OutputT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void BlockedToStriped(InputT (&input_items)[ITEMS_PER_THREAD],
                                                     OutputT (&output_items)[ITEMS_PER_THREAD],
                                                     Int2Type<false> /*time_slicing*/)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = (linear_tid * ITEMS_PER_THREAD) + ITEM;
            if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
            detail::uninitialized_copy(temp_storage.buff + item_offset,
                                       input_items[ITEM]);
        }

        CTA_SYNC();

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = int(ITEM * BLOCK_THREADS) + linear_tid;
            if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
            output_items[ITEM] = temp_storage.buff[item_offset];
        }
    }

    //! @brief Transposes data items from **blocked** arrangement to **striped**
    //!        arrangement. Specialized for warp-timeslicing.
    //!
    //! @param[in] input_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    //!
    //! @param[out] output_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    template <typename OutputT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void BlockedToStriped(InputT (&input_items)[ITEMS_PER_THREAD],
                                                     OutputT (&output_items)[ITEMS_PER_THREAD],
                                                     Int2Type<true> /*time_slicing*/)
    {
        InputT temp_items[ITEMS_PER_THREAD];

        #pragma unroll
        for (int SLICE = 0; SLICE < TIME_SLICES; SLICE++)
        {
            const int SLICE_OFFSET  = SLICE * TIME_SLICED_ITEMS;
            const int SLICE_OOB     = SLICE_OFFSET + TIME_SLICED_ITEMS;

            CTA_SYNC();

            if (warp_id == SLICE)
            {
                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = (lane_id * ITEMS_PER_THREAD) + ITEM;
                    if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                    detail::uninitialized_copy(temp_storage.buff + item_offset,
                                               input_items[ITEM]);
                }
            }

            CTA_SYNC();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                // Read a strip of items
                const int STRIP_OFFSET  = ITEM * BLOCK_THREADS;
                const int STRIP_OOB     = STRIP_OFFSET + BLOCK_THREADS;

                if ((SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET))
                {
                    int item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET;
                    if ((item_offset >= 0) && (item_offset < TIME_SLICED_ITEMS))
                    {
                        if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                        temp_items[ITEM] = temp_storage.buff[item_offset];
                    }
                }
            }
        }

        // Copy
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            output_items[ITEM] = temp_items[ITEM];
        }
    }

    //! @brief Transposes data items from **blocked** arrangement to **warp-striped** arrangement.
    //!        Specialized for no timeslicing
    //!
    //! @param[in] input_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    //!
    //! @param[out] output_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    template <typename OutputT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void BlockedToWarpStriped(InputT (&input_items)[ITEMS_PER_THREAD],
                                                         OutputT (&output_items)[ITEMS_PER_THREAD],
                                                         Int2Type<false> /*time_slicing*/)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = warp_offset + ITEM + (lane_id * ITEMS_PER_THREAD);
            if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
            detail::uninitialized_copy(temp_storage.buff + item_offset,
                                       input_items[ITEM]);
        }

        WARP_SYNC(0xffffffff);

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = warp_offset + (ITEM * WARP_TIME_SLICED_THREADS) + lane_id;
            if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
            output_items[ITEM] = temp_storage.buff[item_offset];
        }
    }

    //! @brief Transposes data items from **blocked** arrangement to **warp-striped** arrangement.
    //!        Specialized for warp-timeslicing
    //!
    //! @param[in] input_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    //!
    //! @param[out] output_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    template <typename OutputT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void BlockedToWarpStriped(InputT (&input_items)[ITEMS_PER_THREAD],
                                                         OutputT (&output_items)[ITEMS_PER_THREAD],
                                                         Int2Type<true> /*time_slicing*/)
    {
        if (warp_id == 0)
        {
            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = ITEM + (lane_id * ITEMS_PER_THREAD);
                if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                detail::uninitialized_copy(temp_storage.buff + item_offset,
                                           input_items[ITEM]);
            }

            WARP_SYNC(0xffffffff);

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = (ITEM * WARP_TIME_SLICED_THREADS) + lane_id;
                if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                output_items[ITEM] = temp_storage.buff[item_offset];
            }
        }

        #pragma unroll
        for (unsigned int SLICE = 1; SLICE < TIME_SLICES; ++SLICE)
        {
            CTA_SYNC();

            if (warp_id == SLICE)
            {
                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = ITEM + (lane_id * ITEMS_PER_THREAD);
                    if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                    detail::uninitialized_copy(temp_storage.buff + item_offset,
                                               input_items[ITEM]);
                }

                WARP_SYNC(0xffffffff);

                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = (ITEM * WARP_TIME_SLICED_THREADS) + lane_id;
                    if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                    output_items[ITEM] = temp_storage.buff[item_offset];
                }
            }
        }
    }

    //! @brief Transposes data items from **striped** arrangement to **blocked** arrangement.
    //!        Specialized for no timeslicing.
    //!
    //! @param[in] input_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    //!
    //! @param[out] output_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    template <typename OutputT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void StripedToBlocked(InputT (&input_items)[ITEMS_PER_THREAD],
                                                     OutputT (&output_items)[ITEMS_PER_THREAD],
                                                     Int2Type<false> /*time_slicing*/)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = int(ITEM * BLOCK_THREADS) + linear_tid;
            if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
            detail::uninitialized_copy(temp_storage.buff + item_offset,
                                       input_items[ITEM]);
        }

        CTA_SYNC();

        // No timeslicing
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = (linear_tid * ITEMS_PER_THREAD) + ITEM;
            if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
            output_items[ITEM] = temp_storage.buff[item_offset];
        }
    }

    //! @brief Transposes data items from **striped** arrangement to **blocked** arrangement.
    //!        Specialized for warp-timeslicing.
    //!
    //! @param[in] input_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    //!
    //! @param[out] output_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    template <typename OutputT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void StripedToBlocked(InputT (&input_items)[ITEMS_PER_THREAD],
                                                     OutputT (&output_items)[ITEMS_PER_THREAD],
                                                     Int2Type<true> /*time_slicing*/)
    {
        // Warp time-slicing
        InputT temp_items[ITEMS_PER_THREAD];

        #pragma unroll
        for (int SLICE = 0; SLICE < TIME_SLICES; SLICE++)
        {
            const int SLICE_OFFSET  = SLICE * TIME_SLICED_ITEMS;
            const int SLICE_OOB     = SLICE_OFFSET + TIME_SLICED_ITEMS;

            CTA_SYNC();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                // Write a strip of items
                const int STRIP_OFFSET  = ITEM * BLOCK_THREADS;
                const int STRIP_OOB     = STRIP_OFFSET + BLOCK_THREADS;

                if ((SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET))
                {
                    int item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET;
                    if ((item_offset >= 0) && (item_offset < TIME_SLICED_ITEMS))
                    {
                        if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                        detail::uninitialized_copy(temp_storage.buff +
                                                     item_offset,
                                                   input_items[ITEM]);
                    }
                }
            }

            CTA_SYNC();

            if (warp_id == SLICE)
            {
                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = (lane_id * ITEMS_PER_THREAD) + ITEM;
                    if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                    temp_items[ITEM] = temp_storage.buff[item_offset];
                }
            }
        }

        // Copy
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            output_items[ITEM] = temp_items[ITEM];
        }
    }

    //! @brief Transposes data items from **warp-striped** arrangement to **blocked** arrangement.
    //!        Specialized for no timeslicing
    //!
    //! @param[in] input_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    //!
    //! @param[out] output_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    template <typename OutputT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void WarpStripedToBlocked(InputT (&input_items)[ITEMS_PER_THREAD],
                                                         OutputT (&output_items)[ITEMS_PER_THREAD],
                                                         Int2Type<false> /*time_slicing*/)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = warp_offset + (ITEM * WARP_TIME_SLICED_THREADS) + lane_id;
            if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
            detail::uninitialized_copy(temp_storage.buff + item_offset,
                                       input_items[ITEM]);
        }

        WARP_SYNC(0xffffffff);

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = warp_offset + ITEM + (lane_id * ITEMS_PER_THREAD);
            if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
            detail::uninitialized_copy(output_items + ITEM,
                                       temp_storage.buff[item_offset]);
        }
    }

    //! @brief Transposes data items from **warp-striped** arrangement to **blocked** arrangement.
    //!        Specialized for warp-timeslicing
    //!
    //! @param[in] input_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    //!
    //! @param[out] output_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    template <typename OutputT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void WarpStripedToBlocked(InputT (&input_items)[ITEMS_PER_THREAD],
                                                         OutputT (&output_items)[ITEMS_PER_THREAD],
                                                         Int2Type<true> /*time_slicing*/)
    {
        #pragma unroll
        for (unsigned int SLICE = 0; SLICE < TIME_SLICES; ++SLICE)
        {
            CTA_SYNC();

            if (warp_id == SLICE)
            {
                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = (ITEM * WARP_TIME_SLICED_THREADS) + lane_id;
                    if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                    detail::uninitialized_copy(temp_storage.buff + item_offset,
                                               input_items[ITEM]);
                }

                WARP_SYNC(0xffffffff);

                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = ITEM + (lane_id * ITEMS_PER_THREAD);
                    if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                    output_items[ITEM] = temp_storage.buff[item_offset];
                }
            }
        }
    }

    //! @brief Exchanges data items annotated by rank into **blocked** arrangement.
    //!        Specialized for no timeslicing.
    //!
    //! @param[in] input_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    //!
    //! @param[out] output_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    //!
    //! @param[in] ranks
    //!   Corresponding scatter ranks
    template <typename OutputT, typename OffsetT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterToBlocked(InputT (&input_items)[ITEMS_PER_THREAD],
                                                     OutputT (&output_items)[ITEMS_PER_THREAD],
                                                     OffsetT (&ranks)[ITEMS_PER_THREAD],
                                                     Int2Type<false> /*time_slicing*/)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = ranks[ITEM];
            if (INSERT_PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
            detail::uninitialized_copy(temp_storage.buff + item_offset,
                                       input_items[ITEM]);
        }

        CTA_SYNC();

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = (linear_tid * ITEMS_PER_THREAD) + ITEM;
            if (INSERT_PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
            output_items[ITEM] = temp_storage.buff[item_offset];
        }
    }

    //! @brief Exchanges data items annotated by rank into **blocked** arrangement.
    //!        Specialized for warp-timeslicing.
    //!
    //! @param[in] input_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    //!
    //! @param[out] output_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    //!
    //! @param[in] ranks
    //!   Corresponding scatter ranks
    template <typename OutputT, typename OffsetT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterToBlocked(InputT (&input_items)[ITEMS_PER_THREAD],
                                                     OutputT (&output_items)[ITEMS_PER_THREAD],
                                                     OffsetT ranks[ITEMS_PER_THREAD],
                                                     Int2Type<true> /*time_slicing*/)
    {
        InputT temp_items[ITEMS_PER_THREAD];

        #pragma unroll
        for (int SLICE = 0; SLICE < TIME_SLICES; SLICE++)
        {
            CTA_SYNC();

            const int SLICE_OFFSET = TIME_SLICED_ITEMS * SLICE;

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = ranks[ITEM] - SLICE_OFFSET;
                if ((item_offset >= 0) && (item_offset < WARP_TIME_SLICED_ITEMS))
                {
                    if (INSERT_PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                    detail::uninitialized_copy(temp_storage.buff + item_offset,
                                               input_items[ITEM]);
                }
            }

            CTA_SYNC();

            if (warp_id == SLICE)
            {
                #pragma unroll
                for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
                {
                    int item_offset = (lane_id * ITEMS_PER_THREAD) + ITEM;
                    if (INSERT_PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                    temp_items[ITEM] = temp_storage.buff[item_offset];
                }
            }
        }

        // Copy
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            output_items[ITEM] = temp_items[ITEM];
        }
    }

    //! @brief Exchanges data items annotated by rank into **striped** arrangement.
    //!        Specialized for no timeslicing.
    //!
    //! @param[in] input_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    //!
    //! @param[out] output_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    //!
    //! @param[in] ranks
    //!   Corresponding scatter ranks
    template <typename OutputT, typename OffsetT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterToStriped(InputT (&input_items)[ITEMS_PER_THREAD],
                                                     OutputT (&output_items)[ITEMS_PER_THREAD],
                                                     OffsetT (&ranks)[ITEMS_PER_THREAD],
                                                     Int2Type<false> /*time_slicing*/)
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = ranks[ITEM];
            if (INSERT_PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
            detail::uninitialized_copy(temp_storage.buff + item_offset,
                                       input_items[ITEM]);
        }

        CTA_SYNC();

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = int(ITEM * BLOCK_THREADS) + linear_tid;
            if (INSERT_PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
            output_items[ITEM] = temp_storage.buff[item_offset];
        }
    }

    //! @brief Exchanges data items annotated by rank into **striped** arrangement.
    //!        Specialized for warp-timeslicing.
    //!
    //! @param[in] input_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    //!
    //! @param[out] output_items
    //!   Items to exchange, converting between **blocked** and **striped** arrangements.
    //!
    //! @param[in] ranks
    //!   Corresponding scatter ranks
    template <typename OutputT, typename OffsetT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterToStriped(InputT (&input_items)[ITEMS_PER_THREAD],
                                                     OutputT (&output_items)[ITEMS_PER_THREAD],
                                                     OffsetT (&ranks)[ITEMS_PER_THREAD],
                                                     Int2Type<true> /*time_slicing*/)
    {
        InputT temp_items[ITEMS_PER_THREAD];

        #pragma unroll
        for (int SLICE = 0; SLICE < TIME_SLICES; SLICE++)
        {
            const int SLICE_OFFSET  = SLICE * TIME_SLICED_ITEMS;
            const int SLICE_OOB     = SLICE_OFFSET + TIME_SLICED_ITEMS;

            CTA_SYNC();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                int item_offset = ranks[ITEM] - SLICE_OFFSET;
                if ((item_offset >= 0) && (item_offset < WARP_TIME_SLICED_ITEMS))
                {
                    if (INSERT_PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
                    detail::uninitialized_copy(temp_storage.buff + item_offset,
                                               input_items[ITEM]);
                }
            }

            CTA_SYNC();

            #pragma unroll
            for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
            {
                // Read a strip of items
                const int STRIP_OFFSET  = ITEM * BLOCK_THREADS;
                const int STRIP_OOB     = STRIP_OFFSET + BLOCK_THREADS;

                if ((SLICE_OFFSET < STRIP_OOB) && (SLICE_OOB > STRIP_OFFSET))
                {
                    int item_offset = STRIP_OFFSET + linear_tid - SLICE_OFFSET;
                    if ((item_offset >= 0) && (item_offset < TIME_SLICED_ITEMS))
                    {
                        if (INSERT_PADDING) item_offset += item_offset >> LOG_SMEM_BANKS;
                        temp_items[ITEM] = temp_storage.buff[item_offset];
                    }
                }
            }
        }

        // Copy
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            output_items[ITEM] = temp_items[ITEM];
        }
    }


public:

    //! @name Collective constructors
    //! @{

    /**
     * @brief Collective constructor using a private static allocation of shared memory as temporary storage.
     */
    _CCCL_DEVICE _CCCL_FORCEINLINE BlockExchange()
    :
        temp_storage(PrivateStorage()),
        linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z)),
        lane_id(LaneId()),
        warp_id((WARPS == 1) ? 0 : linear_tid / WARP_THREADS),
        warp_offset(warp_id * WARP_TIME_SLICED_ITEMS)
    {}

    /**
     * @brief Collective constructor using the specified memory allocation as temporary storage.
     *
     * @param[in] temp_storage
     *   Reference to memory allocation having layout type TempStorage
     */
    _CCCL_DEVICE _CCCL_FORCEINLINE BlockExchange(TempStorage &temp_storage)
        : temp_storage(temp_storage.Alias())
        , linear_tid(RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z))
        , lane_id(LaneId())
        , warp_id((WARPS == 1) ? 0 : linear_tid / WARP_THREADS)
        , warp_offset(warp_id * WARP_TIME_SLICED_ITEMS)
    {}


    //! @} end member group
    //! @name Structured exchanges
    //! @{

    //! @rst
    //! Transposes data items from **striped** arrangement to **blocked** arrangement.
    //!
    //! - @smemreuse
    //!
    //! Snippet
    //! +++++++
    //!
    //! The code snippet below illustrates the conversion from a "striped" to a "blocked" arrangement
    //! of 512 integer items partitioned across 128 threads where each thread owns 4 items.
    //!
    //! .. code-block:: c++
    //!
    //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_exchange.cuh>
    //!
    //!    __global__ void ExampleKernel(int *d_data, ...)
    //!    {
    //!        // Specialize BlockExchange for a 1D block of 128 threads owning 4 integer items each
    //!        typedef cub::BlockExchange<int, 128, 4> BlockExchange;
    //!
    //!        // Allocate shared memory for BlockExchange
    //!        __shared__ typename BlockExchange::TempStorage temp_storage;
    //!
    //!        // Load a tile of ordered data into a striped arrangement across block threads
    //!        int thread_data[4];
    //!        cub::LoadDirectStriped<128>(threadIdx.x, d_data, thread_data);
    //!
    //!        // Collectively exchange data into a blocked arrangement across threads
    //!        BlockExchange(temp_storage).StripedToBlocked(thread_data, thread_data);
    //!
    //! Suppose the set of striped input ``thread_data`` across the block of threads is
    //! ``{ [0,128,256,384], [1,129,257,385], ..., [127,255,383,511] }`` after loading from
    //! device-accessible memory. The corresponding output ``thread_data`` in those threads will be
    //! ``{ [0,1,2,3], [4,5,6,7], [8,9,10,11], ..., [508,509,510,511] }``.
    //! @endrst
    //!
    //! @param[in] input_items
    //!   Items to exchange, converting between **striped** and **blocked** arrangements.
    //!
    //! @param[out] output_items
    //!   Items from exchange, converting between **striped** and **blocked** arrangements.
    template <typename OutputT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void StripedToBlocked(InputT (&input_items)[ITEMS_PER_THREAD],
                                                     OutputT (&output_items)[ITEMS_PER_THREAD])
    {
        StripedToBlocked(input_items, output_items, Int2Type<WARP_TIME_SLICING>());
    }

    //! @rst
    //! Transposes data items from **blocked** arrangement to **striped** arrangement.
    //!
    //! - @smemreuse
    //!
    //! Snippet
    //! +++++++
    //!
    //! The code snippet below illustrates the conversion from a "blocked" to a "striped" arrangement
    //! of 512 integer items partitioned across 128 threads where each thread owns 4 items.
    //!
    //! .. code-block:: c++
    //!
    //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_exchange.cuh>
    //!
    //!    __global__ void ExampleKernel(int *d_data, ...)
    //!    {
    //!        // Specialize BlockExchange for a 1D block of 128 threads owning 4 integer items each
    //!        typedef cub::BlockExchange<int, 128, 4> BlockExchange;
    //!
    //!        // Allocate shared memory for BlockExchange
    //!        __shared__ typename BlockExchange::TempStorage temp_storage;
    //!
    //!        // Obtain a segment of consecutive items that are blocked across threads
    //!        int thread_data[4];
    //!        ...
    //!
    //!        // Collectively exchange data into a striped arrangement across threads
    //!        BlockExchange(temp_storage).BlockedToStriped(thread_data, thread_data);
    //!
    //!        // Store data striped across block threads into an ordered tile
    //!        cub::StoreDirectStriped<STORE_DEFAULT, 128>(threadIdx.x, d_data, thread_data);
    //!
    //! Suppose the set of blocked input ``thread_data`` across the block of threads is
    //! ``{ [0,1,2,3], [4,5,6,7], [8,9,10,11], ..., [508,509,510,511] }``.
    //! The corresponding output ``thread_data`` in those threads will be
    //! ``{ [0,128,256,384], [1,129,257,385], ..., [127,255,383,511] }`` in
    //! preparation for storing to device-accessible memory.
    //! @endrst
    //!
    //! @param[in] input_items
    //!   Items to exchange, converting between **striped** and **blocked** arrangements.
    //!
    //! @param[out] output_items
    //!   Items from exchange, converting between **striped** and **blocked** arrangements.
    template <typename OutputT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void BlockedToStriped(InputT (&input_items)[ITEMS_PER_THREAD],
                                                     OutputT (&output_items)[ITEMS_PER_THREAD])
    {
        BlockedToStriped(input_items, output_items, Int2Type<WARP_TIME_SLICING>());
    }

    //! @rst
    //! Transposes data items from **warp-striped** arrangement to **blocked** arrangement.
    //!
    //! - @smemreuse
    //!
    //!
    //! Snippet
    //! +++++++
    //!
    //! The code snippet below illustrates the conversion from a "warp-striped" to a "blocked"
    //! arrangement of 512 integer items partitioned across 128 threads where each thread owns 4
    //! items.
    //!
    //! .. code-block:: c++
    //!
    //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_exchange.cuh>
    //!
    //!    __global__ void ExampleKernel(int *d_data, ...)
    //!    {
    //!        // Specialize BlockExchange for a 1D block of 128 threads owning 4 integer items each
    //!        typedef cub::BlockExchange<int, 128, 4> BlockExchange;
    //!
    //!        // Allocate shared memory for BlockExchange
    //!        __shared__ typename BlockExchange::TempStorage temp_storage;
    //!
    //!        // Load a tile of ordered data into a warp-striped arrangement across warp threads
    //!        int thread_data[4];
    //!        cub::LoadSWarptriped<LOAD_DEFAULT>(threadIdx.x, d_data, thread_data);
    //!
    //!        // Collectively exchange data into a blocked arrangement across threads
    //!        BlockExchange(temp_storage).WarpStripedToBlocked(thread_data);
    //!
    //! Suppose the set of warp-striped input ``thread_data`` across the block of threads is
    //! ``{ [0,32,64,96], [1,33,65,97], [2,34,66,98], ..., [415,447,479,511] }``
    //! after loading from device-accessible memory. (The first 128 items are striped across
    //! the first warp of 32 threads, the second 128 items are striped across the second warp, etc.)
    //! The corresponding output ``thread_data`` in those threads will be
    //! ``{ [0,1,2,3], [4,5,6,7], [8,9,10,11], ..., [508,509,510,511] }``.
    //! @endrst
    //!
    //! @param[in] input_items
    //!   Items to exchange, converting between **striped** and **blocked** arrangements.
    //!
    //! @param[out] output_items
    //!   Items from exchange, converting between **striped** and **blocked** arrangements.
    template <typename OutputT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void WarpStripedToBlocked(InputT (&input_items)[ITEMS_PER_THREAD],
                                                         OutputT (&output_items)[ITEMS_PER_THREAD])
    {
        WarpStripedToBlocked(input_items, output_items, Int2Type<WARP_TIME_SLICING>());
    }

    //! @rst
    //! Transposes data items from **blocked** arrangement to **warp-striped** arrangement.
    //!
    //! - @smemreuse
    //!
    //!
    //! Snippet
    //! +++++++
    //!
    //! The code snippet below illustrates the conversion from a "blocked" to a "warp-striped"
    //! arrangement of 512 integer items partitioned across 128 threads where each thread owns 4
    //! items.
    //!
    //! .. code-block:: c++
    //!
    //!    #include <cub/cub.cuh>   // or equivalently <cub/block/block_exchange.cuh>
    //!
    //!    __global__ void ExampleKernel(int *d_data, ...)
    //!    {
    //!        // Specialize BlockExchange for a 1D block of 128 threads owning 4 integer items each
    //!        typedef cub::BlockExchange<int, 128, 4> BlockExchange;
    //!
    //!        // Allocate shared memory for BlockExchange
    //!        __shared__ typename BlockExchange::TempStorage temp_storage;
    //!
    //!        // Obtain a segment of consecutive items that are blocked across threads
    //!        int thread_data[4];
    //!        ...
    //!
    //!        // Collectively exchange data into a warp-striped arrangement across threads
    //!        BlockExchange(temp_storage).BlockedToWarpStriped(thread_data, thread_data);
    //!
    //!        // Store data striped across warp threads into an ordered tile
    //!        cub::StoreDirectStriped<STORE_DEFAULT, 128>(threadIdx.x, d_data, thread_data);
    //!
    //! Suppose the set of blocked input ``thread_data`` across the block of threads is
    //! ``{ [0,1,2,3], [4,5,6,7], [8,9,10,11], ..., [508,509,510,511] }``.
    //! The corresponding output ``thread_data`` in those threads will be
    //! ``{ [0,32,64,96], [1,33,65,97], [2,34,66,98], ..., [415,447,479,511] }``
    //! in preparation for storing to device-accessible memory. (The first 128 items are striped
    //! across the first warp of 32 threads, the second 128 items are striped across the second warp, etc.)
    //! @endrst
    //!
    //! @param[in] input_items
    //!   Items to exchange, converting between **striped** and **blocked** arrangements.
    //!
    //! @param[out] output_items
    //!   Items from exchange, converting between **striped** and **blocked** arrangements.
    template <typename OutputT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void BlockedToWarpStriped(InputT (&input_items)[ITEMS_PER_THREAD],
                                                         OutputT (&output_items)[ITEMS_PER_THREAD])
    {
        BlockedToWarpStriped(input_items, output_items, Int2Type<WARP_TIME_SLICING>());
    }

    //! @}  end member group
    //! @name Scatter exchanges
    //! @{

    //! @rst
    //! Exchanges data items annotated by rank into **blocked** arrangement.
    //!
    //! - @smemreuse
    //! @endrst
    //!
    //! @tparam OffsetT
    //!   **[inferred]** Signed integer type for local offsets
    //!
    //! @param[in] input_items
    //!   Items to exchange, converting between **striped** and **blocked** arrangements.
    //!
    //! @param[out] output_items
    //!   Items from exchange, converting between **striped** and **blocked** arrangements.
    //!
    //! @param[in] ranks
    //!   Corresponding scatter ranks
    template <typename OutputT, typename OffsetT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterToBlocked(InputT (&input_items)[ITEMS_PER_THREAD],
                                                     OutputT (&output_items)[ITEMS_PER_THREAD],
                                                     OffsetT (&ranks)[ITEMS_PER_THREAD])
    {
        ScatterToBlocked(input_items, output_items, ranks, Int2Type<WARP_TIME_SLICING>());
    }

    //! @rst
    //! Exchanges data items annotated by rank into **striped** arrangement.
    //!
    //! - @smemreuse
    //!
    //! @endrst
    //!
    //! @tparam OffsetT
    //!   **[inferred]** Signed integer type for local offsets
    //!
    //! @param[in] input_items
    //!   Items to exchange, converting between **striped** and **blocked** arrangements.
    //!
    //! @param[out] output_items
    //!   Items from exchange, converting between **striped** and **blocked** arrangements.
    //!
    //! @param[in] ranks
    //!   Corresponding scatter ranks
    template <typename OutputT, typename OffsetT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterToStriped(InputT (&input_items)[ITEMS_PER_THREAD],
                                                     OutputT (&output_items)[ITEMS_PER_THREAD],
                                                     OffsetT (&ranks)[ITEMS_PER_THREAD])
    {
        ScatterToStriped(input_items, output_items, ranks, Int2Type<WARP_TIME_SLICING>());
    }

    //! @rst
    //! Exchanges data items annotated by rank into **striped** arrangement. Items with rank -1 are not exchanged.
    //!
    //! - @smemreuse
    //!
    //! @endrst
    //!
    //! @tparam OffsetT
    //!   **[inferred]** Signed integer type for local offsets
    //!
    //! @param[in] input_items
    //!   Items to exchange, converting between **striped** and **blocked** arrangements.
    //!
    //! @param[out] output_items
    //!   Items from exchange, converting between **striped** and **blocked** arrangements.
    //!
    //! @param[in] ranks
    //!   Corresponding scatter ranks
    template <typename OutputT, typename OffsetT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    ScatterToStripedGuarded(InputT (&input_items)[ITEMS_PER_THREAD],
                            OutputT (&output_items)[ITEMS_PER_THREAD],
                            OffsetT (&ranks)[ITEMS_PER_THREAD])
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = ranks[ITEM];
            if (INSERT_PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
            if (ranks[ITEM] >= 0)
                temp_storage.buff[item_offset] = input_items[ITEM];
        }

        CTA_SYNC();

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = int(ITEM * BLOCK_THREADS) + linear_tid;
            if (INSERT_PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
            output_items[ITEM] = temp_storage.buff[item_offset];
        }
    }

    //! @rst
    //! Exchanges valid data items annotated by rank into **striped** arrangement.
    //!
    //! - @smemreuse
    //!
    //! @endrst
    //!
    //! @tparam OffsetT
    //!   **[inferred]** Signed integer type for local offsets
    //!
    //! @tparam ValidFlag
    //!   **[inferred]** FlagT type denoting which items are valid
    //!
    //! @param[in] input_items
    //!   Items to exchange, converting between **striped** and **blocked** arrangements.
    //!
    //! @param[out] output_items
    //!   Items from exchange, converting between **striped** and **blocked** arrangements.
    //!
    //! @param[in] ranks
    //!   Corresponding scatter ranks
    //!
    //! @param[in] is_valid
    //!   Corresponding flag denoting item validity
    template <typename OutputT, typename OffsetT, typename ValidFlag>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    ScatterToStripedFlagged(InputT (&input_items)[ITEMS_PER_THREAD],
                            OutputT (&output_items)[ITEMS_PER_THREAD],
                            OffsetT (&ranks)[ITEMS_PER_THREAD],
                            ValidFlag (&is_valid)[ITEMS_PER_THREAD])
    {
        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = ranks[ITEM];
            if (INSERT_PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
            if (is_valid[ITEM])
                temp_storage.buff[item_offset] = input_items[ITEM];
        }

        CTA_SYNC();

        #pragma unroll
        for (int ITEM = 0; ITEM < ITEMS_PER_THREAD; ITEM++)
        {
            int item_offset = int(ITEM * BLOCK_THREADS) + linear_tid;
            if (INSERT_PADDING) item_offset = SHR_ADD(item_offset, LOG_SMEM_BANKS, item_offset);
            output_items[ITEM] = temp_storage.buff[item_offset];
        }
    }

    //! @}  end member group

#ifndef DOXYGEN_SHOULD_SKIP_THIS    // Do not document

    /**
     * @param[in-out] items
     *   Items to exchange, converting between **striped** and **blocked** arrangements.
     */
    _CCCL_DEVICE _CCCL_FORCEINLINE void StripedToBlocked(InputT (&items)[ITEMS_PER_THREAD])
    {
        StripedToBlocked(items, items);
    }

    /**
     * @param[in-out] items
     *   Items to exchange, converting between **striped** and **blocked** arrangements.
     */
    _CCCL_DEVICE _CCCL_FORCEINLINE void BlockedToStriped(InputT (&items)[ITEMS_PER_THREAD])
    {
        BlockedToStriped(items, items);
    }

    /**
     * @param[in-out] items
     *   Items to exchange, converting between **striped** and **blocked** arrangements.
     */
    _CCCL_DEVICE _CCCL_FORCEINLINE void WarpStripedToBlocked(InputT (&items)[ITEMS_PER_THREAD])
    {
        WarpStripedToBlocked(items, items);
    }

    /**
     * @param[in-out] items
     *   Items to exchange, converting between **striped** and **blocked** arrangements.
     */
    _CCCL_DEVICE _CCCL_FORCEINLINE void BlockedToWarpStriped(InputT (&items)[ITEMS_PER_THREAD])
    {
        BlockedToWarpStriped(items, items);
    }

    /**
     * @param[in-out] items
     *   Items to exchange, converting between **striped** and **blocked** arrangements.
     *
     * @param[in] ranks
     *   Corresponding scatter ranks
     */
    template <typename OffsetT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterToBlocked(InputT (&items)[ITEMS_PER_THREAD],
                                                     OffsetT (&ranks)[ITEMS_PER_THREAD])
    {
        ScatterToBlocked(items, items, ranks);
    }

    /**
     * @param[in-out] items
     *   Items to exchange, converting between **striped** and **blocked** arrangements.
     *
     * @param[in] ranks
     *   Corresponding scatter ranks
     */
    template <typename OffsetT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterToStriped(InputT (&items)[ITEMS_PER_THREAD],
                                                     OffsetT (&ranks)[ITEMS_PER_THREAD])
    {
        ScatterToStriped(items, items, ranks);
    }

    /**
     * @param[in-out] items
     *   Items to exchange, converting between **striped** and **blocked** arrangements.
     *
     * @param[in] ranks
     *   Corresponding scatter ranks
     */
    template <typename OffsetT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterToStripedGuarded(InputT (&items)[ITEMS_PER_THREAD],
                                                            OffsetT (&ranks)[ITEMS_PER_THREAD])
    {
        ScatterToStripedGuarded(items, items, ranks);
    }

    /**
     * @param[in-out] items
     *   Items to exchange, converting between **striped** and **blocked** arrangements.
     *
     * @param[in] ranks
     *   Corresponding scatter ranks
     *
     * @param[in] is_valid
     *   Corresponding flag denoting item validity
     */
    template <typename OffsetT, typename ValidFlag>
    _CCCL_DEVICE _CCCL_FORCEINLINE void ScatterToStripedFlagged(InputT (&items)[ITEMS_PER_THREAD],
                                                            OffsetT (&ranks)[ITEMS_PER_THREAD],
                                                            ValidFlag (&is_valid)[ITEMS_PER_THREAD])
    {
        ScatterToStriped(items, items, ranks, is_valid);
    }

#endif // DOXYGEN_SHOULD_SKIP_THIS


};


CUB_NAMESPACE_END
