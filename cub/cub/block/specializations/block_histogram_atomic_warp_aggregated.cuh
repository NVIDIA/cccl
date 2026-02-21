/******************************************************************************
 * Copyright (c) 2025-, NVIDIA CORPORATION.  All rights reserved.
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
 * @file
 * The cub::BlockHistogramAtomicWarpAggregated class provides atomic-based
 * methods for constructing block-wide histograms from data samples
 * partitioned across a CUDA thread block.
 */

#pragma once

#include <cub/config.cuh>

#include <cuda/atomic>

#include <type_traits>

#include <cooperative_groups.h>

#include <cooperative_groups/reduce.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

CUB_NAMESPACE_BEGIN
namespace detail
{

namespace cg = cooperative_groups;

template <int ITEMS_PER_THREAD, int BINS, bool HISTOGRAM_IN_SMEM = false, bool SATURATE = true>
struct BlockHistogramAtomicWarpAggregated
{
  struct TempStorage
  {};

  _CCCL_DEVICE _CCCL_FORCEINLINE BlockHistogramAtomicWarpAggregated(TempStorage&) {}

  using scope_type = std::conditional_t<HISTOGRAM_IN_SMEM, ::cuda::thread_scope_block, cuda::thread_scope_device>;

  template <typename T, typename CounterT, typename = std::enable_if_t<SATURATE == false>>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Composite(T (&items)[ITEMS_PER_THREAD], CounterT* histogram)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
      cg::coalesced_group active = cg::coalesced_threads();
      T label                    = items[i];
      auto bin_group             = cg::labeled_partition(active, label);

      CounterT votes = cg::reduce(bin_group, CounterT{1}, cg::plus<CounterT>());

      if (bin_group.thread_rank() == 0)
      {
        cuda::atomic_ref<CounterT, scope_type> bin_ref(histogram[label]);
        bin_ref.fetch_add(votes, cuda::memory_order_relaxed);
      }
      bin_group.sync();
    }
  }

  template <typename T, typename CounterT, CounterT MAX_VALUE, typename /* enable_if_t<SATURATE == true> */ = void>
  _CCCL_DEVICE _CCCL_FORCEINLINE void Composite(T (&items)[ITEMS_PER_THREAD], CounterT* histogram)
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ITEMS_PER_THREAD; ++i)
    {
      cg::coalesced_group active = cg::coalesced_threads();
      T label                    = items[i];
      auto bin_group             = cg::labeled_partition(active, label);

      CounterT votes = cg::reduce(bin_group, CounterT{1}, cg::plus<CounterT>());

      if (bin_group.thread_rank() == 0)
      {
        cuda::atomic_ref<CounterT, scope_type> bin_ref(histogram[label]);

        CounterT old_val = bin_ref.load(cuda::memory_order_relaxed);
        CounterT desired;

        do
        {
          if (old_val == MAX_VALUE)
          {
            // We've already saturated this bin, no action is required.
            // We *could* break here, but that'll introduce an extra
            // predicate.  Note that `bin_group.thread_rank() == 0`
            // doesn't imply only one thread per warp will be active;
            // there may have been multiple matching labels, so we
            // could still have multiple threads active at this point.
            // Finally, a single CAS that simply rewrites the same value
            // is cheap, only requiring a single memory round-trip.

            desired = MAX_VALUE;
          }
          else
          {
            CounterT room      = MAX_VALUE - old_val;
            CounterT increment = (votes > room) ? room : votes;
            desired            = old_val + increment;
          }
        } while (
          !bin_ref.compare_exchange_weak(old_val, desired, cuda::memory_order_relaxed, cuda::memory_order_relaxed));
      }
      bin_group.sync();
    }
  }
};
} // namespace detail

CUB_NAMESPACE_END
