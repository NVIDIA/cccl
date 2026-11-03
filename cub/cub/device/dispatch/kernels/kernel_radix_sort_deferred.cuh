// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/deferred_parameter.cuh>
#include <cub/device/dispatch/kernels/kernel_radix_sort.cuh>

#include <cuda/atomic>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::radix_sort
{
// The batch bin arrays are allocated separately so their size follows the policy's digit count.
struct radix_sort_batch_state
{
  ::cuda::std::uint64_t next_tile;
  ::cuda::std::uint64_t batch_idx;
  // Batch that may next use each cleared lookback buffer.
  ::cuda::std::uint64_t ready_for[2];
  // Tiles whose lookbacks and bin-offset accesses have completed.
  unsigned int completed;
};

// An empty bit range still copies separate input/output, using the device count.
template <typename KeyT, typename ValueT, typename OffsetT, typename CountT>
_CCCL_KERNEL_ATTRIBUTES void deferred_radix_sort_copy_kernel(
  KeyT* keys_out, const KeyT* keys_in, ValueT* values_out, const ValueT* values_in, CountT count)
{
  const auto num_items = static_cast<::cuda::std::uint64_t>(detail::parameter_from_device<OffsetT>(count));
  for (::cuda::std::uint64_t i = blockIdx.x * blockDim.x + threadIdx.x; i < num_items;
       i += static_cast<::cuda::std::uint64_t>(gridDim.x) * blockDim.x)
  {
    keys_out[i] = keys_in[i];
    if constexpr (!::cuda::std::is_same_v<ValueT, NullType>)
    {
      values_out[i] = values_in[i];
    }
  }
}

template <typename OffsetT>
_CCCL_KERNEL_ATTRIBUTES void deferred_radix_sort_init_kernel(
  int* lookback, int num_states, radix_sort_batch_state* state, OffsetT* batch_bins_in, const OffsetT* bins, int digits)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_states; i += blockDim.x * gridDim.x)
  {
    lookback[i] = 0;
  }
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < digits; i += blockDim.x * gridDim.x)
  {
    batch_bins_in[i] = bins[i];
  }
  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    state->next_tile    = 0;
    state->batch_idx    = 0;
    state->ready_for[0] = 0;
    state->ready_for[1] = 1;
    state->completed    = 0;
  }
}

template <typename PolicySelector,
          SortOrder Order,
          typename KeyT,
          typename ValueT,
          typename OffsetT,
          typename CountT,
          typename DecomposerT>
_CCCL_KERNEL_ATTRIBUTES
__launch_bounds__(current_policy<PolicySelector>().onesweep.threads_per_block, 2) void deferred_onesweep_kernel(
  int* lookback,
  radix_sort_batch_state* state,
  OffsetT* batch_bins_in,
  OffsetT* batch_bins_out,
  KeyT* keys_out,
  const KeyT* keys_in,
  ValueT* values_out,
  const ValueT* values_in,
  CountT count,
  int batch_tiles,
  int current_bit,
  int num_bits,
  DecomposerT decomposer)
{
  constexpr auto policy = current_policy<PolicySelector>().onesweep;
  using agent_policy    = agent_radix_sort_onesweep_policy<
    0,
    0,
    void,
    policy.rank_private_partitions,
    policy.rank_algorithm,
    policy.scan_algorithm,
    policy.store_algorithm,
    policy.radix_bits,
    NoScaling<policy.threads_per_block, policy.items_per_thread>>;
  using agent_t =
    AgentRadixSortOnesweep<agent_policy, Order == SortOrder::Descending, KeyT, ValueT, OffsetT, int, DecomposerT, true>;
  constexpr int tile_items = agent_t::TILE_ITEMS;
  constexpr int digits     = agent_t::RADIX_DIGITS;
  __shared__ typename agent_t::TempStorage storage;
  __shared__ ::cuda::std::uint64_t tile;
  __shared__ ::cuda::std::uint64_t acquired_batch;
  // Last tile to finish lookback and publish its ending bin offsets.
  __shared__ bool is_last_tile;

  const OffsetT num_items = detail::parameter_from_device<OffsetT>(count);
  // Compute the ceiling without overflowing the count's type.
  const ::cuda::std::uint64_t num_tiles = num_items / tile_items + (num_items % tile_items != 0);
  // Only the leader uses this cache. A published batch remains ready until
  // every tile in it reports prefix readiness. Acquire once per CTA and batch.
  if (threadIdx.x == 0)
  {
    acquired_batch = num_tiles;
  }

  while (true)
  {
    if (threadIdx.x == 0)
    {
      tile = ::cuda::atomic_ref<::cuda::std::uint64_t, ::cuda::thread_scope_device>{state->next_tile}.fetch_add(
        1, ::cuda::std::memory_order_relaxed);
      if (tile < num_tiles)
      {
        const auto batch_idx = tile / batch_tiles;
        if (batch_idx != acquired_batch)
        {
          while (::cuda::atomic_ref<::cuda::std::uint64_t, ::cuda::thread_scope_device>{state->batch_idx}.load(
                   ::cuda::std::memory_order_acquire)
                 != batch_idx)
          {
          }
          acquired_batch = batch_idx;
        }
      }
    }
    __syncthreads();
    if (tile >= num_tiles)
    {
      return;
    }

    const auto batch_idx   = tile / batch_tiles;
    const auto batch_begin = batch_idx * batch_tiles;
    const int buffer_idx   = static_cast<int>(batch_idx % 2);
    int* batch_lookback    = lookback + buffer_idx * batch_tiles * digits;
    const int local_tile   = static_cast<int>(tile - batch_begin);
    const int tiles_in_batch =
      static_cast<int>(::cuda::std::min(num_tiles - batch_begin, static_cast<::cuda::std::uint64_t>(batch_tiles)));
    const OffsetT batch_offset = static_cast<OffsetT>(batch_begin) * tile_items;
    const int batch_items =
      static_cast<int>(::cuda::std::min(num_items - batch_offset, static_cast<OffsetT>(batch_tiles) * tile_items));
    const ValueT* batch_values = values_in;
    if constexpr (!agent_t::KEYS_ONLY)
    {
      batch_values += batch_offset;
    }

    // Input addresses are batch-relative, output addresses and batch bins remain global.
    // Lookback entries contain only batch-local counts, never a global bucket start.
    agent_t(
      storage,
      batch_lookback,
      nullptr,
      batch_bins_out,
      batch_bins_in,
      keys_out,
      keys_in + batch_offset,
      values_out,
      batch_values,
      batch_items,
      current_bit,
      num_bits,
      decomposer,
      local_tile)
      .Process([&] {
        // The agent's pre-scatter barrier has gathered this CTA's lookback,
        // captured bin offsets, and ending-bin writes before reporting readiness.
        // The final acquire-release RMW gathers these accesses before
        // copying bins or allowing the next batch to reuse the completion counter.
        if (threadIdx.x == 0)
        {
          is_last_tile = ::cuda::atomic_ref<unsigned int, ::cuda::thread_scope_device>{state->completed}.fetch_add(
                           1, ::cuda::std::memory_order_acq_rel)
                         + 1
                      == static_cast<unsigned int>(tiles_in_batch);
        }
        __syncthreads();
        if (is_last_tile && batch_begin + tiles_in_batch < num_tiles)
        {
          // The previous retiree may still be clearing the next buffer. It remains
          // resident until its clear completes, so waiting here cannot strand it.
          if (threadIdx.x == 0)
          {
            while (
              ::cuda::atomic_ref<::cuda::std::uint64_t, ::cuda::thread_scope_device>{state->ready_for[1 - buffer_idx]}
                .load(::cuda::std::memory_order_acquire)
              != batch_idx + 1)
            {
            }
          }
          for (int i = threadIdx.x; i < digits; i += blockDim.x)
          {
            batch_bins_in[i] = batch_bins_out[i];
          }
          __syncthreads();
          if (threadIdx.x == 0)
          {
            ::cuda::atomic_ref<unsigned int, ::cuda::thread_scope_device>{state->completed}.store(
              0, ::cuda::std::memory_order_relaxed);
            ::cuda::atomic_ref<::cuda::std::uint64_t, ::cuda::thread_scope_device>{state->batch_idx}.store(
              batch_idx + 1, ::cuda::std::memory_order_release);
          }
        }
      });

    if (is_last_tile && batch_begin + tiles_in_batch < num_tiles)
    {
      // Every lookback reader finished before this CTA published the next batch.
      // Other CTAs may still scatter using private shared offsets, but none use
      // this lookback buffer. Prepare it for batch_idx + 2 after our own scatter.
      // Do not touch the active batch's bin arrays or completion counter.
      if (num_tiles - batch_begin > 2 * static_cast<::cuda::std::uint64_t>(batch_tiles))
      {
        for (int i = threadIdx.x; i < batch_tiles * digits; i += blockDim.x)
        {
          batch_lookback[i] = 0;
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
          ::cuda::atomic_ref<::cuda::std::uint64_t, ::cuda::thread_scope_device>{state->ready_for[buffer_idx]}.store(
            batch_idx + 2, ::cuda::std::memory_order_release);
        }
      }
    }
    // Keep every thread in this CTA on the same iteration, including the retiree.
    __syncthreads();
  }
}
} // namespace detail::radix_sort

CUB_NAMESPACE_END
