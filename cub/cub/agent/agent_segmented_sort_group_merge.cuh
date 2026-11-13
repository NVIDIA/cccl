// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Expand GroupMerge settings into capacity bins and implement the cooperative device sorting worker.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_merge_sort.cuh>
#include <cub/detail/null_type.cuh>
#include <cub/detail/segmented_sort_lrb.cuh>
#include <cub/thread/thread_sort.cuh>

#include <cuda/__cmath/ilog.h>
#include <cuda/__cmath/ipow.h>
#include <cuda/__cmath/pow2.h>
#include <cuda/__cmath/round_down.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/integer_sequence.h>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_sort_lrb
{
// Enumerate the supported thread/IPT pairs and their capacities in strictly increasing capacity order.
template <typename PolicyGetter>
struct group_merge_expansion
{
  static constexpr auto policy = PolicyGetter{}();

  static_assert(policy.min_threads > 0 && ::cuda::is_power_of_two(policy.min_threads));
  static_assert(policy.max_threads >= policy.min_threads && ::cuda::is_power_of_two(policy.max_threads)
                && policy.max_threads <= policy.block_threads);
  static_assert(policy.min_ipt > 0 && policy.max_ipt >= policy.min_ipt);

  static constexpr int ipt_choices    = policy.max_ipt - policy.min_ipt + 1;
  static constexpr int thread_choices = ::cuda::ilog2(policy.max_threads) - ::cuda::ilog2(policy.min_threads) + 1;
  static_assert(thread_choices == 1 || policy.max_ipt < 2 * policy.min_ipt,
                "GroupMerge policy: with multiple thread counts, max_ipt must be < 2 * min_ipt so that threads * ipt "
                "strictly increases across bins (bin order must equal capacity order)");

  static constexpr int size = thread_choices * ipt_choices;

  // Map a local bin to its thread/IPT pair during compile-time expansion.
  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr bin_config enumerate_shape(int bin_idx) noexcept
  {
    const int thread_config_idx = bin_idx / ipt_choices;
    const int ipt_config_idx    = bin_idx % ipt_choices;
    return {policy.min_threads * ::cuda::ipow(2, thread_config_idx), policy.min_ipt + ipt_config_idx};
  }

  // Select each pair once and reuse it for its capacity and configuration type.
  template <int Bin>
  struct bin_selection
  {
    static constexpr bin_config chosen = enumerate_shape(Bin);
    static constexpr int capacity      = chosen.threads_per_segment * chosen.items_per_thread;
    using type                         = bin_shape<chosen.threads_per_segment, chosen.items_per_thread>;
  };

  // Build the capacity-list type from the selected configurations.
  template <::cuda::std::size_t... BinIndices>
  static auto expand(::cuda::std::index_sequence<BinIndices...>)
    -> capacity_list<bin_selection<BinIndices>::capacity...>;

  using capacities = decltype(expand(::cuda::std::make_index_sequence<size>{}));

  template <int Bin>
  using shape_for_bin = typename bin_selection<Bin>::type;
};

// Sort segments cooperatively using thread-local sorting and merges across threads.
// Warp groups have independent shared key storage and never use CTA barriers.
template <typename PolicyGetter>
struct agent_group_merge
{
  static constexpr auto policy           = PolicyGetter{}();
  using expansion                        = group_merge_expansion<PolicyGetter>;
  static constexpr bool is_warp          = policy.max_threads <= 32;
  static constexpr int threads_per_group = is_warp ? 32 : policy.block_threads;

  static_assert(policy.block_threads > 0 && policy.block_threads <= 1024);
  static_assert(policy.grid_size_multiplier > 0 && policy.min_blocks_per_sm > 0);
  static_assert(policy.block_threads % 32 == 0);
  static_assert(is_warp || policy.min_threads >= 64);
  static_assert(policy.block_threads % policy.max_threads == 0);
  static_assert(policy.min_ipt >= 9 && policy.max_ipt <= 15, "GroupMerge groups support IPT 9..15");

  // Dynamic shared memory for key exchange/merging, including one prefetch slot per group.
  template <typename Key>
  [[nodiscard]] _CCCL_HOST_DEVICE_API static constexpr size_t scratch_bytes() noexcept
  {
    return sizeof(Key) * ((policy.block_threads * policy.max_ipt) + (is_warp ? policy.block_threads / 32 : 1));
  }

  // Synchronize the scheduling group: one full warp or the whole block.
  _CCCL_DEVICE_API static void sync() noexcept
  {
    if constexpr (is_warp)
    {
      __syncwarp();
    }
    else
    {
      __syncthreads();
    }
  }

  // Obtain an input segment ID from segment_idx, load the segment, sort thread-local runs,
  // merge across cooperating threads, and write the valid keys back.
  // A segment_idx of ~0u gives an unused segment zero length; its threads still reach every sync.
  // Configuration and merge rounds are uniform within the warp/block synchronization group.
  template <int ItemsPerThread, typename Key, typename BeginIterator, typename EndIterator>
  _CCCL_DEVICE_API static void sort_segment(
    unsigned segment_idx,
    bin_config config,
    const partition_result& bins,
    BeginIterator begins,
    EndIterator ends,
    const Key* in,
    Key* out,
    Key* keys_shared)
  {
    const ::cuda::std::less<Key> compare_op{};

    const int thread_idx_in_group = threadIdx.x % threads_per_group;
    if constexpr (is_warp)
    {
      keys_shared += (threadIdx.x / 32) * ((32 * policy.max_ipt) + 1);
    }
    const int thread_idx_in_segment = thread_idx_in_group % config.threads_per_segment;
    const int segment_begin_idx_shared =
      ::cuda::round_down(thread_idx_in_group, config.threads_per_segment) * ItemsPerThread;

    const bool active               = segment_idx != ~0u;
    const unsigned input_segment_id = active ? bins.segment_ids[segment_idx] : 0;
    const auto begin =
      active ? begins[input_segment_id] : ::cuda::std::remove_cvref_t<decltype(begins[input_segment_id])>{};
    const int segment_size = active ? static_cast<int>(ends[input_segment_id] - begin) : 0;

    Key keys[ItemsPerThread];
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      const int local                               = (item * config.threads_per_segment) + thread_idx_in_segment;
      keys_shared[segment_begin_idx_shared + local] = local < segment_size ? in[begin + local] : Key{};
    }

    // SerialMerge prefetches one unused key beyond its runs.
    if (thread_idx_in_group == 0)
    {
      keys_shared[threads_per_group * ItemsPerThread] = Key{};
    }
    sync();
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      keys[item] = keys_shared[segment_begin_idx_shared + (thread_idx_in_segment * ItemsPerThread) + item];
    }

    if (thread_idx_in_segment * ItemsPerThread < segment_size)
    {
      Key max_key = keys[0];
      for (int item = 1; item < ItemsPerThread; ++item)
      {
        if ((thread_idx_in_segment * ItemsPerThread) + item < segment_size)
        {
          max_key = ::cuda::std::max(max_key, keys[item], compare_op);
        }
        else
        {
          keys[item] = max_key;
        }
      }
      NullType values[ItemsPerThread];
      StableOddEvenSort(keys, values, compare_op);
    }

    for (int merged_threads = 2; merged_threads <= config.threads_per_segment; merged_threads *= 2)
    {
      sync();
      for (int item = 0; item < ItemsPerThread; ++item)
      {
        keys_shared[(thread_idx_in_group * ItemsPerThread) + item] = keys[item];
      }
      sync();
      const int start                  = (thread_idx_in_group & ~(merged_threads - 1)) * ItemsPerThread;
      const int segment_end_idx_shared = segment_begin_idx_shared + segment_size;
      const int keys1_beg_shared       = (::cuda::std::min) (segment_end_idx_shared, start);
      const int keys1_end_shared =
        (::cuda::std::min) (segment_end_idx_shared, keys1_beg_shared + (merged_threads / 2 * ItemsPerThread));
      const int keys2_end_shared =
        (::cuda::std::min) (segment_end_idx_shared, keys1_end_shared + (merged_threads / 2 * ItemsPerThread));
      const int diag           = (::cuda::std::min) (keys2_end_shared - keys1_beg_shared,
                                                     (thread_idx_in_group & (merged_threads - 1)) * ItemsPerThread);
      const int partition_diag = MergePath(
        keys_shared + keys1_beg_shared,
        keys_shared + keys1_end_shared,
        keys1_end_shared - keys1_beg_shared,
        keys2_end_shared - keys1_end_shared,
        diag,
        compare_op);
      int indices[ItemsPerThread];
      SerialMerge(
        keys_shared,
        keys1_beg_shared + partition_diag,
        keys1_end_shared + diag - partition_diag,
        keys1_end_shared - keys1_beg_shared - partition_diag,
        keys2_end_shared - keys1_end_shared - diag + partition_diag,
        keys,
        indices,
        compare_op,
        Key{});
    }
    sync();
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      keys_shared[(thread_idx_in_group * ItemsPerThread) + item] = keys[item];
    }
    sync();
    for (int item = 0; item < ItemsPerThread; ++item)
    {
      const int local = (item * config.threads_per_segment) + thread_idx_in_segment;
      if (local < segment_size)
      {
        out[begin + local] = keys_shared[segment_begin_idx_shared + local];
      }
    }
    sync();
  }

  // Translate runtime IPT into a specialization with a fixed-size per-thread key array.
  template <typename Key, typename BeginIterator, typename EndIterator>
  _CCCL_DEVICE_API static void invoke(
    unsigned segment_idx,
    bin_config config,
    const partition_result& bins,
    BeginIterator begins,
    EndIterator ends,
    const Key* in,
    Key* out,
    Key* keys_shared)
  {
    // Select the fixed-size per-thread arrays for the chosen IPT.
    switch (config.items_per_thread)
    {
      case 9:
        sort_segment<9>(segment_idx, config, bins, begins, ends, in, out, keys_shared);
        break;
      case 10:
        sort_segment<10>(segment_idx, config, bins, begins, ends, in, out, keys_shared);
        break;
      case 11:
        sort_segment<11>(segment_idx, config, bins, begins, ends, in, out, keys_shared);
        break;
      case 12:
        sort_segment<12>(segment_idx, config, bins, begins, ends, in, out, keys_shared);
        break;
      case 13:
        sort_segment<13>(segment_idx, config, bins, begins, ends, in, out, keys_shared);
        break;
      case 14:
        sort_segment<14>(segment_idx, config, bins, begins, ends, in, out, keys_shared);
        break;
      case 15:
        sort_segment<15>(segment_idx, config, bins, begins, ends, in, out, keys_shared);
        break;
      default:
        _CCCL_UNREACHABLE();
    }
  }
};
} // namespace detail::segmented_sort_lrb

CUB_NAMESPACE_END
