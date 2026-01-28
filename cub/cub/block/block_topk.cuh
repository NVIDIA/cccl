// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! The @c cub::detail::block_topk class provides a :ref:`collective <collective-primitives>` method for selecting the
//! top-k elements from a set of items within a CUDA thread block.

#pragma once

#include <cub/config.cuh>

#include <cstdint>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/block/block_exchange.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

CUB_NAMESPACE_BEGIN

namespace detail
{
enum class block_topk_algorithm
{
  air_top_k
};

template <typename T, topk::select SelectDirection>
struct twiddle_keys_in_op_t
{
  using sort_key_t = typename Traits<T>::UnsignedBits;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE sort_key_t operator()(T key) const
  {
    auto sort_key = reinterpret_cast<typename Traits<T>::UnsignedBits&>(key);
    sort_key      = Traits<T>::TwiddleIn(sort_key);
    if constexpr (SelectDirection != topk::select::min)
    {
      sort_key = ~sort_key;
    }
    return sort_key;
  }
};

template <typename T, topk::select SelectDirection>
struct twiddle_keys_out_op_t
{
  using sort_key_t = typename Traits<T>::UnsignedBits;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T operator()(sort_key_t sort_key) const
  {
    if constexpr (SelectDirection != topk::select::min)
    {
      sort_key = ~sort_key;
    }
    sort_key = Traits<T>::TwiddleOut(sort_key);
    return reinterpret_cast<T&>(sort_key);
  }
};

// TODO (elstehle): Add documentation
template <typename KeyT, int BlockThreads, int ItemsPerThread, typename ValueT = NullType, int RadixBits = 11>
class non_deterministic_air_topk
{
private:
  static constexpr int block_threads    = BlockThreads;
  static constexpr int items_per_thread = ItemsPerThread;
  static constexpr int tile_items       = block_threads * items_per_thread;
  static constexpr int num_buckets      = (1 << RadixBits);

  // Calculate number of buckets processed per thread
  static constexpr int buckets_per_thread = ::cuda::ceil_div(num_buckets, block_threads);
  static constexpr bool keys_only         = ::cuda::std::is_same_v<ValueT, NullType>;

  using histo_counter_t = ::cuda::std::uint32_t;
  using block_scan_t    = BlockScan<histo_counter_t, block_threads, BLOCK_SCAN_WARP_SCANS>;

  using block_exchange_keys_t   = BlockExchange<KeyT, block_threads, items_per_thread>;
  using block_exchange_values_t = BlockExchange<ValueT, block_threads, items_per_thread>;

  using traits                 = detail::radix::traits_t<KeyT>;
  using bit_ordered_type       = typename traits::bit_ordered_type;
  using bit_ordered_conversion = typename traits::bit_ordered_conversion_policy;
  using bit_ordered_inversion  = typename traits::bit_ordered_inversion_policy;

  using fundamental_digit_extractor_t = BFEDigitExtractor<KeyT>;

  struct TempStorage_
  {
    union
    {
      histo_counter_t histogram[num_buckets];
      typename block_exchange_keys_t::TempStorage exchange_keys;
      typename block_exchange_values_t::TempStorage exchange_values;
    };
    typename block_scan_t::TempStorage scan_temp_storage;
    struct
    {
      histo_counter_t k;
      histo_counter_t len;
      int bucket;
    } pass_state;
    histo_counter_t selected_prefix[block_threads];
    histo_counter_t candidate_prefix[block_threads];
  };

  /// Shared storage reference
  TempStorage_& storage;

  /// Linear thread-id
  unsigned int linear_tid;

  // Initialize histogram bins to zero
  _CCCL_DEVICE _CCCL_FORCEINLINE void init_histograms()
  {
    const int base = static_cast<int>(linear_tid) * buckets_per_thread;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < buckets_per_thread; ++i)
    {
      const int bin_idx = base + i;
      if (bin_idx < num_buckets)
      {
        storage.histogram[bin_idx] = 0;
      }
    }
  }

  // Compute histogram over keys. digit_extractor is a function object that returns the bin for each key.
  template <typename DigitExtractorT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void compute_histograms(
    const bit_ordered_type (&unsigned_keys)[items_per_thread],
    DigitExtractorT digit_extractor,
    bit_ordered_type prefix_mask,
    bit_ordered_type kth_key_bits)
  {
    const bit_ordered_type kth_prefix = kth_key_bits & prefix_mask;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      const bit_ordered_type key = unsigned_keys[i];
      if ((key & prefix_mask) == kth_prefix)
      {
        const auto digit = digit_extractor.Digit(key);
        atomicAdd(&storage.histogram[digit], histo_counter_t{1});
      }
    }
  }

  // Compute prefix sum over buckets
  _CCCL_DEVICE _CCCL_FORCEINLINE void compute_bin_offsets()
  {
    histo_counter_t thread_buckets[buckets_per_thread]{};
    const int base = static_cast<int>(linear_tid) * buckets_per_thread;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < buckets_per_thread; ++i)
    {
      const int bin_idx = base + i;
      if (bin_idx < num_buckets)
      {
        thread_buckets[i] = storage.histogram[bin_idx];
      }
    }

    block_scan_t(storage.scan_temp_storage).InclusiveSum(thread_buckets, thread_buckets);

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < buckets_per_thread; ++i)
    {
      const int bin_idx = base + i;
      if (bin_idx < num_buckets)
      {
        storage.histogram[bin_idx] = thread_buckets[i];
      }
    }

    __syncthreads();
  }

  // Identify the bucket that the k-th item falls into
  _CCCL_DEVICE _CCCL_FORCEINLINE void choose_bucket(histo_counter_t k)
  {
    const int base = static_cast<int>(linear_tid) * buckets_per_thread;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < buckets_per_thread; ++i)
    {
      const int bin_idx = base + i;
      if (bin_idx < num_buckets)
      {
        const histo_counter_t prev = (bin_idx == 0) ? 0 : storage.histogram[bin_idx - 1];
        const histo_counter_t cur  = storage.histogram[bin_idx];

        if (prev < k && cur >= k)
        {
          storage.pass_state.k      = k - prev;
          storage.pass_state.len    = cur - prev;
          storage.pass_state.bucket = bin_idx;
        }
      }
    }
  }

  template <detail::topk::select SelectDirection, bool HasValues, typename ValuesT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  select_impl(KeyT (&keys)[items_per_thread], ValuesT (&values)[items_per_thread], int k, int begin_bit, int end_bit)
  {
    if (k <= 0)
    {
      return;
    }

    if (begin_bit < 0)
    {
      begin_bit = 0;
    }

    const int max_bit = int(sizeof(KeyT) * 8);
    if (end_bit > max_bit)
    {
      end_bit = max_bit;
    }

    if (k >= tile_items)
    {
      return;
    }

    identity_decomposer_t decomposer;

    bit_ordered_type unsigned_keys[items_per_thread];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      bit_ordered_type raw = reinterpret_cast<bit_ordered_type&>(keys[i]);
      bit_ordered_type val = bit_ordered_conversion::to_bit_ordered(decomposer, raw);
      val                  = BaseDigitExtractor<KeyT>::ProcessFloatMinusZero(val);
      if constexpr (SelectDirection == detail::topk::select::max)
      {
        val = bit_ordered_inversion::inverse(decomposer, val);
      }
      unsigned_keys[i] = val;
    }

    bit_ordered_type kth_key_bits = 0;
    bit_ordered_type prefix_mask  = 0;

    const int total_bits = end_bit - begin_bit;
    const int num_passes = (total_bits > 0) ? ::cuda::ceil_div(total_bits, RadixBits) : 0;

    histo_counter_t current_k   = static_cast<histo_counter_t>(k);
    histo_counter_t current_len = tile_items;

    for (int pass = 0; pass < num_passes; ++pass)
    {
      const int pass_end_bit = end_bit - pass * RadixBits;
      int pass_begin_bit     = pass_end_bit - RadixBits;
      if (pass_begin_bit < begin_bit)
      {
        pass_begin_bit = begin_bit;
      }
      const int pass_bits = pass_end_bit - pass_begin_bit;
      if (pass_bits <= 0)
      {
        break;
      }

      const ::cuda::std::uint64_t pass_mask_wide =
        ((::cuda::std::uint64_t{1} << pass_bits) - ::cuda::std::uint64_t{1}) << pass_begin_bit;
      const bit_ordered_type pass_mask = static_cast<bit_ordered_type>(pass_mask_wide);

      init_histograms();
      __syncthreads();

      auto digit_extractor =
        traits::template digit_extractor<fundamental_digit_extractor_t>(pass_begin_bit, pass_bits, decomposer);
      compute_histograms(unsigned_keys, digit_extractor, prefix_mask, kth_key_bits);
      __syncthreads();

      compute_bin_offsets();
      __syncthreads();

      if (linear_tid == 0)
      {
        storage.pass_state.k      = 0;
        storage.pass_state.len    = 0;
        storage.pass_state.bucket = 0;
      }
      __syncthreads();

      choose_bucket(current_k);
      __syncthreads();

      current_k   = storage.pass_state.k;
      current_len = storage.pass_state.len;
      kth_key_bits |= bit_ordered_type(storage.pass_state.bucket) << pass_begin_bit;
      prefix_mask |= pass_mask;

      if (current_len == current_k)
      {
        break;
      }
    }

    const bit_ordered_type kth_prefix = kth_key_bits & prefix_mask;

    bool selected_flags[items_per_thread];
    bool candidate_flags[items_per_thread];
    histo_counter_t selected_count_thread  = 0;
    histo_counter_t candidate_count_thread = 0;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      const bit_ordered_type key_prefix = unsigned_keys[i] & prefix_mask;
      const bool is_selected            = key_prefix < kth_prefix;
      const bool is_candidate           = key_prefix == kth_prefix;
      selected_flags[i]                 = is_selected;
      candidate_flags[i]                = is_candidate;
      selected_count_thread += is_selected;
      candidate_count_thread += is_candidate;
    }

    storage.selected_prefix[linear_tid]  = selected_count_thread;
    storage.candidate_prefix[linear_tid] = candidate_count_thread;
    __syncthreads();

    // TODO (elstehle): Replace with block scan, once issue has been identified.
    if (linear_tid == 0)
    {
      histo_counter_t sum = 0;
      for (int t = 0; t < block_threads; ++t)
      {
        const histo_counter_t count = storage.selected_prefix[t];
        storage.selected_prefix[t]  = sum;
        sum += count;
      }
      storage.pass_state.k = sum;

      sum = 0;
      for (int t = 0; t < block_threads; ++t)
      {
        const histo_counter_t count = storage.candidate_prefix[t];
        storage.candidate_prefix[t] = sum;
        sum += count;
      }
      storage.pass_state.len = sum;

      const histo_counter_t k_count   = static_cast<histo_counter_t>(k);
      histo_counter_t candidate_limit = 0;
      if (storage.pass_state.k < k_count)
      {
        const histo_counter_t remaining = k_count - storage.pass_state.k;
        candidate_limit                 = (remaining < storage.pass_state.len) ? remaining : storage.pass_state.len;
      }
      storage.pass_state.bucket = static_cast<int>(candidate_limit);
    }
    __syncthreads();

    const histo_counter_t selected_prefix  = storage.selected_prefix[linear_tid];
    const histo_counter_t candidate_prefix = storage.candidate_prefix[linear_tid];
    const histo_counter_t selected_total   = storage.pass_state.k;
    const histo_counter_t candidate_limit  = static_cast<histo_counter_t>(storage.pass_state.bucket);

    const histo_counter_t kept_total = selected_total + candidate_limit;

    histo_counter_t candidate_seen         = 0;
    histo_counter_t candidate_taken_thread = 0;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      if (candidate_flags[i])
      {
        const histo_counter_t candidate_global = candidate_prefix + candidate_seen;
        const bool take                        = candidate_global < candidate_limit;
        candidate_taken_thread += take;
        candidate_seen++;
      }
    }

    const histo_counter_t kept_count_thread     = selected_count_thread + candidate_taken_thread;
    const histo_counter_t rejected_count_thread = static_cast<histo_counter_t>(items_per_thread) - kept_count_thread;

    storage.selected_prefix[linear_tid] = rejected_count_thread;
    __syncthreads();

    // TODO (elstehle): Replace with block scan, once issue has been identified.
    if (linear_tid == 0)
    {
      histo_counter_t sum = 0;
      for (int t = 0; t < block_threads; ++t)
      {
        const histo_counter_t count = storage.selected_prefix[t];
        storage.selected_prefix[t]  = sum;
        sum += count;
      }
    }
    __syncthreads();
    const histo_counter_t rejected_prefix = storage.selected_prefix[linear_tid];

    int ranks[items_per_thread];
    histo_counter_t selected_local = 0;
    histo_counter_t rejected_local = 0;
    candidate_seen                 = 0;

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < items_per_thread; ++i)
    {
      if (selected_flags[i])
      {
        ranks[i] = static_cast<int>(selected_prefix + selected_local);
        selected_local++;
      }
      else if (candidate_flags[i])
      {
        const histo_counter_t candidate_global = candidate_prefix + candidate_seen;
        if (candidate_global < candidate_limit)
        {
          ranks[i] = static_cast<int>(selected_total + candidate_global);
        }
        else
        {
          ranks[i] = static_cast<int>(kept_total + rejected_prefix + rejected_local);
          rejected_local++;
        }
        candidate_seen++;
      }
      else
      {
        ranks[i] = static_cast<int>(kept_total + rejected_prefix + rejected_local);
        rejected_local++;
      }
    }

    __syncthreads();
    block_exchange_keys_t(storage.exchange_keys).ScatterToBlocked(keys, ranks);

    if constexpr (HasValues)
    {
      __syncthreads();
      block_exchange_values_t(storage.exchange_values).ScatterToBlocked(values, ranks);
    }
  }

public:
  struct TempStorage : Uninitialized<TempStorage_>
  {};

  _CCCL_DEVICE _CCCL_FORCEINLINE non_deterministic_air_topk(TempStorage& storage)
      : storage(storage.Alias())
      , linear_tid(RowMajorTid(BlockThreads, 1, 1))
  {}

  template <detail::topk::select SelectDirection>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  select_keys(KeyT (&keys)[items_per_thread], int k, int begin_bit = 0, int end_bit = sizeof(KeyT) * 8)
  {
    NullType values[ItemsPerThread];
    select_impl<SelectDirection, false>(keys, values, k, begin_bit, end_bit);
  }

  template <detail::topk::select SelectDirection>
  _CCCL_DEVICE _CCCL_FORCEINLINE void select_pairs(
    KeyT (&keys)[items_per_thread],
    ValueT (&values)[items_per_thread],
    int k,
    int begin_bit = 0,
    int end_bit   = sizeof(KeyT) * 8)
  {
    select_impl<SelectDirection, true>(keys, values, k, begin_bit, end_bit);
  }
};

// TODO (elstehle): Add documentation
template <typename KeyT, int BlockDimX, int ItemsPerThread, typename ValueT = NullType>
class block_topk
{
private:
  using internal_block_topk_t = non_deterministic_air_topk<KeyT, BlockDimX, ItemsPerThread, ValueT>;

public:
  struct TempStorage
  {
    typename internal_block_topk_t::TempStorage topk_storage;
  };

private:
  TempStorage& storage;

public:
  _CCCL_DEVICE _CCCL_FORCEINLINE block_topk(TempStorage& storage)
      : storage(storage)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void max_pairs(
    KeyT (&keys)[ItemsPerThread],
    ValueT (&values)[ItemsPerThread],
    int k,
    int begin_bit = 0,
    int end_bit   = sizeof(KeyT) * 8)
  {
    internal_block_topk_t(storage.topk_storage)
      .template select_pairs<detail::topk::select::max>(keys, values, k, begin_bit, end_bit);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void
  max_keys(KeyT (&keys)[ItemsPerThread], int k, int begin_bit = 0, int end_bit = sizeof(KeyT) * 8)
  {
    internal_block_topk_t(storage.topk_storage)
      .template select_keys<detail::topk::select::max>(keys, k, begin_bit, end_bit);
  }
  _CCCL_DEVICE _CCCL_FORCEINLINE void min_pairs(
    KeyT (&keys)[ItemsPerThread],
    ValueT (&values)[ItemsPerThread],
    int k,
    int begin_bit = 0,
    int end_bit   = sizeof(KeyT) * 8)
  {
    internal_block_topk_t(storage.topk_storage)
      .template select_pairs<detail::topk::select::min>(keys, values, k, begin_bit, end_bit);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void
  min_keys(KeyT (&keys)[ItemsPerThread], int k, int begin_bit = 0, int end_bit = sizeof(KeyT) * 8)
  {
    internal_block_topk_t(storage.topk_storage)
      .template select_keys<detail::topk::select::min>(keys, k, begin_bit, end_bit);
  }
};
} // namespace detail

CUB_NAMESPACE_END
