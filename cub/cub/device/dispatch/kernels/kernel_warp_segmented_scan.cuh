// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Implement kernel for DeviceSegmentedScan with warp-wide workers processing
//! individual segments.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/segmented_scan_helpers.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_scan.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/thread/thread_reduce.cuh> // ThreadReduce
#include <cub/thread/thread_scan.cuh> // detail::ThreadInclusiveScan
#include <cub/util_macro.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/warp_load.cuh>
#include <cub/warp/warp_reduce.cuh>
#include <cub/warp/warp_scan.cuh>
#include <cub/warp/warp_store.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/iterator>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/span>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
template <typename SegmentedScanPolicyGetterT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ScanOpT,
          typename InitValueT,
          typename AccumT,
          bool ForceInclusive = false>
struct agent_warp_segmented_scan
{
private:
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  using input_t = cub::detail::it_value_t<InputIteratorT>;

  static constexpr auto agent_policy = SegmentedScanPolicyGetterT{}().warp;

  // Input iterator wrapper type (for applying cache modifier)
  // Wrap the native input pointer with CacheModifiedInputIterator
  // or directly use the supplied input iterator type
  using wrapped_input_iterator_t =
    ::cuda::std::conditional_t<::cuda::std::is_pointer_v<InputIteratorT>,
                               CacheModifiedInputIterator<agent_policy.load_modifier, input_t, OffsetT>,
                               InputIteratorT>;

  // Using cub::NullType means no initial value is provided
  static constexpr bool has_init = !::cuda::std::is_same_v<InitValueT, NullType>;
  // We are relying on either initial value being `NullType`
  // or the ForceInclusive tag to be true for inclusive scan
  // to get picked up.
  static constexpr bool is_inclusive     = ForceInclusive || !has_init;
  static constexpr int threads_per_block = agent_policy.threads_per_block;
  static constexpr int items_per_thread  = agent_policy.items_per_thread;
  static constexpr int tile_items        = warp_threads * items_per_thread;
  static constexpr int max_segments      = agent_policy.max_segments;

  static_assert(0 == threads_per_block % warp_threads, "Block size must be a multiple of native warp size");

  static constexpr auto warps_in_block = threads_per_block / warp_threads;

  static_assert(warps_in_block > 0, "Number of warps must be positive");

  static constexpr bool multi_segment_enabled = (max_segments > 1);

  static constexpr auto load_algorithm  = agent_policy.load_algorithm;
  static constexpr auto store_algorithm = agent_policy.store_algorithm;

  using warp_load_t  = WarpLoad<AccumT, items_per_thread, load_algorithm>;
  using warp_store_t = WarpStore<AccumT, items_per_thread, store_algorithm>;
  using warp_scan_t  = WarpScan<AccumT>;

  union _single_segment_algorithms_storage_t
  {
    typename warp_load_t::TempStorage load[warps_in_block];
    typename warp_store_t::TempStorage store[warps_in_block];
    typename warp_scan_t::TempStorage scan[warps_in_block];
  };

  struct _single_segment_temp_storage_t
  {
    _single_segment_algorithms_storage_t reused;
  };

  using augmented_accum_t = agent_segmented_scan_compute_t<AccumT, max_segments>;

  using warp_load_aug_t  = WarpLoad<augmented_accum_t, items_per_thread, load_algorithm>;
  using warp_store_aug_t = WarpStore<augmented_accum_t, items_per_thread, store_algorithm>;
  using warp_scan_aug_t  = WarpScan<augmented_accum_t>;

  using warp_scan_offsets_t = WarpScan<OffsetT>;
  using warp_reduce_t       = WarpReduce<unsigned int>;

  union _multiple_segment_algorithms_storage_t
  {
    // storage for single segment per warp method scan_one_segment
    typename warp_load_t::TempStorage load[warps_in_block];
    typename warp_store_t::TempStorage store[warps_in_block];
    typename warp_scan_t::TempStorage scan[warps_in_block];
    // storage for multiple segments per warp method scan_segments
    typename warp_load_aug_t::TempStorage load_aug[warps_in_block];
    typename warp_store_aug_t::TempStorage store_aug[warps_in_block];
    typename warp_scan_aug_t::TempStorage scan_aug[warps_in_block];
    typename warp_scan_offsets_t::TempStorage offsets_scan[warps_in_block];
    typename warp_reduce_t::TempStorage min_reduce[warps_in_block];
  };

  struct _multi_segment_temp_storage_t
  {
    OffsetT logical_segment_offsets[warps_in_block][max_segments];
    unsigned int fixed_size_mask[warps_in_block];
    _multiple_segment_algorithms_storage_t reused;
  };

  using _TempStorage =
    ::cuda::std::conditional_t<multi_segment_enabled, _multi_segment_temp_storage_t, _single_segment_temp_storage_t>;

  static_assert(sizeof(_TempStorage) <= cub::detail::max_smem_per_block);

  _TempStorage& temp_storage; ///< Reference to temp_storage
  wrapped_input_iterator_t d_in; ///< Input data
  OutputIteratorT d_out; ///< Output data
  ScanOpT scan_op; ///< Binary associative scan operator
  InitValueT initial_value; ///< The initial value element for ScanOpT
  unsigned int warp_id; ///< Warp identifier within CTA
  unsigned int lane_id; ///< Thread identified within warp

private:
  struct segment_size_preprocessing_scope
  {
    static constexpr unsigned worker_thread_count = warp_threads;

    agent_warp_segmented_scan& agent;

    _CCCL_DEVICE _CCCL_FORCEINLINE unsigned worker_id() const
    {
      return agent.lane_id;
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE void init_fixed_size_mask()
    {
      if (agent.lane_id == 0)
      {
        agent.temp_storage.fixed_size_mask[agent.warp_id] = 1u;
      }
    }

    template <typename PrefixCallbackT>
    _CCCL_DEVICE _CCCL_FORCEINLINE OffsetT
    inclusive_scan_segment_size(OffsetT segment_size, PrefixCallbackT& prefix_callback_op)
    {
      warp_scan_offsets_t offset_scanner(agent.temp_storage.reused.offsets_scan[agent.warp_id]);

      OffsetT prefix;
      OffsetT warp_aggregate;
      offset_scanner.InclusiveSum(segment_size, prefix, warp_aggregate);
      const OffsetT warp_prefix = prefix_callback_op(warp_aggregate);

      return warp_prefix + prefix;
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE void store_logical_segment_offset(unsigned work_id, OffsetT prefix)
    {
      _CCCL_ASSERT(work_id < max_segments, "Access violation in work_id index");
      _CCCL_ASSERT(agent.warp_id < warps_in_block, "Access violation in warp_id index");
      agent.temp_storage.logical_segment_offsets[agent.warp_id][work_id] = prefix;
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE OffsetT first_logical_segment_offset() const
    {
      return agent.temp_storage.logical_segment_offsets[agent.warp_id][0];
    }

    template <typename OpT>
    _CCCL_DEVICE _CCCL_FORCEINLINE unsigned int reduce_fixed_size_check(unsigned int fixed_size_check, OpT min_op)
    {
      warp_reduce_t min_reducer(agent.temp_storage.reused.min_reduce[agent.warp_id]);
      return min_reducer.Reduce(fixed_size_check, min_op);
    }

    template <typename OpT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void update_fixed_size_mask(unsigned int worker_fixed_size_check, OpT min_op)
    {
      if (agent.lane_id == 0)
      {
        agent.temp_storage.fixed_size_mask[agent.warp_id] =
          min_op(agent.temp_storage.fixed_size_mask[agent.warp_id], worker_fixed_size_check);
      }
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE void sync()
    {
      __syncwarp();
    }
  };

public:
  // Alias wrapper allowing storage to be unioned
  using TempStorage = Uninitialized<_TempStorage>;

  _CCCL_DEVICE _CCCL_FORCEINLINE agent_warp_segmented_scan(
    TempStorage& temp_storage, InputIteratorT d_in, OutputIteratorT d_out, ScanOpT scan_op, InitValueT initial_value)
      : temp_storage(temp_storage.Alias())
      , d_in(d_in)
      , d_out(d_out)
      , scan_op(scan_op)
      , initial_value(initial_value)
      , warp_id(threadIdx.x >> log2_warp_threads)
      , lane_id(threadIdx.x & (warp_threads - 1))
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void
  scan_one_segment(OffsetT input_begin_idx, OffsetT input_end_idx, OffsetT output_begin_idx)
  {
    const OffsetT segment_items = ::cuda::std::max(input_end_idx, input_begin_idx) - input_begin_idx;
    const OffsetT n_chunks      = ::cuda::ceil_div(segment_items, tile_items);

    AccumT exclusive_prefix{};
    worker_prefix_callback_t prefix_op{exclusive_prefix, scan_op};

    for (OffsetT chunk_id = 0; chunk_id < n_chunks;)
    {
      const OffsetT chunk_begin = input_begin_idx + chunk_id * tile_items;
      const OffsetT chunk_end   = (::cuda::std::min) (chunk_begin + tile_items, input_end_idx);

      // chunk_size <= TILE_ITEMS, casting to int is safe
      const int chunk_size = static_cast<int>(chunk_end - chunk_begin);

      AccumT thread_values[items_per_thread];

      // load
      {
        warp_load_t loader(temp_storage.reused.load[warp_id]);
        if (chunk_size == tile_items)
        {
          loader.Load(d_in + chunk_begin, thread_values);
        }
        else
        {
          constexpr AccumT oob_default{};
          loader.Load(d_in + chunk_begin, thread_values, chunk_size, oob_default);
        }
      }
      __syncwarp();

      // scan
      {
        warp_scan_t scanner(temp_storage.reused.scan[warp_id]);
        if (chunk_id == 0)
        {
          // Initialize exclusive_prefix, referenced from prefix_op
          scan_first_tile(scanner, thread_values, initial_value, scan_op, exclusive_prefix);
        }
        else
        {
          scan_later_tile(scanner, thread_values, scan_op, prefix_op);
        }
      }
      __syncwarp();

      // store
      {
        const OffsetT out_offset = output_begin_idx + chunk_id * tile_items;
        warp_store_t storer(temp_storage.reused.store[warp_id]);

        if (chunk_size == tile_items)
        {
          storer.Store(d_out + out_offset, thread_values);
        }
        else
        {
          storer.Store(d_out + out_offset, thread_values, chunk_size);
        }
      }
      // Avoiding synchronization at the end of last chunk
      // could save up to 10% of performance for very short segments
      if (++chunk_id < n_chunks)
      {
        __syncwarp();
      }
    }
  };

  //! @brief Scan dynamically given number of segments of values
  //! All arguments are warp-uniform
  template <typename InputBeginOffsetIteratorT,
            typename InputEndOffsetIteratorT,
            typename OutputBeginOffsetIteratorT,
            ::cuda::std::size_t MaxNumSegments                  = max_segments,
            ::cuda::std::enable_if_t<(MaxNumSegments > 1), int> = 0>
  _CCCL_DEVICE _CCCL_FORCEINLINE void scan_segments(
    InputBeginOffsetIteratorT input_begin_idx_it,
    InputEndOffsetIteratorT input_end_idx_it,
    OutputBeginOffsetIteratorT output_begin_idx_it,
    int n_segments)
  {
    static_assert(::cuda::std::is_convertible_v<::cuda::std::iter_reference_t<InputBeginOffsetIteratorT>, OffsetT>,
                  "Unexpected iterator type");
    static_assert(::cuda::std::is_convertible_v<::cuda::std::iter_reference_t<InputEndOffsetIteratorT>, OffsetT>,
                  "Unexpected iterator type");
    static_assert(::cuda::std::is_convertible_v<::cuda::std::iter_reference_t<OutputBeginOffsetIteratorT>, OffsetT>,
                  "Unexpected iterator type");

    static_assert(MaxNumSegments <= max_segments,
                  "Template parameter MaxNumSegments must not exceed size of shared array");

    _CCCL_ASSERT(n_segments > 0, "Number of segments should be greater than zero");
    _CCCL_ASSERT(n_segments <= max_segments, "Number of segments should not exceed statically provisioned storage");

    n_segments = preprocess_segment_sizes<MaxNumSegments, OffsetT>(
      segment_size_preprocessing_scope{*this}, input_begin_idx_it, input_end_idx_it, n_segments);

    // All accesses of logical_segment_offsets from now on are read-only. Elements of
    // logical_segment_offsets[warp_id] are only accessed by threads with the same warp_id.

    const auto cum_sizes_count = static_cast<::cuda::std::size_t>(n_segments);
    const ::cuda::std::span<OffsetT> cum_sizes{temp_storage.logical_segment_offsets[warp_id], cum_sizes_count};

    const OffsetT items_per_warp = cum_sizes[n_segments - 1];

    if (temp_storage.fixed_size_mask[warp_id] && items_per_warp > 0)
    {
      // fast path: assumes all segments have identical size (checked via fixed_size_mask)
      // fixed-size searcher can cheaply identify which segment an element belongs to
      // using segment_id = elem_id / segment_size;
      const auto segment_size = cum_sizes[0];
      _CCCL_ASSERT((segment_size > 0) && ((items_per_warp % segment_size) == 0),
                   "Precondition violated, likely due to a race condition");

      const bag_of_fixed_size_segments searcher{cum_sizes[0]};
      scan_segments_chunked(searcher, input_begin_idx_it, output_begin_idx_it, items_per_warp);
    }
    else
    {
      constexpr bool use_branchless = true;

      if constexpr (use_branchless)
      {
        // searcher locates segment_id using branchless linear/binary search in cum_sizes
        const auto searcher = make_statically_bound_bag_of_segments<MaxNumSegments>(cum_sizes);
        scan_segments_chunked(searcher, input_begin_idx_it, output_begin_idx_it, items_per_warp);
      }
      else
      {
        // searcher locates segment_id using linear/binary search in cum_sizes
        const bag_of_segments searcher{cum_sizes};
        scan_segments_chunked(searcher, input_begin_idx_it, output_begin_idx_it, items_per_warp);
      }
    }
  }

private:
  struct warp_chunked_scan_scope
  {
    agent_warp_segmented_scan& agent;

    template <typename IteratorT, typename ItemT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    load(IteratorT it, ItemT (&thread_values)[items_per_thread], int chunk_size, ItemT oob_default)
    {
      warp_load_aug_t loader(agent.temp_storage.reused.load_aug[agent.warp_id]);
      if (chunk_size == tile_items)
      {
        loader.Load(it, thread_values);
      }
      else
      {
        loader.Load(it, thread_values, chunk_size, oob_default);
      }
    }

    template <typename ItemT, typename OpT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    scan_first_tile(ItemT (&items)[items_per_thread], ItemT init_value, OpT scan_op, ItemT& warp_aggregate)
    {
      warp_scan_aug_t scanner(agent.temp_storage.reused.scan_aug[agent.warp_id]);
      agent.scan_first_tile(scanner, items, init_value, scan_op, warp_aggregate);
    }

    template <typename ItemT, typename OpT, typename PrefixCallbackT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void
    scan_later_tile(ItemT (&items)[items_per_thread], OpT scan_op, PrefixCallbackT& prefix_op)
    {
      warp_scan_aug_t scanner(agent.temp_storage.reused.scan_aug[agent.warp_id]);
      agent.scan_later_tile(scanner, items, scan_op, prefix_op);
    }

    template <typename IteratorT, typename ItemT>
    _CCCL_DEVICE _CCCL_FORCEINLINE void store(IteratorT it, ItemT (&thread_values)[items_per_thread], int chunk_size)
    {
      warp_store_aug_t storer(agent.temp_storage.reused.store_aug[agent.warp_id]);
      if (chunk_size == tile_items)
      {
        storer.Store(it, thread_values);
      }
      else
      {
        storer.Store(it, thread_values, chunk_size);
      }
    }

    _CCCL_DEVICE _CCCL_FORCEINLINE void sync()
    {
      __syncwarp();
    }
  };

  template <typename SearcherT, typename InputBeginOffsetIteratorT, typename OutputBeginOffsetIteratorT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void scan_segments_chunked(
    const SearcherT& searcher,
    InputBeginOffsetIteratorT input_begin_idx_it,
    OutputBeginOffsetIteratorT output_begin_idx_it,
    OffsetT items_per_warp)
  {
    multi_segment_scan_chunked<has_init, is_inclusive, items_per_thread, tile_items, OffsetT, AccumT>(
      warp_chunked_scan_scope{*this},
      d_in,
      d_out,
      scan_op,
      initial_value,
      searcher,
      input_begin_idx_it,
      output_begin_idx_it,
      items_per_warp);
  }

  template <typename ScannerT,
            typename ItemTy,
            typename InitValueTy,
            typename ScanOpTy,
            bool IsInclusive = is_inclusive,
            bool HasInit     = has_init>
  _CCCL_DEVICE _CCCL_FORCEINLINE void scan_first_tile(
    ScannerT& scanner,
    ItemTy (&items)[items_per_thread],
    InitValueTy init_value,
    ScanOpTy scan_op,
    ItemTy& warp_aggregate)
  {
    // TODO: specialize for items_per_thread == 1
    ItemTy thread_aggregate = cub::ThreadReduce(items, scan_op);
    if constexpr (HasInit)
    {
      const ItemTy converted_init_value = convert_initial_value<ItemTy>(init_value);
      scanner.ExclusiveScan(thread_aggregate, thread_aggregate, converted_init_value, scan_op, warp_aggregate);
      warp_aggregate = scan_op(converted_init_value, warp_aggregate);
    }
    else
    {
      static_assert(IsInclusive, "Unexpected ExclusiveScan without initial value call");
      scanner.ExclusiveScan(thread_aggregate, thread_aggregate, scan_op, warp_aggregate);
    }
    if constexpr (IsInclusive)
    {
      ThreadScanInclusive(items, items, scan_op, thread_aggregate, has_init || (lane_id != 0));
    }
    else
    {
      ThreadScanExclusive(items, items, scan_op, thread_aggregate, has_init || (lane_id != 0));
    }
  }

  template <typename ScannerT, typename ItemTy, typename ScanOpTy, typename PrefixCallbackT, bool IsInclusive = is_inclusive>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  scan_later_tile(ScannerT& scanner, ItemTy (&items)[items_per_thread], ScanOpTy scan_op, PrefixCallbackT& prefix_op)
  {
    // TODO: specialize for items_per_thread == 1
    const ItemTy init_value = prefix_op.current_prefix();
    ItemTy thread_aggregate = cub::ThreadReduce(items, scan_op);
    ItemTy warp_aggregate;

    scanner.ExclusiveScan(thread_aggregate, thread_aggregate, init_value, scan_op, warp_aggregate);

    if constexpr (IsInclusive)
    {
      detail::ThreadScanInclusive(items, items, scan_op, thread_aggregate);
    }
    else
    {
      detail::ThreadScanExclusive(items, items, scan_op, thread_aggregate);
    }
    prefix_op(warp_aggregate);
  }
};

template <typename PolicySelector,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorInputT,
          typename EndOffsetIteratorInputT,
          typename BeginOffsetIteratorOutputT,
          typename OffsetT,
          typename ScanOpT,
          typename InitValueT,
          typename AccumT,
          bool ForceInclusive,
          typename ActualInitValueT = typename InitValueT::value_type>
#if _CCCL_HAS_CONCEPTS()
  requires segmented_scan_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(current_policy<PolicySelector>().warp.threads_per_block)
  _CCCL_KERNEL_ATTRIBUTES void device_warp_segmented_scan_kernel(
    _CCCL_GRID_CONSTANT const InputIteratorT d_in,
    _CCCL_GRID_CONSTANT const OutputIteratorT d_out,
    _CCCL_GRID_CONSTANT const BeginOffsetIteratorInputT begin_offset_d_in,
    _CCCL_GRID_CONSTANT const EndOffsetIteratorInputT end_offset_d_in,
    _CCCL_GRID_CONSTANT const BeginOffsetIteratorOutputT begin_offset_d_out,
    _CCCL_GRID_CONSTANT const OffsetT n_segments,
    _CCCL_GRID_CONSTANT const ScanOpT scan_op,
    _CCCL_GRID_CONSTANT const InitValueT init_value,
    _CCCL_GRID_CONSTANT const int num_segments_per_worker)
{
  static constexpr auto policy = current_policy<PolicySelector>();
  static_assert(policy.warp.load_modifier != CacheLoadModifier::LOAD_LDG,
                "The memory consistency model does not apply to texture accesses");

  struct policy_getter
  {
    constexpr auto operator()() const
    {
      return policy;
    }
  };

  using agent_t = agent_warp_segmented_scan<
    policy_getter,
    InputIteratorT,
    OutputIteratorT,
    OffsetT,
    ScanOpT,
    ActualInitValueT,
    AccumT,
    ForceInclusive>;

  __shared__ typename agent_t::TempStorage temp_storage;

  const ActualInitValueT _init_value = init_value;

  _CCCL_ASSERT(num_segments_per_worker > 0, "Number of segments to be processed by warp must be positive");
  _CCCL_ASSERT(num_segments_per_worker <= policy.warp.max_segments,
               "Requested number of segments to be processed by warp exceeds compile-time maximum");

  static constexpr unsigned int warps_in_block = int(policy.warp.threads_per_block) >> log2_warp_threads;
  const unsigned int warp_id                   = threadIdx.x >> log2_warp_threads;

  const auto work_id = num_segments_per_worker * (blockIdx.x * warps_in_block) + warp_id;

  if (work_id >= n_segments)
  {
    return;
  }

  agent_t agent(temp_storage, d_in, d_out, scan_op, _init_value);

  if constexpr (policy.warp.max_segments == 1)
  {
    const OffsetT input_begin_idx  = begin_offset_d_in[work_id];
    const OffsetT input_end_idx    = end_offset_d_in[work_id];
    const OffsetT output_begin_idx = begin_offset_d_out[work_id];

    agent.scan_one_segment(input_begin_idx, input_end_idx, output_begin_idx);
  }
  else
  {
    // Agent consumes interleaved segments to improve CTA' memory access locality

    // agent accesses offset iterators with index: thread_work_id = chunk_id * worker_thread_count + lane_id;
    // for 0 <= chunk_id < ::cuda::ceil_div<unsigned>(n_segments, worker_thread_count)
    //
    //  total_offset = num_segments_per_worker * (blockIdx.x * warps_in_block) + warp_id +
    //      warps_in_block * thread_work_id;
    //
    using IdT                = decltype(work_id);
    const auto segment_count = static_cast<IdT>(n_segments);

    const int n_segments_per_warp =
      (work_id + (num_segments_per_worker - 1) * warps_in_block < segment_count)
        ? num_segments_per_worker
        : ::cuda::ceil_div(segment_count - work_id, warps_in_block);

    if (num_segments_per_worker == 1)
    {
      // The branch should be taken by all warps. Otherwise, since consume_range and consume_ranges methods
      // re-use shared memory using different logic, race condition arises for temporary storage in shared memory.

      if (n_segments_per_warp == 1)
      {
        // only those warps that do not read past end of segment iterators do the work
        agent.scan_one_segment(begin_offset_d_in[work_id], end_offset_d_in[work_id], begin_offset_d_out[work_id]);
      }
    }
    else
    {
      const ::cuda::strided_iterator raked_input_begin_idx_it{begin_offset_d_in + work_id, warps_in_block};
      const ::cuda::strided_iterator raked_input_end_idx_it{end_offset_d_in + work_id, warps_in_block};
      const ::cuda::strided_iterator raked_output_begin_idx_it{begin_offset_d_out + work_id, warps_in_block};
      agent.scan_segments(
        raked_input_begin_idx_it, raked_input_end_idx_it, raked_output_begin_idx_it, n_segments_per_warp);
    }
  }
}
} // namespace detail::segmented_scan

CUB_NAMESPACE_END
