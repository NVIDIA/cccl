// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Implement kernel for DeviceSegmentedScan with block-wide workers processing
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

#include <cub/block/block_load.cuh>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/detail/segmented_scan_helpers.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_scan.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/util_macro.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/iterator>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/span>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
//! @brief agent_segmented_scan implements CTAs independently processing one or more segments
//!        of a device-wide segmented prefix scan.
//!
//! @tparam SegmentedScanPolicyGetterT
//!   Nullary callable type for getting segmented_scan_policy
//!
//! @tparam InputIteratorT
//!   Random-access input iterator type
//!
//! @tparam OutputIteratorT
//!   Random-access output iterator type
//!
//! @tparam OffsetT
//!   Integer type for global offsets
//!
//! @tparam ScanOpT
//!   Scan functor type
//!
//! @tparam InitValueT
//!   The init_value element for ScanOpT type (cub::NullType for inclusive scan)
//!
//! @tparam AccumT
//!   The type of intermediate accumulator (according to P2322R6)
//!
template <typename SegmentedScanPolicyGetterT,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ScanOpT,
          typename InitValueT,
          typename AccumT,
          bool ForceInclusive = false>
struct agent_segmented_scan
{
private:
  //---------------------------------------------------------------------
  // Types and constants
  //---------------------------------------------------------------------

  using input_t = cub::detail::it_value_t<InputIteratorT>;

  static constexpr auto agent_policy = SegmentedScanPolicyGetterT{}().block;

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
  static constexpr bool is_inclusive    = ForceInclusive || !has_init;
  static constexpr int block_threads    = agent_policy.block_threads;
  static constexpr int items_per_thread = agent_policy.items_per_thread;
  static constexpr int tile_items       = block_threads * items_per_thread;
  static constexpr int max_segments     = agent_policy.max_segments;

  static constexpr bool multi_segment_enabled = (max_segments > 1);

  static constexpr auto load_algorithm  = agent_policy.load_algorithm;
  static constexpr auto store_algorithm = agent_policy.store_algorithm;
  static constexpr auto scan_algorithm  = agent_policy.scan_algorithm;

  using block_load_t  = BlockLoad<AccumT, block_threads, items_per_thread, load_algorithm>;
  using block_store_t = BlockStore<AccumT, block_threads, items_per_thread, store_algorithm>;
  using block_scan_t  = BlockScan<AccumT, block_threads, scan_algorithm>;

  union _single_segment_algorithms_storage_t
  {
    typename block_load_t::TempStorage load;
    typename block_store_t::TempStorage store;
    typename block_scan_t::TempStorage scan;
  };

  struct _single_segment_temp_storage_t
  {
    _single_segment_algorithms_storage_t reused;
  };

  using augmented_accum_t = agent_segmented_scan_compute_t<AccumT, max_segments>;

  using block_load_aug_t    = BlockLoad<augmented_accum_t, block_threads, items_per_thread, load_algorithm>;
  using block_store_aug_t   = BlockStore<augmented_accum_t, block_threads, items_per_thread, store_algorithm>;
  using block_scan_aug_t    = BlockScan<augmented_accum_t, block_threads, scan_algorithm>;
  using block_offset_scan_t = BlockScan<OffsetT, block_threads, scan_algorithm>;
  using block_reduce_t      = BlockReduce<unsigned int, block_threads>;

  union _multiple_segment_algorithms_storage_t
  {
    typename block_load_t::TempStorage load;
    typename block_store_t::TempStorage store;
    typename block_scan_t::TempStorage scan;
    typename block_load_aug_t::TempStorage load_aug;
    typename block_store_aug_t::TempStorage store_aug;
    typename block_scan_aug_t::TempStorage scan_aug;
    typename block_offset_scan_t::TempStorage offset_scan;
    typename block_reduce_t::TempStorage min_reduce;
  };

  struct _multi_segment_temp_storage_t
  {
    OffsetT logical_segment_offsets[max_segments];
    unsigned int fixed_size_mask;
    _multiple_segment_algorithms_storage_t reused;
  };

  using _TempStorage =
    ::cuda::std::conditional_t<multi_segment_enabled, _multi_segment_temp_storage_t, _single_segment_temp_storage_t>;

  _TempStorage& temp_storage; ///< Reference to temp_storage
  wrapped_input_iterator_t d_in; ///< Input data
  OutputIteratorT d_out; ///< Output data
  ScanOpT scan_op; ///< Binary associative scan operator
  InitValueT initial_value; ///< The initial value element for ScanOpT

public:
  // Alias wrapper allowing storage to be unioned
  using TempStorage = Uninitialized<_TempStorage>;

  //---------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------

  _CCCL_DEVICE _CCCL_FORCEINLINE agent_segmented_scan(
    TempStorage& temp_storage, InputIteratorT d_in, OutputIteratorT d_out, ScanOpT scan_op, InitValueT initial_value)
      : temp_storage(temp_storage.Alias())
      , d_in(d_in)
      , d_out(d_out)
      , scan_op(scan_op)
      , initial_value(initial_value)
  {}

  //! @brief Scan one segment of values
  //!
  //! @param input_begin_idx
  //!   Index of start of the segment in input array
  //!
  //! @param input_end_idx
  //!  Index of end of the segment in input array
  //!
  //! @param output_begin_idx
  //!  Index of start of the segment's prefix scan result in the output array
  //!
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
      {
        block_load_t loader(temp_storage.reused.load);
        if (chunk_size == tile_items)
        {
          loader.Load(d_in + chunk_begin, thread_values);
        }
        else
        {
          loader.Load(d_in + chunk_begin, thread_values, chunk_size, AccumT{});
        }
      }
      __syncthreads();

      {
        block_scan_t scanner(temp_storage.reused.scan);
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
      __syncthreads();

      {
        block_store_t storer(temp_storage.reused.store);
        if (chunk_size == tile_items)
        {
          storer.Store(d_out + output_begin_idx + chunk_id * tile_items, thread_values);
        }
        else
        {
          storer.Store(d_out + output_begin_idx + chunk_id * tile_items, thread_values, chunk_size);
        }
      }
      if (++chunk_id < n_chunks)
      {
        __syncthreads();
      }
    }
  };

  //! @brief Scan dynamically specified number of segments of values
  template <typename InputBeginOffsetIteratorT,
            typename InputEndOffsetIteratorT,
            typename OutputBeginOffsetIteratorT,
            ::cuda::std::size_t NumSegments                  = max_segments,
            ::cuda::std::enable_if_t<(NumSegments > 1), int> = 0>
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

    static_assert(NumSegments <= max_segments,
                  "Template value NumSegments of consume_ranges method must not exceed class template value "
                  "controlling size of shared memory array");

    _CCCL_ASSERT(n_segments > 0, "Number of segments per worker should be positive");
    _CCCL_ASSERT(n_segments <= NumSegments, "Number of segments per worker exceeds statically provisioned storage");

    // cooperatively compute inclusive scan of sizes of segments to be processed by this block
    {
      const auto tid = threadIdx.x;

      // assume all segments have the same size
      if (tid == 0)
      {
        temp_storage.fixed_size_mask = 1u;
      }

      n_segments               = ::cuda::std::min(n_segments, static_cast<int>(NumSegments));
      unsigned n_chunks        = ::cuda::ceil_div(n_segments, block_threads);
      OffsetT exclusive_prefix = 0;
      using plus_t             = ::cuda::std::plus<>;
      const plus_t offsets_scan_op{};
      worker_prefix_callback_t prefix_callback_op{exclusive_prefix, offsets_scan_op};

      for (unsigned chunk_id = 0; chunk_id < n_chunks; ++chunk_id)
      {
        const unsigned work_id = chunk_id * block_threads + tid;

        // TODO: use BlockLoad to load
        const OffsetT input_segment_begin = (work_id < n_segments) ? input_begin_idx_it[work_id] : 0;
        const OffsetT input_segment_end   = (work_id < n_segments) ? input_end_idx_it[work_id] : 0;
        const OffsetT segment_size = ::cuda::std::max(input_segment_end, input_segment_begin) - input_segment_begin;

        block_offset_scan_t offset_scanner(temp_storage.reused.offset_scan);

        OffsetT prefix;
        offset_scanner.InclusiveSum(segment_size, prefix, prefix_callback_op);

        if (work_id < n_segments)
        {
          temp_storage.logical_segment_offsets[work_id] = prefix;
        }
        __syncthreads();

        block_reduce_t min_reducer(temp_storage.reused.min_reduce);
        ::cuda::minimum<unsigned int> min_op;

        const unsigned int fixed_size_check =
          ((work_id >= n_segments) || (prefix == (work_id + 1) * temp_storage.logical_segment_offsets[0])) ? 1u : 0u;
        const unsigned int block_fixed_size_check = min_reducer.Reduce(fixed_size_check, min_op);
        if (tid == 0)
        {
          temp_storage.fixed_size_mask = min_op(temp_storage.fixed_size_mask, block_fixed_size_check);
        }
        __syncthreads();
      }
    }

    const ::cuda::std::span<OffsetT> cum_sizes{
      temp_storage.logical_segment_offsets, static_cast<::cuda::std::size_t>(n_segments)};

    const OffsetT items_per_block = cum_sizes[n_segments - 1];

    if (temp_storage.fixed_size_mask && (items_per_block > 0))
    {
      // fast path: assumes all segments have identical size (checked via fixed_size_mask)
      // fixed-size searcher can cheaply identify which segment an element belongs to
      // using segment_id = elem_id / segment_size;
      const auto segment_size = cum_sizes[0];
      _CCCL_ASSERT((segment_size > 0) && ((items_per_block % segment_size) == 0),
                   "Precondition violated, likely due to a race condition");

      const bag_of_fixed_size_segments searcher{segment_size};
      scan_segments_chunked(searcher, input_begin_idx_it, output_begin_idx_it, items_per_block);
    }
    else
    {
      // Branchless search is faster
      constexpr bool use_branchless = true;

      if constexpr (use_branchless)
      {
        /*
## varying_size_segments

### [0] NVIDIA RTX A6000

| T{ct} | OffsetT{ct} | Elements{io} | #Seg{io} | $Seg/Worker{io} | Worker{io} | Samples | BWUtil |
|-------|-------------|--------------|----------|-----------------|------------|---------|--------|
|   I32 |         I32 |    2^27      |       57 |             216 |      block |    672x | 51.97% |
|   I32 |         I32 |    2^27      |       57 |              18 |       warp |   2832x | 86.45% |
|   I64 |         I32 |    2^27      |       57 |             216 |      block |    720x | 76.07% |
|   I64 |         I32 |    2^27      |       57 |              18 |       warp |    784x | 86.95% |
         */
        // searcher locates segment_id using branchless linear/binary search in cum_sizes
        const auto searcher = make_statically_bound_bag_of_segments<NumSegments>(cum_sizes);
        scan_segments_chunked(searcher, input_begin_idx_it, output_begin_idx_it, items_per_block);
      }
      else
      {
        /*
## varying_size_segments

### [0] NVIDIA RTX A6000

| T{ct} | OffsetT{ct} | Elements{io} | #Seg{io} | #Seg/Worker{io} | Worker{io} | Samples | BWUtil |
|-------|-------------|--------------|----------|-----------------|------------|---------|--------|
|   I32 |         I32 |     2^27     |       57 |             117 |      block |    944x | 39.85% |
|   I32 |         I32 |     2^27     |       57 |              18 |       warp |   2821x | 86.41% |
|   I64 |         I32 |     2^27     |       57 |             117 |      block |    544x | 53.11% |
|   I64 |         I32 |     2^27     |       57 |              18 |       warp |    752x | 86.90% |
         */

        // searcher locates segment_id using linear/binary search in cum_sizes
        // binary search is using while-loop based cub::std::upper_bound
        bag_of_segments searcher{cum_sizes};
        scan_segments_chunked(searcher, input_begin_idx_it, output_begin_idx_it, items_per_block);
      }
    }
  }

private:
  template <typename SearcherT, typename InputBeginOffsetIteratorT, typename OutputBeginOffsetIteratorT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void scan_segments_chunked(
    const SearcherT& searcher,
    InputBeginOffsetIteratorT input_begin_idx_it,
    OutputBeginOffsetIteratorT output_begin_idx_it,
    OffsetT items_per_block)
  {
    const OffsetT n_chunks = ::cuda::ceil_div(items_per_block, tile_items);

    using augmented_scan_op_t = schwarz_scan_op<ScanOpT, AccumT>;

    augmented_scan_op_t augmented_scan_op{scan_op};

    augmented_accum_t exclusive_prefix{};
    worker_prefix_callback_t prefix_op{exclusive_prefix, augmented_scan_op};

    augmented_accum_t thread_flag_values[items_per_thread];
    for (OffsetT chunk_id = 0; chunk_id < n_chunks;)
    {
      const OffsetT chunk_begin = chunk_id * tile_items;
      const OffsetT chunk_end   = (::cuda::std::min) (chunk_begin + tile_items, items_per_block);

      // chunk_size <= TILE_ITEMS, casting to int is safe
      const int chunk_size = static_cast<int>(chunk_end - chunk_begin);

      // load values, and pack them into head_flag-value pairs
      {
        constexpr auto oob_default = augmented_value_t{AccumT{}, false};

        block_load_aug_t loader(temp_storage.reused.load_aug);
        if constexpr (has_init)
        {
          const packer_iv<ScanOpT, AccumT> packer_op{scan_op, initial_value};
          multi_segmented_input_iterator it_in{d_in, chunk_begin, searcher, input_begin_idx_it, packer_op};

          if (chunk_size == tile_items)
          {
            loader.Load(it_in, thread_flag_values);
          }
          else
          {
            loader.Load(it_in, thread_flag_values, chunk_size, oob_default);
          }
        }
        else
        {
          constexpr packer<AccumT> packer_op{};
          multi_segmented_input_iterator it_in{d_in, chunk_begin, searcher, input_begin_idx_it, packer_op};

          if (chunk_size == tile_items)
          {
            loader.Load(it_in, thread_flag_values);
          }
          else
          {
            loader.Load(it_in, thread_flag_values, chunk_size, oob_default);
          }
        }
      }
      __syncthreads();

      {
        block_scan_aug_t scanner(temp_storage.reused.scan_aug);
        if (chunk_id == 0)
        {
          const auto augmented_init_value = [&]() {
            if constexpr (has_init)
            {
              return augmented_value_t{initial_value, false};
            }
            else
            {
              return augmented_value_t{AccumT{}, false};
            }
          }();
          // Initialize exclusive_prefix, referenced from prefix_op
          scan_first_tile(scanner, thread_flag_values, augmented_init_value, augmented_scan_op, exclusive_prefix);
        }
        else
        {
          scan_later_tile(scanner, thread_flag_values, augmented_scan_op, prefix_op);
        }
      }
      __syncthreads();

      // store prefix-scan values, discarding head flags
      {
        const OffsetT out_offset = chunk_id * tile_items;

        block_store_aug_t storer(temp_storage.reused.store_aug);
        if constexpr (is_inclusive)
        {
          constexpr projector<AccumT> projector_op{};
          multi_segmented_output_iterator it_out{d_out, out_offset, searcher, output_begin_idx_it, projector_op};

          if (chunk_size == tile_items)
          {
            storer.Store(it_out, thread_flag_values);
          }
          else
          {
            storer.Store(it_out, thread_flag_values, chunk_size);
          }
        }
        else
        {
          const projector_iv<AccumT> projector_op{initial_value};
          multi_segmented_output_iterator it_out{d_out, out_offset, searcher, output_begin_idx_it, projector_op};
          if (chunk_size == tile_items)
          {
            storer.Store(it_out, thread_flag_values);
          }
          else
          {
            storer.Store(it_out, thread_flag_values, chunk_size);
          }
        }
      }
      // Avoiding synchronization at the end of last chunk
      // could save up to 10% of performance for very short segments
      if (++chunk_id < n_chunks)
      {
        __syncthreads();
      }
    }
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
    ItemTy& block_aggregate)
  {
    if constexpr (HasInit)
    {
      const ItemTy converted_init_value = convert_initial_value<ItemTy>(init_value);
      if constexpr (IsInclusive)
      {
        scanner.InclusiveScan(items, items, converted_init_value, scan_op, block_aggregate);
      }
      else
      {
        scanner.ExclusiveScan(items, items, converted_init_value, scan_op, block_aggregate);
      }
      block_aggregate = scan_op(converted_init_value, block_aggregate);
    }
    else
    {
      static_assert(IsInclusive, "Unexpected ExclusiveScan without initial value call");
      scanner.InclusiveScan(items, items, scan_op, block_aggregate);
    }
  }

  template <typename ScannerT, typename ItemTy, typename ScanOpTy, typename PrefixCallback, bool IsInclusive = is_inclusive>
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  scan_later_tile(ScannerT& scanner, ItemTy (&items)[items_per_thread], ScanOpTy scan_op, PrefixCallback& prefix_op)
  {
    if constexpr (IsInclusive)
    {
      scanner.InclusiveScan(items, items, scan_op, prefix_op);
    }
    else
    {
      scanner.ExclusiveScan(items, items, scan_op, prefix_op);
    }
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
__launch_bounds__(current_policy<PolicySelector>().block.block_threads)
  _CCCL_KERNEL_ATTRIBUTES void device_segmented_scan_kernel(
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
  static_assert(policy.block.load_modifier != CacheLoadModifier::LOAD_LDG,
                "The memory consistency model does not apply to texture accesses");

  struct policy_getter
  {
    constexpr auto operator()() const
    {
      return policy;
    }
  };

  using agent_t =
    agent_segmented_scan<policy_getter,
                         InputIteratorT,
                         OutputIteratorT,
                         OffsetT,
                         ScanOpT,
                         ActualInitValueT,
                         AccumT,
                         ForceInclusive>;

  __shared__ typename agent_t::TempStorage temp_storage;

  const ActualInitValueT _init_value = init_value;

  _CCCL_ASSERT(num_segments_per_worker > 0, "Number of segments to be processed by block must be positive");
  _CCCL_ASSERT(num_segments_per_worker <= policy.block.max_segments,
               "Requested number of segments to be processed by block exceeds compile-time maximum");

  const auto work_id = num_segments_per_worker * blockIdx.x;

  agent_t agent(temp_storage, d_in, d_out, scan_op, _init_value);

  if constexpr (policy.block.max_segments == 1)
  {
    _CCCL_ASSERT(num_segments_per_worker == 1, "Inconsistent parameters in device_segmented_scan_kernel");
    _CCCL_ASSERT(work_id < n_segments, "device_segmented_scan_kernel launch configuration results in access violation");

    const OffsetT input_begin_idx  = begin_offset_d_in[work_id];
    const OffsetT input_end_idx    = end_offset_d_in[work_id];
    const OffsetT output_begin_idx = begin_offset_d_out[work_id];

    agent.scan_one_segment(input_begin_idx, input_end_idx, output_begin_idx);
  }
  else
  {
    if (work_id >= n_segments)
    {
      return;
    }

    const auto start_offset         = work_id;
    const auto suggested_end_offset = start_offset + num_segments_per_worker;

    using IdT             = decltype(work_id);
    const auto end_offset = ::cuda::std::min<IdT>(suggested_end_offset, n_segments);
    int size              = end_offset - start_offset;

    auto worker_input_begin_idx_it  = begin_offset_d_in + start_offset;
    auto worker_input_end_idx_it    = end_offset_d_in + start_offset;
    auto worker_output_begin_idx_it = begin_offset_d_out + start_offset;

    if (size == 1)
    {
      agent.scan_one_segment(worker_input_begin_idx_it[0], worker_input_end_idx_it[0], worker_output_begin_idx_it[0]);
    }
    else
    {
      agent.scan_segments(worker_input_begin_idx_it, worker_input_end_idx_it, worker_output_begin_idx_it, size);
    }
  }
}
} // namespace detail::segmented_scan

CUB_NAMESPACE_END
