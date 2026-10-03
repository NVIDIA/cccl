// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! Implement kernels for DeviceSegmentedScan that balance work across blocks by items
//! instead of by segments.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/block/block_exchange.cuh>
#include <cub/block/block_load.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_scan.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_macro.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__limits/numeric_limits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/make_unsigned.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
// Type of logical work positions. Output segments do not overlap, so the total work is at most the offset maximum
// plus one, which the unsigned type of a signed offset holds. An unsigned offset uses 64 bits.
template <typename OffsetT>
using load_balanced_work_t = ::cuda::std::
  conditional_t<::cuda::std::is_signed_v<OffsetT>, ::cuda::std::make_unsigned_t<OffsetT>, ::cuda::std::uint64_t>;

template <typename OffsetT>
struct boundary_record
{
  load_balanced_work_t<OffsetT> offset;
  OffsetT first_segment;
  bool is_head;
};

// Returns min(tiles, F x resident_blocks), where
// F = clamp(floor(tiles / (min_tiles_per_block x resident_blocks)), 1, max_factor).
// A zero min_tiles_per_block forces F = max_factor.
template <typename WorkT>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr int load_balanced_block_count(
  WorkT total_work, WorkT tile_size, int resident_blocks, int min_tiles_per_block, int max_factor)
{
  const auto tiles =
    static_cast<::cuda::std::int64_t>(total_work / tile_size + (total_work % tile_size != 0 ? WorkT{1} : WorkT{0}));
  ::cuda::std::int64_t factor = max_factor;
  if (min_tiles_per_block > 0)
  {
    factor = tiles / (static_cast<::cuda::std::int64_t>(min_tiles_per_block) * resident_blocks);
    factor = factor < 1 ? 1 : (factor > max_factor ? max_factor : factor);
  }
  return static_cast<int>((::cuda::std::min) (tiles, factor * resident_blocks));
}

template <typename WorkT>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr WorkT even_split_boundary(WorkT total_work, int partitions, int partition)
{
  const WorkT quotient              = total_work / static_cast<WorkT>(partitions);
  const WorkT remainder             = total_work % static_cast<WorkT>(partitions);
  const WorkT offset                = static_cast<WorkT>(partition);
  const auto remainder_contribution = (static_cast<::cuda::std::int64_t>(remainder) * partition) / partitions;
  return quotient * offset + static_cast<WorkT>(remainder_contribution);
}

template <typename OffsetIteratorT, typename OffsetT, typename ValueT>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE OffsetT
upper_bound_offset_in_range(OffsetIteratorT offsets, OffsetT first, OffsetT last, ValueT value)
{
  OffsetT count = last - first;
  while (count > 0)
  {
    const OffsetT step = count / 2;
    const OffsetT it   = first + step;
    if (offsets[it] <= value)
    {
      first = it + 1;
      count -= step + 1;
    }
    else
    {
      count = step;
    }
  }
  return first;
}

template <typename OffsetT>
_CCCL_DEVICE _CCCL_FORCEINLINE OffsetT warp_kary_partition_offset(OffsetT count, int partition)
{
  const OffsetT warp_size = static_cast<OffsetT>(detail::warp_threads);
  const OffsetT quotient  = count / warp_size;
  const OffsetT remainder = count % warp_size;
  return quotient * static_cast<OffsetT>(partition) + (remainder * static_cast<OffsetT>(partition)) / warp_size;
}

template <typename OffsetIteratorT, typename OffsetT, typename ValueT>
_CCCL_DEVICE _CCCL_FORCEINLINE OffsetT
warp_kary_upper_bound_offset(OffsetIteratorT offsets, OffsetT num_offsets, ValueT value)
{
  constexpr unsigned int warp_mask = 0xffffffffu;
  const int lane                   = static_cast<int>(threadIdx.x) & (detail::warp_threads - 1);

  OffsetT first = 0;
  OffsetT count = num_offsets;
  while (count > static_cast<OffsetT>(detail::warp_threads))
  {
    const OffsetT probe_offset = warp_kary_partition_offset(count, lane + 1) - OffsetT{1};
    const OffsetT probe        = first + probe_offset;
    const bool before          = offsets[probe] <= value;
    const unsigned int mask    = __ballot_sync(warp_mask, before);
    const int bucket           = __popc(mask);

    if (bucket == detail::warp_threads)
    {
      first += count;
      count = 0;
    }
    else
    {
      const OffsetT bucket_begin = warp_kary_partition_offset(count, bucket);
      const OffsetT bucket_end   = warp_kary_partition_offset(count, bucket + 1);
      first += bucket_begin;
      count = bucket_end - bucket_begin;
    }
  }

  const OffsetT result = lane == 0 ? upper_bound_offset_in_range(offsets, first, first + count, value) : OffsetT{0};
  return ShuffleIndex<detail::warp_threads>(result, 0, warp_mask);
}

template <typename OffsetT>
_CCCL_DEVICE _CCCL_FORCEINLINE boundary_record<OffsetT> make_even_boundary_record(
  const load_balanced_work_t<OffsetT>* offsets, OffsetT num_segments, load_balanced_work_t<OffsetT> boundary)
{
  const OffsetT first_segment = warp_kary_upper_bound_offset(offsets, num_segments + OffsetT{1}, boundary) - OffsetT{1};
  return {boundary, first_segment, offsets[first_segment] == boundary};
}

inline constexpr int work_index_threads          = 128;
inline constexpr int work_index_items_per_thread = 8;
inline constexpr int work_index_tile_items       = work_index_threads * work_index_items_per_thread;

//! @brief Initializes the tile state of the work-index scan.
//!
//! @tparam ScanTileStateT
//!   Tile status interface type
//!
template <typename ScanTileStateT>
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(work_index_threads) void device_segmented_scan_load_balanced_init_kernel(
  ScanTileStateT tile_state, int num_tiles)
{
  tile_state.InitializeStatus(num_tiles);
}

// A work-index value that also records whether some segment in its range is not followed by a segment beginning
// at its end.
template <typename OffsetT>
struct contiguity_work_index_value
{
  load_balanced_work_t<OffsetT> value;
  bool noncontiguous;
};

template <typename OffsetT>
struct contiguity_work_index_plus
{
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE contiguity_work_index_value<OffsetT>
  operator()(contiguity_work_index_value<OffsetT> lhs, contiguity_work_index_value<OffsetT> rhs) const
  {
    return {static_cast<load_balanced_work_t<OffsetT>>(lhs.value + rhs.value), lhs.noncontiguous || rhs.noncontiguous};
  }
};

template <typename OffsetT>
_CCCL_DEVICE _CCCL_FORCEINLINE load_balanced_work_t<OffsetT> make_work_index_value(OffsetT begin, OffsetT end)
{
  using work_t            = load_balanced_work_t<OffsetT>;
  using unsigned_offset_t = ::cuda::std::make_unsigned_t<OffsetT>;
  if (end <= begin)
  {
    return work_t{0};
  }
  return static_cast<work_t>(static_cast<unsigned_offset_t>(end))
       - static_cast<work_t>(static_cast<unsigned_offset_t>(begin));
}

// Type the work-index kernel scans.
template <typename OffsetT, bool PublishContiguity>
using work_index_value_t =
  ::cuda::std::conditional_t<PublishContiguity, contiguity_work_index_value<OffsetT>, load_balanced_work_t<OffsetT>>;

//! @brief Scans segment sizes into the work index P with a decoupled look-back. The grid also initializes the
//!        main kernel's tile state. scan_state is initialized by a previous launch when scan_tiles is greater
//!        than one.
//!
//! @tparam BeginOffsetIteratorT
//!   Random-access input iterator type for the segment beginning offsets
//!
//! @tparam EndOffsetIteratorT
//!   Random-access input iterator type for the segment ending offsets
//!
//! @tparam OffsetT
//!   Integer type for global offsets
//!
//! @tparam MainTileStateT
//!   Tile status interface type of the main kernel
//!
//! @tparam PublishContiguity
//!   Also store after P whether every segment is followed by a segment beginning at its end
//!
template <typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT,
          typename MainTileStateT,
          bool PublishContiguity>
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(work_index_threads) void device_segmented_scan_work_index_kernel(
  BeginOffsetIteratorT begin_offsets,
  EndOffsetIteratorT end_offsets,
  OffsetT num_segments,
  load_balanced_work_t<OffsetT>* work_index,
  ScanTileState<work_index_value_t<OffsetT, PublishContiguity>> scan_state,
  int scan_tiles,
  MainTileStateT main_state,
  int main_tiles)
{
  using work_t                               = load_balanced_work_t<OffsetT>;
  static constexpr bool publishes_contiguity = PublishContiguity;
  using value_t                              = work_index_value_t<OffsetT, PublishContiguity>;
  using scan_op_t =
    ::cuda::std::conditional_t<publishes_contiguity, contiguity_work_index_plus<OffsetT>, ::cuda::std::plus<work_t>>;
  using exchange_t   = BlockExchange<value_t, work_index_threads, work_index_items_per_thread>;
  using block_scan_t = BlockScan<value_t, work_index_threads, BLOCK_SCAN_WARP_SCANS>;
  using prefix_op_t  = TilePrefixCallbackOp<value_t, scan_op_t, ScanTileState<value_t>>;
  struct storage_t
  {
    typename prefix_op_t::TempStorage prefix;
    union
    {
      typename exchange_t::TempStorage exchange;
      typename block_scan_t::TempStorage scan;
    } tile;
  };
  __shared__ storage_t storage;

  main_state.InitializeStatus(main_tiles);
  if (blockIdx.x == 0 && threadIdx.x == 0)
  {
    work_index[0] = work_t{0};
  }
  const int tile = static_cast<int>(blockIdx.x);
  if (tile >= scan_tiles)
  {
    return;
  }

  const auto tile_begin = static_cast<::cuda::std::int64_t>(tile) * work_index_tile_items;
  const auto segments   = static_cast<::cuda::std::int64_t>(num_segments);
  value_t sizes[work_index_items_per_thread];
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int item = 0; item < work_index_items_per_thread; ++item)
  {
    const auto segment = tile_begin + item * work_index_threads + static_cast<int>(threadIdx.x);
    if constexpr (publishes_contiguity)
    {
      const auto index = static_cast<OffsetT>(segment);
      OffsetT begin{};
      OffsetT end{};
      if (segment < segments)
      {
        begin = static_cast<OffsetT>(begin_offsets[index]);
        end   = static_cast<OffsetT>(end_offsets[index]);
      }
      OffsetT next_begin  = __shfl_down_sync(0xffffffffu, begin, 1);
      const bool has_next = segment + 1 < segments;
      if (has_next && static_cast<int>(threadIdx.x) % detail::warp_threads == detail::warp_threads - 1)
      {
        next_begin = static_cast<OffsetT>(begin_offsets[index + OffsetT{1}]);
      }
      sizes[item] = {make_work_index_value(begin, end), has_next && next_begin != (::cuda::std::max) (end, begin)};
    }
    else if (segment < segments)
    {
      const auto index    = static_cast<OffsetT>(segment);
      const OffsetT begin = static_cast<OffsetT>(begin_offsets[index]);
      const OffsetT end   = static_cast<OffsetT>(end_offsets[index]);
      sizes[item]         = make_work_index_value(begin, end);
    }
    else
    {
      sizes[item] = value_t{};
    }
  }
  exchange_t(storage.tile.exchange).StripedToBlocked(sizes, sizes);
  __syncthreads();

  if (tile == 0)
  {
    value_t aggregate{};
    block_scan_t(storage.tile.scan).InclusiveScan(sizes, sizes, scan_op_t{}, aggregate);
    if (threadIdx.x == 0 && scan_tiles > 1)
    {
      scan_state.SetInclusive(0, aggregate);
    }
  }
  else
  {
    prefix_op_t prefix_op(scan_state, storage.prefix, scan_op_t{}, tile);
    block_scan_t(storage.tile.scan).InclusiveScan(sizes, sizes, scan_op_t{}, prefix_op);
  }
  __syncthreads();

  exchange_t(storage.tile.exchange).BlockedToStriped(sizes, sizes);
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int item = 0; item < work_index_items_per_thread; ++item)
  {
    const auto segment = tile_begin + item * work_index_threads + static_cast<int>(threadIdx.x);
    if (segment < segments)
    {
      if constexpr (publishes_contiguity)
      {
        work_index[segment + 1] = sizes[item].value;
        if (segment + 1 == segments)
        {
          work_index[segment + 2] = static_cast<work_t>(!sizes[item].noncontiguous);
        }
      }
      else
      {
        work_index[segment + 1] = sizes[item];
      }
    }
  }
}

// Output segments start where their input segments start.
struct reuse_input_begin
{};

//! @brief Addresses of the load-balanced scan's work items: item w of segment i lies at
//!        input_begin[i] + (w - P[i]) and output_begin[i] + (w - P[i]).
//!
//! @tparam InputBeginIteratorT
//!   Random-access input iterator type for the beginning offsets of the input segments
//!
//! @tparam OutputBeginIteratorT
//!   Random-access input iterator type for the beginning offsets of the output segments, or
//!   reuse_input_begin
//!
template <typename InputBeginIteratorT, typename OutputBeginIteratorT = reuse_input_begin>
struct general_addressing
{
  InputBeginIteratorT input_begin;
  OutputBeginIteratorT output_begin;
};

//! @brief agent_segmented_scan_load_balanced implements CTAs each scanning one range of the work items of a
//!        device-wide segmented prefix scan. A range may begin and end inside a segment.
//!
//! @tparam PolicyGetter
//!   Nullary callable type for getting SegmentedScanLoadBalancedPolicy
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
//! @tparam ForceInclusive
//!   Scan inclusively although an initial value is provided
//!
//! @tparam AddressingT
//!   Specialization of general_addressing
//!
template <typename PolicyGetter,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ScanOpT,
          typename InitValueT,
          typename AccumT,
          bool ForceInclusive,
          typename AddressingT>
struct agent_segmented_scan_load_balanced
{
private:
  using output_begin_iterator_t = decltype(AddressingT::output_begin);

  static constexpr auto policy                = PolicyGetter{}();
  static constexpr int threads_per_block      = policy.threads_per_block;
  static constexpr int items_per_thread       = policy.items_per_thread;
  static constexpr int items_per_tile         = threads_per_block * items_per_thread;
  static constexpr bool has_init              = !::cuda::std::is_same_v<InitValueT, NullType>;
  static constexpr bool is_inclusive          = ForceInclusive || !has_init;
  static constexpr bool has_output_deltas     = !::cuda::std::is_same_v<output_begin_iterator_t, reuse_input_begin>;
  static constexpr bool reads_contiguity_flag = !has_output_deltas;
  // Entry 0 of a tile's delta table is the segment carried into the tile; entry k > 0 is its k-th head.
  static constexpr int delta_table_size = items_per_tile + 1;

  using flag_value_t = KeyValuePair<int, AccumT>;
  using work_t       = load_balanced_work_t<OffsetT>;
  // Wrapping arithmetic: a segment's begin may lie below its work index, so deltas are stored modulo 2^n.
  using delta_t        = ::cuda::std::make_unsigned_t<OffsetT>;
  using pair_scan_op_t = ScanBySegmentOp<ScanOpT>;
  using input_value_t  = it_value_t<InputIteratorT>;
  using wrapped_input_iterator_t =
    ::cuda::std::conditional_t<::cuda::std::is_pointer_v<InputIteratorT>,
                               CacheModifiedInputIterator<policy.load_modifier, input_value_t, OffsetT>,
                               InputIteratorT>;
  using block_load_t       = BlockLoad<input_value_t, threads_per_block, items_per_thread, policy.load_algorithm>;
  using block_scan_t       = BlockScan<flag_value_t, threads_per_block, policy.scan_algorithm>;
  using block_store_t      = BlockStore<AccumT, threads_per_block, items_per_thread, policy.store_algorithm>;
  using block_count_scan_t = BlockScan<int, threads_per_block, policy.scan_algorithm>;
  using scan_tile_state_t  = ReduceByKeyScanTileState<AccumT, int>;
  using prefix_callback_t  = TilePrefixCallbackOp<flag_value_t, pair_scan_op_t, scan_tile_state_t>;

  union tile_storage_t
  {
    typename block_load_t::TempStorage load;
    typename block_scan_t::TempStorage scan;
    typename block_store_t::TempStorage store;
  };

  struct tile_deltas_t
  {
    Uninitialized<delta_t[delta_table_size]> input;
    Uninitialized<delta_t[has_output_deltas ? delta_table_size : 1]> output;
    typename block_count_scan_t::TempStorage count_scan;
    delta_t carried_input;
    delta_t carried_output;
    delta_t uniform_input;
    delta_t uniform_output;
    int entry_count;
    bool input_is_uniform;
    bool output_is_uniform;
    bool input_is_contiguous;
  };

  struct local_storage_t
  {
    tile_storage_t tile;
    Uninitialized<unsigned char[items_per_tile]> head_flags;
    Uninitialized<flag_value_t[threads_per_block]> thread_tails;
    tile_deltas_t deltas;
    OffsetT segment_cursor;
    work_t next_head;
    work_t tail_begin;
    OffsetT tail_segment;
    bool begins_at_head;
  };

  static_assert(sizeof(typename prefix_callback_t::TempStorage) <= sizeof(tile_storage_t),
                "look-back storage must not overlap the per-tile state after the tile storage");

  union _TempStorage
  {
    local_storage_t local;
    typename prefix_callback_t::TempStorage prefix;
  };

  _TempStorage& temp_storage;
  wrapped_input_iterator_t d_in;
  OutputIteratorT d_out;
  const work_t* offsets;
  ScanOpT scan_op;
  InitValueT init_value;
  pair_scan_op_t pair_scan_op;
  AddressingT addressing;

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static delta_t make_delta(OffsetT begin, work_t work)
  {
    return static_cast<delta_t>(begin) - static_cast<delta_t>(work);
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static OffsetT element_index(work_t work, delta_t delta)
  {
    return static_cast<OffsetT>(static_cast<delta_t>(work) + delta);
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static flag_value_t
  select_item(const flag_value_t (&items)[items_per_thread], int index)
  {
    flag_value_t result = items[items_per_thread - 1];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < items_per_thread - 1; ++item)
    {
      if (item == index)
      {
        result = items[item];
      }
    }
    return result;
  }

  // Largest index r in [known, num_segments] with offsets[r] == head. Requires offsets[known] == head.
  _CCCL_DEVICE _CCCL_FORCEINLINE OffsetT last_offset_equal_to(OffsetT known, work_t head, OffsetT num_segments) const
  {
    OffsetT lo   = known;
    OffsetT step = 1;
    while (step <= num_segments - lo && offsets[lo + step] == head)
    {
      lo += step;
      if (step <= num_segments - lo)
      {
        step += step;
      }
    }
    const OffsetT last = step <= num_segments - lo ? lo + step : num_segments + OffsetT{1};
    return upper_bound_offset_in_range(offsets, lo + OffsetT{1}, last, head) - OffsetT{1};
  }

  // first_segment is the segment holding range_begin.
  _CCCL_DEVICE _CCCL_FORCEINLINE void
  find_tail(work_t range_begin, work_t range_end, OffsetT num_segments, OffsetT first_segment)
  {
    const OffsetT tail_segment =
      upper_bound_offset_in_range(offsets, first_segment + OffsetT{1}, num_segments + OffsetT{1}, range_end - work_t{1})
      - OffsetT{1};
    const work_t tail_head          = offsets[tail_segment];
    temp_storage.local.tail_begin   = (::cuda::std::max) (range_begin, tail_head);
    temp_storage.local.tail_segment = tail_segment;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void
  initialize_cursor([[maybe_unused]] work_t range_begin, OffsetT num_segments, OffsetT first_segment)
  {
    auto& shared_head_flags = temp_storage.local.head_flags.Alias();
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < items_per_thread; ++item)
    {
      shared_head_flags[static_cast<int>(threadIdx.x) * items_per_thread + item] = 0;
    }
    if (threadIdx.x == 0)
    {
      bool input_is_contiguous = false;
      if constexpr (reads_contiguity_flag)
      {
        auto& deltas               = temp_storage.local.deltas;
        input_is_contiguous        = offsets[num_segments + OffsetT{1}] != work_t{0};
        deltas.input_is_contiguous = input_is_contiguous;
        if (input_is_contiguous)
        {
          deltas.uniform_input    = make_delta(static_cast<OffsetT>(addressing.input_begin[0]), work_t{0});
          deltas.input_is_uniform = true;
        }
      }
      _CCCL_ASSERT(first_segment < num_segments, "the first segment of a range exists");
      const work_t head = offsets[first_segment];
      _CCCL_ASSERT(head <= range_begin, "a range begins in its first segment");
      temp_storage.local.segment_cursor = first_segment;
      temp_storage.local.next_head      = head;
      if (!input_is_contiguous)
      {
        temp_storage.local.deltas.carried_input =
          make_delta(static_cast<OffsetT>(addressing.input_begin[first_segment]), head);
        if constexpr (has_output_deltas)
        {
          temp_storage.local.deltas.carried_output =
            make_delta(static_cast<OffsetT>(addressing.output_begin[first_segment]), head);
        }
      }
    }
    __syncthreads();
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void record_head(OffsetT segment, work_t head, work_t tile_begin, int entry)
  {
    temp_storage.local.head_flags.Alias()[static_cast<int>(head - tile_begin)] = 1;
    temp_storage.local.deltas.input.Alias()[entry] =
      make_delta(static_cast<OffsetT>(addressing.input_begin[segment]), head);
    if constexpr (has_output_deltas)
    {
      temp_storage.local.deltas.output.Alias()[entry] =
        make_delta(static_cast<OffsetT>(addressing.output_begin[segment]), head);
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void finish_tile_deltas()
  {
    const int lane        = static_cast<int>(threadIdx.x);
    auto& deltas          = temp_storage.local.deltas;
    const int entry_count = deltas.entry_count;
    const int first_entry = (entry_count > 0 && temp_storage.local.head_flags.Alias()[0] != 0) ? 1 : 0;

    const auto check = [&](const delta_t* table, delta_t& uniform_delta, bool& is_uniform, delta_t& carried) {
      const delta_t reference = table[first_entry];
      bool mismatch           = false;
      for (int entry = first_entry + 1 + lane; entry <= entry_count; entry += detail::warp_threads)
      {
        mismatch = mismatch || table[entry] != reference;
      }
      const bool any_mismatch = __any_sync(0xffffffffu, mismatch);
      if (lane == 0)
      {
        uniform_delta = reference;
        is_uniform    = !any_mismatch;
        carried       = table[entry_count];
      }
    };
    check(deltas.input.Alias(), deltas.uniform_input, deltas.input_is_uniform, deltas.carried_input);
    if constexpr (has_output_deltas)
    {
      check(deltas.output.Alias(), deltas.uniform_output, deltas.output_is_uniform, deltas.carried_output);
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void discover_general_heads(work_t tile_begin, work_t tile_end, OffsetT num_segments)
  {
    if (threadIdx.x >= detail::warp_threads)
    {
      return;
    }
    auto& deltas = temp_storage.local.deltas;
    if (threadIdx.x == 0)
    {
      deltas.input.Alias()[0] = deltas.carried_input;
      if constexpr (has_output_deltas)
      {
        deltas.output.Alias()[0] = deltas.carried_output;
      }

      OffsetT segment = temp_storage.local.segment_cursor;
      work_t head     = temp_storage.local.next_head;
      while (segment < num_segments && head < tile_begin)
      {
        ++segment;
        if (segment < num_segments)
        {
          head = offsets[segment];
        }
      }

      int entry_count = 0;
      while (segment < num_segments && head < tile_end)
      {
        const work_t next = offsets[segment + OffsetT{1}];
        if (next != head)
        {
          record_head(segment, head, tile_begin, ++entry_count);
        }
        else
        {
          // Skip the run of empty segments at head.
          segment = last_offset_equal_to(segment + OffsetT{1}, head, num_segments) - OffsetT{1};
        }
        ++segment;
        head = next;
      }
      temp_storage.local.segment_cursor = segment;
      temp_storage.local.next_head      = head;
      deltas.entry_count                = entry_count;
    }
    __syncwarp();
    finish_tile_deltas();
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void load_general_tile(
    work_t tile_begin,
    work_t tile_end,
    OffsetT num_segments,
    flag_value_t (&items)[items_per_thread],
    int (&entries)[items_per_thread])
  {
    const int valid_items = static_cast<int>(tile_end - tile_begin);
    discover_general_heads(tile_begin, tile_end, num_segments);
    __syncthreads();

    auto& shared_head_flags = temp_storage.local.head_flags.Alias();
    int head_counts[items_per_thread];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < items_per_thread; ++item)
    {
      const int local_index          = static_cast<int>(threadIdx.x) * items_per_thread + item;
      head_counts[item]              = shared_head_flags[local_index];
      shared_head_flags[local_index] = 0;
    }

    auto& deltas             = temp_storage.local.deltas;
    const delta_t tile_delta = deltas.uniform_input;
    const bool gather        = !deltas.input_is_uniform;
    const bool needs_entries = gather || (has_output_deltas && !deltas.output_is_uniform);
    const OffsetT tile_start = element_index(tile_begin, tile_delta);
    input_value_t values[items_per_thread];
    if (needs_entries)
    {
      block_count_scan_t(deltas.count_scan).InclusiveSum(head_counts, entries);
    }
    if (gather)
    {
      const delta_t* input_deltas = deltas.input.Alias();
      // Pads the items past the end of a partial tile.
      input_value_t filler{};
      if ((static_cast<int>(threadIdx.x) + 1) * items_per_thread > valid_items)
      {
        filler = d_in[tile_start];
      }
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = 0; item < items_per_thread; ++item)
      {
        const int local_index = static_cast<int>(threadIdx.x) * items_per_thread + item;
        values[item] = local_index < valid_items
                       ? d_in[element_index(tile_begin + static_cast<work_t>(local_index), input_deltas[entries[item]])]
                       : filler;
      }
    }
    else if (valid_items == items_per_tile)
    {
      block_load_t(temp_storage.local.tile.load).Load(d_in + tile_start, values);
    }
    else
    {
      block_load_t(temp_storage.local.tile.load).Load(d_in + tile_start, values, valid_items, d_in[tile_start]);
    }
    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < items_per_thread; ++item)
    {
      const bool is_head = head_counts[item] != 0;
      items[item]        = flag_value_t{static_cast<int>(is_head), values[item]};
      if constexpr (has_init)
      {
        if (is_head)
        {
          items[item].value = scan_op(init_value, items[item].value);
        }
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void store_tile(
    work_t tile_begin, int valid_items, AccumT (&output_items)[items_per_thread], const int (&entries)[items_per_thread])
  {
    auto& deltas          = temp_storage.local.deltas;
    const bool is_uniform = has_output_deltas ? deltas.output_is_uniform : deltas.input_is_uniform;
    const delta_t delta   = has_output_deltas ? deltas.uniform_output : deltas.uniform_input;
    if (!is_uniform)
    {
      const delta_t* table = has_output_deltas ? deltas.output.Alias() : deltas.input.Alias();
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = 0; item < items_per_thread; ++item)
      {
        const int local_index = static_cast<int>(threadIdx.x) * items_per_thread + item;
        if (local_index < valid_items)
        {
          d_out[element_index(tile_begin + static_cast<work_t>(local_index), table[entries[item]])] =
            output_items[item];
        }
      }
      return;
    }
    const OffsetT tile_start = element_index(tile_begin, delta);
    if (valid_items == items_per_tile)
    {
      block_store_t(temp_storage.local.tile.store).Store(d_out + tile_start, output_items);
    }
    else
    {
      block_store_t(temp_storage.local.tile.store).Store(d_out + tile_start, output_items, valid_items);
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void load_tile(
    work_t tile_begin,
    work_t tile_end,
    OffsetT num_segments,
    flag_value_t (&items)[items_per_thread],
    int (&entries)[items_per_thread])
  {
    if constexpr (reads_contiguity_flag)
    {
      if (temp_storage.local.deltas.input_is_contiguous)
      {
        load_contiguous_tile(tile_begin, tile_end, num_segments, items, temp_storage.local.deltas.uniform_input);
        return;
      }
    }
    load_general_tile(tile_begin, tile_end, num_segments, items, entries);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void load_contiguous_tile(
    work_t tile_begin, work_t tile_end, OffsetT num_segments, flag_value_t (&items)[items_per_thread], delta_t delta)
  {
    const int valid_items    = static_cast<int>(tile_end - tile_begin);
    const OffsetT tile_start = element_index(tile_begin, delta);
    input_value_t values[items_per_thread];

    if (valid_items == items_per_tile)
    {
      block_load_t(temp_storage.local.tile.load).Load(d_in + tile_start, values);
    }
    else
    {
      block_load_t(temp_storage.local.tile.load).Load(d_in + tile_start, values, valid_items, d_in[tile_start]);
    }

    auto& shared_head_flags = temp_storage.local.head_flags.Alias();
    if (threadIdx.x == 0)
    {
      OffsetT segment = temp_storage.local.segment_cursor;
      work_t head     = temp_storage.local.next_head;
      while (segment < num_segments && head < tile_begin)
      {
        ++segment;
        if (segment < num_segments)
        {
          head = offsets[segment];
        }
      }

      while (segment < num_segments && head < tile_end)
      {
        shared_head_flags[static_cast<int>(head - tile_begin)] = 1;
        ++segment;
        if (segment < num_segments)
        {
          const work_t next = offsets[segment];
          if (next == head)
          {
            // Skip the run of empty segments at head.
            segment = last_offset_equal_to(segment, head, num_segments);
          }
          head = next;
        }
      }
      temp_storage.local.segment_cursor = segment;
      temp_storage.local.next_head      = head;
    }
    __syncthreads();

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < items_per_thread; ++item)
    {
      const int local_index          = static_cast<int>(threadIdx.x) * items_per_thread + item;
      const bool is_head             = shared_head_flags[local_index] != 0;
      shared_head_flags[local_index] = 0;
      items[item]                    = flag_value_t{static_cast<int>(is_head), values[item]};
      if constexpr (has_init)
      {
        if (is_head)
        {
          items[item].value = scan_op(init_value, items[item].value);
        }
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE flag_value_t
  reduce_range(work_t range_begin, work_t range_end, OffsetT num_segments, OffsetT first_segment)
  {
    initialize_cursor(range_begin, num_segments, first_segment);

    flag_value_t range_aggregate{};
    bool first_tile = true;
    for (work_t tile_begin = range_begin; tile_begin < range_end;)
    {
      const work_t tile_size = (::cuda::std::min) (range_end - tile_begin, static_cast<work_t>(items_per_tile));
      const work_t tile_end  = tile_begin + tile_size;
      const int valid_items  = static_cast<int>(tile_end - tile_begin);
      flag_value_t items[items_per_thread];
      int entries[items_per_thread];
      load_tile(tile_begin, tile_end, num_segments, items, entries);

      flag_value_t ignored_aggregate;
      block_scan_t(temp_storage.local.tile.scan).InclusiveScan(items, items, pair_scan_op, ignored_aggregate);
      __syncthreads();

      const int last_index = valid_items - 1;
      if (static_cast<int>(threadIdx.x) == last_index / items_per_thread)
      {
        temp_storage.local.thread_tails.Alias()[0] =
          valid_items == items_per_tile
            ? items[items_per_thread - 1]
            : select_item(items, last_index % items_per_thread);
      }
      __syncthreads();

      const flag_value_t tile_aggregate = temp_storage.local.thread_tails.Alias()[0];
      range_aggregate                   = first_tile ? tile_aggregate : pair_scan_op(range_aggregate, tile_aggregate);
      first_tile                        = false;
      __syncthreads();
      tile_begin = tile_end;
    }
    return range_aggregate;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE flag_value_t
  obtain_prefix(flag_value_t range_aggregate, int tile_idx, scan_tile_state_t& tile_state)
  {
    _CCCL_ASSERT(tile_idx > 0, "the first block begins at a segment head");

    prefix_callback_t prefix_op(tile_state, temp_storage.prefix, pair_scan_op, tile_idx);
    if (threadIdx.x < detail::warp_threads)
    {
      prefix_op(range_aggregate);
    }
    __syncthreads();
    return prefix_op.GetExclusivePrefix();
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE flag_value_t obtain_predecessor_prefix(int tile_idx, scan_tile_state_t& tile_state)
  {
    _CCCL_ASSERT(tile_idx > 0, "the first block begins at a segment head");

    prefix_callback_t prefix_op(tile_state, temp_storage.prefix, pair_scan_op, tile_idx);
    if (threadIdx.x < detail::warp_threads)
    {
      using status_word_t       = typename prefix_callback_t::StatusWord;
      using delay_constructor_t = detail::default_delay_constructor_t<flag_value_t>;
      int predecessor_idx       = tile_idx - threadIdx.x - 1;
      status_word_t predecessor_status;
      flag_value_t window_aggregate;
      delay_constructor_t construct_delay(tile_idx);
      prefix_op.ProcessWindow(predecessor_idx, predecessor_status, window_aggregate, construct_delay());

      flag_value_t exclusive_prefix = window_aggregate;
      while (__all_sync(0xffffffff, (predecessor_status != status_word_t(SCAN_TILE_INCLUSIVE))))
      {
        predecessor_idx -= detail::warp_threads;
        prefix_op.ProcessWindow(predecessor_idx, predecessor_status, window_aggregate, construct_delay());
        exclusive_prefix = pair_scan_op(window_aggregate, exclusive_prefix);
      }

      if (threadIdx.x == 0)
      {
        detail::uninitialized_copy_single(&temp_storage.local.thread_tails.Alias()[0], exclusive_prefix);
      }
    }
    __syncthreads();
    return temp_storage.local.thread_tails.Alias()[0];
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void
  publish_without_lookback(flag_value_t range_aggregate, int tile_idx, scan_tile_state_t& tile_state)
  {
    if (threadIdx.x == 0)
    {
      tile_state.SetInclusive(tile_idx, range_aggregate);
    }
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE flag_value_t scan_range(
    work_t range_begin, work_t range_end, OffsetT num_segments, flag_value_t range_prefix, OffsetT first_segment)
  {
    initialize_cursor(range_begin, num_segments, first_segment);
    flag_value_t running_prefix = range_prefix;

    for (work_t tile_begin = range_begin; tile_begin < range_end;)
    {
      const work_t tile_size = (::cuda::std::min) (range_end - tile_begin, static_cast<work_t>(items_per_tile));
      const work_t tile_end  = tile_begin + tile_size;
      const int valid_items  = static_cast<int>(tile_end - tile_begin);
      flag_value_t items[items_per_thread];
      int entries[items_per_thread];
      bool head_flags[items_per_thread];
      load_tile(tile_begin, tile_end, num_segments, items, entries);

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = 0; item < items_per_thread; ++item)
      {
        head_flags[item] = items[item].key != 0;
      }

      flag_value_t ignored_aggregate;
      block_scan_t(temp_storage.local.tile.scan).InclusiveScan(items, items, pair_scan_op, ignored_aggregate);
      __syncthreads();

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = 0; item < items_per_thread; ++item)
      {
        if (items[item].key == 0)
        {
          items[item] = pair_scan_op(running_prefix, items[item]);
        }
      }

      auto& thread_tails    = temp_storage.local.thread_tails.Alias();
      const int local_begin = static_cast<int>(threadIdx.x) * items_per_thread;
      if (local_begin < valid_items)
      {
        const int thread_items = valid_items - local_begin;
        const int last_item    = (thread_items < items_per_thread ? thread_items : items_per_thread) - 1;
        thread_tails[threadIdx.x] =
          valid_items == items_per_tile ? items[items_per_thread - 1] : select_item(items, last_item);
      }
      __syncthreads();

      AccumT output_items[items_per_thread];
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = 0; item < items_per_thread; ++item)
      {
        const int local_index = static_cast<int>(threadIdx.x) * items_per_thread + item;
        if (local_index < valid_items)
        {
          if constexpr (is_inclusive)
          {
            output_items[item] = items[item].value;
          }
          else
          {
            output_items[item] =
              head_flags[item]
                ? static_cast<AccumT>(init_value)
                : (item == 0 ? (threadIdx.x == 0 ? running_prefix.value : thread_tails[threadIdx.x - 1].value)
                             : items[item - 1].value);
          }
        }
      }

      const int last_thread = (valid_items - 1) / items_per_thread;
      running_prefix        = thread_tails[last_thread];
      store_tile(tile_begin, valid_items, output_items, entries);
      __syncthreads();
      tile_begin = tile_end;
    }
    return running_prefix;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_forward_range(
    work_t range_begin,
    work_t range_end,
    OffsetT num_segments,
    OffsetT first_segment,
    int tile_idx,
    scan_tile_state_t& tile_state)
  {
    const flag_value_t range_aggregate = reduce_range(range_begin, range_end, num_segments, first_segment);
    __syncthreads();
    const flag_value_t range_prefix = obtain_prefix(range_aggregate, tile_idx, tile_state);
    __syncthreads();
    static_cast<void>(scan_range(range_begin, range_end, num_segments, range_prefix, first_segment));
  }

public:
  using TempStorage   = Uninitialized<_TempStorage>;
  using ScanTileState = scan_tile_state_t;

  _CCCL_DEVICE _CCCL_FORCEINLINE agent_segmented_scan_load_balanced(
    TempStorage& temp_storage,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    const work_t* offsets,
    ScanOpT scan_op,
    InitValueT init_value,
    AddressingT addressing)
      : temp_storage(temp_storage.Alias())
      , d_in(d_in)
      , d_out(d_out)
      , offsets(offsets)
      , scan_op(scan_op)
      , init_value(init_value)
      , pair_scan_op(scan_op)
      , addressing(addressing)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_suffix_first_range(
    work_t range_begin,
    work_t range_end,
    OffsetT num_segments,
    OffsetT first_segment,
    bool begins_at_head,
    int tile_idx,
    scan_tile_state_t& tile_state)
  {
    _CCCL_ASSERT(range_begin < range_end, "load-balanced ranges must not be empty");

    if (threadIdx.x == 0)
    {
      find_tail(range_begin, range_end, num_segments, first_segment);
      // Read after the suffix scan.
      temp_storage.local.begins_at_head = begins_at_head;
    }
    __syncthreads();

    const bool has_independent_suffix = begins_at_head || temp_storage.local.tail_begin > range_begin;
    if (!has_independent_suffix)
    {
      consume_forward_range(range_begin, range_end, num_segments, first_segment, tile_idx, tile_state);
      return;
    }

    const flag_value_t suffix_aggregate = scan_range(
      temp_storage.local.tail_begin, range_end, num_segments, flag_value_t{}, temp_storage.local.tail_segment);
    __syncthreads();
    publish_without_lookback(suffix_aggregate, tile_idx, tile_state);
    __syncthreads();

    if (temp_storage.local.tail_begin == range_begin)
    {
      return;
    }

    const flag_value_t range_prefix =
      temp_storage.local.begins_at_head ? flag_value_t{} : obtain_predecessor_prefix(tile_idx, tile_state);
    __syncthreads();
    static_cast<void>(
      scan_range(range_begin, temp_storage.local.tail_begin, num_segments, range_prefix, first_segment));
  }
};

template <typename WorkT>
struct range_split
{
  WorkT total;
  int virtual_blocks;
  // Read from shared storage by every thread.
  int block;
};

//! @brief Scans the block's range of the work items. The work is split evenly across the blocks the grid
//!        uses, and a range that begins inside a segment takes that segment's prefix from the blocks before it
//!        with a decoupled look-back.
//!
//! @tparam LoadBalancedPolicySelector
//!   Selector type for SegmentedScanLoadBalancedPolicy
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
//!   Type wrapping the init_value element for ScanOpT; its value_type is cub::NullType for inclusive scan
//!
//! @tparam AccumT
//!   The type of intermediate accumulator (according to P2322R6)
//!
//! @tparam ForceInclusive
//!   Scan inclusively although an initial value is provided
//!
//! @tparam AddressingT
//!   Specialization of general_addressing
//!
template <typename LoadBalancedPolicySelector,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ScanOpT,
          typename InitValueT,
          typename AccumT,
          bool ForceInclusive,
          typename AddressingT,
          typename ActualInitValueT = typename InitValueT::value_type>
#if _CCCL_HAS_CONCEPTS()
  requires segmented_scan_load_balanced_policy_selector<LoadBalancedPolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
__launch_bounds__(current_policy<LoadBalancedPolicySelector>().threads_per_block)
  _CCCL_KERNEL_ATTRIBUTES void device_segmented_scan_load_balanced_kernel(
    InputIteratorT d_in,
    OutputIteratorT d_out,
    const load_balanced_work_t<OffsetT>* work_index,
    OffsetT num_segments,
    ScanOpT scan_op,
    InitValueT init_value,
    int resident_blocks,
    int min_tiles_per_block,
    int max_factor,
    int capacity,
    ReduceByKeyScanTileState<AccumT, int> tile_state,
    AddressingT addressing)
{
  using work_t                 = load_balanced_work_t<OffsetT>;
  static constexpr auto policy = current_policy<LoadBalancedPolicySelector>();
  static_assert(policy.load_modifier != CacheLoadModifier::LOAD_LDG,
                "The memory consistency model does not apply to texture accesses");
  static_assert(policy.threads_per_block >= detail::warp_threads, "load-balanced segmented scan requires a full warp");
  static_assert(policy.items_per_thread > 0, "Policy value for items_per_thread is not positive");
  struct policy_getter
  {
    constexpr auto operator()() const
    {
      return policy;
    }
  };

  using agent_t = agent_segmented_scan_load_balanced<
    policy_getter,
    InputIteratorT,
    OutputIteratorT,
    OffsetT,
    ScanOpT,
    ActualInitValueT,
    AccumT,
    ForceInclusive,
    AddressingT>;

  __shared__ typename agent_t::TempStorage temp_storage;
  __shared__ boundary_record<OffsetT> begin_record;
  __shared__ range_split<work_t> split;
  static_assert(sizeof(typename agent_t::TempStorage) + sizeof(boundary_record<OffsetT>) + sizeof(range_split<work_t>)
                  <= max_smem_per_block,
                "The load-balanced segmented scan's shared memory exceeds max_smem_per_block: reduce the accumulator "
                "size, the policy's tile or the offset width");

  _CCCL_ASSERT(num_segments < ::cuda::std::numeric_limits<OffsetT>::max(), "the work index is indexed by OffsetT");

  constexpr int tile_size = policy.threads_per_block * policy.items_per_thread;
  if (threadIdx.x == 0)
  {
    split.block = static_cast<int>(blockIdx.x);
    split.total = work_index[num_segments];
    split.virtual_blocks =
      (::cuda::std::min) (load_balanced_block_count(
                            split.total,
                            static_cast<work_t>(tile_size),
                            resident_blocks,
                            min_tiles_per_block,
                            max_factor),
                          capacity);
  }

  const ActualInitValueT actual_init_value = init_value;
  agent_t agent(temp_storage, d_in, d_out, work_index, scan_op, actual_init_value, addressing);

  if (threadIdx.x < detail::warp_threads)
  {
    __syncwarp();
    if (split.block < split.virtual_blocks)
    {
      const int block    = split.block;
      const work_t begin = even_split_boundary(split.total, split.virtual_blocks, block);
      const auto record  = make_even_boundary_record(work_index, num_segments, begin);
      if (threadIdx.x == 0)
      {
        begin_record = record;
      }
    }
  }
  __syncthreads();

  const int block          = split.block;
  const int virtual_blocks = split.virtual_blocks;
  if (block >= virtual_blocks)
  {
    return;
  }

  const boundary_record<OffsetT> begin = begin_record;
  const work_t end                     = even_split_boundary(split.total, virtual_blocks, block + 1);
  agent.consume_suffix_first_range(
    begin.offset, end, num_segments, begin.first_segment, begin.is_head, block, tile_state);
}
} // namespace detail::segmented_scan

CUB_NAMESPACE_END
