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
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>
#include <cub/block/block_store.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_scan.cuh>
#include <cub/iterator/cache_modified_input_iterator.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_macro.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
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
// Type of logical work positions for n-bit offsets, where TotalIsSupplied says whether the call supplies the total item
// count. Output segments do not overlap, so outputs whose positions fit OffsetT total at most 2^n. n-bit work holds a
// supplied total below 2^n and, by precondition, the total for a signed OffsetT, which reaches 2^n only when every
// negative output position is written. An unsigned OffsetT without a supplied total uses 64 bits.
template <typename OffsetT, bool TotalIsSupplied>
using load_balanced_work_t =
  ::cuda::std::conditional_t<::cuda::std::is_signed_v<OffsetT> || TotalIsSupplied,
                             ::cuda::std::make_unsigned_t<OffsetT>,
                             ::cuda::std::uint64_t>;

// Type of logical work positions when the total work may reach 2^32 with 32-bit offsets.
using wide_load_balanced_work_t = ::cuda::std::uint64_t;

// Type of an output position, a segment's output begin offset plus an index within the segment. Work wider than
// OffsetT gives it 64 bits, so that output positions need not fit OffsetT.
template <typename OffsetT, typename WorkT>
using load_balanced_output_position_t =
  ::cuda::std::conditional_t<(sizeof(WorkT) > sizeof(OffsetT)), ::cuda::std::int64_t, OffsetT>;

// A range's begin and the segment holding it, and the segment holding the range's last item. tail_begin is the later
// of the range's begin and that segment's head.
template <typename OffsetT, typename WorkT>
struct boundary_record
{
  WorkT offset;
  WorkT tail_begin;
  OffsetT first_segment;
  OffsetT tail_segment;
  bool is_head;
};

template <typename WorkT>
struct range_split
{
  WorkT total;
  int virtual_blocks;
};

// Tiles each block should receive before the grid grows past one wave
inline constexpr int min_tiles_per_block = 16;
// Most waves of resident blocks the grid launches
inline constexpr int max_factor = 5;

// Returns min(tiles, F x resident_blocks), where
// F = clamp(floor(tiles / (min_tiles_per_block x resident_blocks)), 1, max_factor).
template <typename WorkT>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr int
load_balanced_block_count(WorkT total_work, WorkT tile_size, int resident_blocks)
{
  const WorkT tiles      = ::cuda::ceil_div(total_work, tile_size);
  const WorkT wave_tiles = static_cast<WorkT>(resident_blocks);
  const WorkT factor     = ::cuda::std::clamp(
    tiles / (static_cast<WorkT>(min_tiles_per_block) * wave_tiles), WorkT{1}, static_cast<WorkT>(max_factor));
  return static_cast<int>((::cuda::std::min) (tiles, factor * wave_tiles));
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

// Finds the upper bounds of both values as warp_kary_upper_bound_offset does, in the same rounds, and resolves the
// last at most warp_threads candidates of each with one more round.
template <typename OffsetIteratorT, typename OffsetT, typename ValueT>
_CCCL_DEVICE _CCCL_FORCEINLINE void warp_kary_upper_bound_offsets(
  OffsetIteratorT offsets, OffsetT num_offsets, const ValueT (&values)[2], OffsetT (&first)[2])
{
  constexpr unsigned int warp_mask = 0xffffffffu;
  constexpr OffsetT warp_size      = static_cast<OffsetT>(detail::warp_threads);
  const int lane                   = static_cast<int>(threadIdx.x) & (detail::warp_threads - 1);

  OffsetT count[2] = {num_offsets, num_offsets};
  first[0]         = 0;
  first[1]         = 0;
  while (count[0] > warp_size || count[1] > warp_size)
  {
    bool before[2] = {false, false};
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int search = 0; search < 2; ++search)
    {
      if (count[search] > warp_size)
      {
        const OffsetT probe = first[search] + warp_kary_partition_offset(count[search], lane + 1) - OffsetT{1};
        before[search]      = offsets[probe] <= values[search];
      }
    }
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int search = 0; search < 2; ++search)
    {
      if (count[search] > warp_size)
      {
        const int bucket = __popc(__ballot_sync(warp_mask, before[search]));
        if (bucket == detail::warp_threads)
        {
          first[search] += count[search];
          count[search] = 0;
        }
        else
        {
          const OffsetT bucket_begin = warp_kary_partition_offset(count[search], bucket);
          const OffsetT bucket_end   = warp_kary_partition_offset(count[search], bucket + 1);
          first[search] += bucket_begin;
          count[search] = bucket_end - bucket_begin;
        }
      }
    }
  }

  bool before[2] = {false, false};
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int search = 0; search < 2; ++search)
  {
    if (static_cast<OffsetT>(lane) < count[search])
    {
      before[search] = offsets[first[search] + static_cast<OffsetT>(lane)] <= values[search];
    }
  }
  _CCCL_PRAGMA_UNROLL_FULL()
  for (int search = 0; search < 2; ++search)
  {
    first[search] += static_cast<OffsetT>(__popc(__ballot_sync(warp_mask, before[search])));
  }
}

// Run by a full warp. range_begin is less than range_end.
template <typename OffsetT, typename WorkT>
_CCCL_DEVICE _CCCL_FORCEINLINE boundary_record<OffsetT, WorkT>
make_even_boundary_record(const WorkT* offsets, OffsetT num_segments, WorkT range_begin, WorkT range_end)
{
  using work_t           = WorkT;
  const work_t values[2] = {range_begin, range_end - work_t{1}};
  OffsetT bounds[2];
  warp_kary_upper_bound_offsets(offsets, num_segments + OffsetT{1}, values, bounds);
  const OffsetT first_segment = bounds[0] - OffsetT{1};
  const OffsetT tail_segment  = bounds[1] - OffsetT{1};
  const work_t head           = offsets[first_segment];
  const work_t tail_head      = offsets[tail_segment];
  return {range_begin, (::cuda::std::max) (range_begin, tail_head), first_segment, tail_segment, head == range_begin};
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

// Output segments start where their input segments start.
struct reuse_input_begin
{};

// A work-index value that also records, for the input and for the output, whether some segment in its range is not
// followed by a segment beginning at its end, and whether some output segment in its range ends at the last output
// position.
template <typename WorkT>
struct contiguity_work_index_value
{
  WorkT value;
  bool noncontiguous_input;
  bool noncontiguous_output;
  bool output_reaches_maximum;
};

template <typename WorkT>
struct contiguity_work_index_plus
{
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE contiguity_work_index_value<WorkT>
  operator()(contiguity_work_index_value<WorkT> lhs, contiguity_work_index_value<WorkT> rhs) const
  {
    return {static_cast<WorkT>(lhs.value + rhs.value),
            lhs.noncontiguous_input || rhs.noncontiguous_input,
            lhs.noncontiguous_output || rhs.noncontiguous_output,
            lhs.output_reaches_maximum || rhs.output_reaches_maximum};
  }
};

// Bits of work_index[num_segments + 1].
inline constexpr unsigned int contiguous_input_flag       = 1;
inline constexpr unsigned int contiguous_output_flag      = 2;
inline constexpr unsigned int output_reaches_maximum_flag = 4;

template <typename WorkT, typename OffsetT>
_CCCL_DEVICE _CCCL_FORCEINLINE WorkT make_work_index_value(OffsetT begin, OffsetT end)
{
  using unsigned_offset_t = ::cuda::std::make_unsigned_t<OffsetT>;
  if (end <= begin)
  {
    return WorkT{0};
  }
  return static_cast<WorkT>(static_cast<unsigned_offset_t>(end) - static_cast<unsigned_offset_t>(begin));
}

// Whether a segment of size items at begin is followed by a segment beginning at next, modulo 2^n for n-bit positions,
// as the main kernel adds address deltas.
template <typename PositionT, typename WorkT>
_CCCL_DEVICE _CCCL_FORCEINLINE bool is_followed_by(PositionT begin, WorkT size, PositionT next)
{
  using unsigned_position_t = ::cuda::std::make_unsigned_t<PositionT>;
  return static_cast<unsigned_position_t>(
           static_cast<unsigned_position_t>(next) - static_cast<unsigned_position_t>(begin))
      == static_cast<unsigned_position_t>(size);
}

// Whether a segment of size items at begin holds the last position of PositionT. Only then can a segment that follows
// it modulo 2^n for n-bit positions begin at PositionT's first position.
template <typename PositionT, typename WorkT>
_CCCL_DEVICE _CCCL_FORCEINLINE bool reaches_maximum(PositionT begin, WorkT size)
{
  using unsigned_position_t = ::cuda::std::make_unsigned_t<PositionT>;
  return size != WorkT{0}
      && static_cast<unsigned_position_t>(
           static_cast<unsigned_position_t>(begin) + static_cast<unsigned_position_t>(size - WorkT{1}))
           == static_cast<unsigned_position_t>(::cuda::std::numeric_limits<PositionT>::max());
}

//! @brief Scans segment sizes with a decoupled look-back into the work index, where work_index[i] is the number of
//!        items in the segments before segment i, for i in [0, num_segments], and stores in
//!        work_index[num_segments + 1] contiguous_input_flag if every input segment is followed by a segment beginning
//!        at its end, contiguous_output_flag if every output segment is and none holds the last output position, and
//!        output_reaches_maximum_flag if one does. The grid also initializes the main kernel's tile state. scan_state
//!        is initialized by a previous launch when scan_tiles is greater than one.
//!
//! @tparam BeginOffsetIteratorT
//!   Random-access input iterator type for the segment beginning offsets
//!
//! @tparam EndOffsetIteratorT
//!   Random-access input iterator type for the segment ending offsets
//!
//! @tparam OutputBeginOffsetIteratorT
//!   Random-access input iterator type for the beginning offsets of the output segments, or reuse_input_begin
//!
//! @tparam OffsetT
//!   Integer type for global offsets
//!
//! @tparam WorkT
//!   Unsigned integer type for logical work positions
//!
//! @tparam MainTileStateT
//!   Tile status interface type of the main kernel
//!
template <typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OutputBeginOffsetIteratorT,
          typename OffsetT,
          typename WorkT,
          typename MainTileStateT>
_CCCL_KERNEL_ATTRIBUTES __launch_bounds__(work_index_threads) void device_segmented_scan_work_index_kernel(
  BeginOffsetIteratorT begin_offsets,
  EndOffsetIteratorT end_offsets,
  [[maybe_unused]] OutputBeginOffsetIteratorT output_begin_offsets,
  OffsetT num_segments,
  WorkT* work_index,
  ScanTileState<contiguity_work_index_value<WorkT>> scan_state,
  int scan_tiles,
  MainTileStateT main_state,
  int main_tiles)
{
  using work_t                            = WorkT;
  using output_position_t                 = load_balanced_output_position_t<OffsetT, WorkT>;
  static constexpr bool has_output_begins = !::cuda::std::is_same_v<OutputBeginOffsetIteratorT, reuse_input_begin>;
  // Input positions end at a supplied end offset, so only output positions reach the maximum, and 64-bit output
  // positions do not.
  static constexpr bool output_may_reach_maximum =
    has_output_begins && sizeof(output_position_t) < sizeof(::cuda::std::uint64_t);
  using value_t      = contiguity_work_index_value<WorkT>;
  using scan_op_t    = contiguity_work_index_plus<WorkT>;
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
    const auto index   = static_cast<OffsetT>(segment);
    OffsetT begin{};
    OffsetT end{};
    [[maybe_unused]] OffsetT output_begin{};
    if (segment < segments)
    {
      begin = static_cast<OffsetT>(begin_offsets[index]);
      end   = static_cast<OffsetT>(end_offsets[index]);
      if constexpr (has_output_begins)
      {
        output_begin = static_cast<OffsetT>(output_begin_offsets[index]);
      }
    }
    OffsetT next_begin = __shfl_down_sync(0xffffffffu, begin, 1);
    [[maybe_unused]] OffsetT next_output_begin{};
    if constexpr (has_output_begins)
    {
      next_output_begin = __shfl_down_sync(0xffffffffu, output_begin, 1);
    }
    const bool has_next = segment + 1 < segments;
    if (has_next && static_cast<int>(threadIdx.x) % detail::warp_threads == detail::warp_threads - 1)
    {
      next_begin = static_cast<OffsetT>(begin_offsets[index + OffsetT{1}]);
      if constexpr (has_output_begins)
      {
        next_output_begin = static_cast<OffsetT>(output_begin_offsets[index + OffsetT{1}]);
      }
    }
    const work_t size                  = make_work_index_value<work_t>(begin, end);
    sizes[item].value                  = size;
    sizes[item].noncontiguous_input    = has_next && !is_followed_by(begin, size, next_begin);
    sizes[item].noncontiguous_output   = false;
    sizes[item].output_reaches_maximum = false;
    if constexpr (has_output_begins)
    {
      sizes[item].noncontiguous_output =
        has_next
        && !is_followed_by(
          static_cast<output_position_t>(output_begin), size, static_cast<output_position_t>(next_output_begin));
    }
    if constexpr (output_may_reach_maximum)
    {
      sizes[item].output_reaches_maximum = reaches_maximum(static_cast<output_position_t>(output_begin), size);
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
      work_index[segment + 1] = sizes[item].value;
      if (segment + 1 == segments)
      {
        const bool output_reaches_maximum = sizes[item].output_reaches_maximum;
        const bool noncontiguous_output = has_output_begins ? sizes[item].noncontiguous_output || output_reaches_maximum
                                                            : sizes[item].noncontiguous_input;
        work_index[segment + 2]         = static_cast<work_t>(
          (sizes[item].noncontiguous_input ? 0u : contiguous_input_flag)
          | (noncontiguous_output ? 0u : contiguous_output_flag)
          | (output_reaches_maximum ? output_reaches_maximum_flag : 0u));
      }
    }
  }
}

//! @brief Addresses of the load-balanced scan's work items: work item w, in segment i, lies at
//!        input_begin[i] + (w - work_index[i]) and output_begin[i] + (w - work_index[i]).
//!
//! @tparam InputBeginIteratorT
//!   Random-access input iterator type for the beginning offsets of the input segments
//!
//! @tparam OutputBeginIteratorT
//!   Random-access input iterator type for the beginning offsets of the output segments, or
//!   reuse_input_begin
//!
template <typename InputBeginIteratorT, typename OutputBeginIteratorT = reuse_input_begin>
struct segment_addressing
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
//! @tparam WorkT
//!   Unsigned integer type for logical work positions
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
//!   Specialization of segment_addressing
//!
template <typename PolicyGetter,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename WorkT,
          typename ScanOpT,
          typename InitValueT,
          typename AccumT,
          bool ForceInclusive,
          typename AddressingT>
struct agent_segmented_scan_load_balanced
{
private:
  using output_begin_iterator_t = decltype(AddressingT::output_begin);

  static constexpr auto policy            = PolicyGetter{}();
  static constexpr int threads_per_block  = policy.threads_per_block;
  static constexpr int items_per_thread   = policy.items_per_thread;
  static constexpr int items_per_tile     = threads_per_block * items_per_thread;
  static constexpr bool has_init          = !::cuda::std::is_same_v<InitValueT, NullType>;
  static constexpr bool is_inclusive      = ForceInclusive || !has_init;
  static constexpr bool has_output_deltas = !::cuda::std::is_same_v<output_begin_iterator_t, reuse_input_begin>;
  // Entry 0 of a tile's delta table is the segment carried into the tile; entry k > 0 is its k-th head.
  static constexpr int delta_table_size = items_per_tile + 1;

  using flag_value_t = KeyValuePair<int, AccumT>;
  using work_t       = WorkT;
  // Wrapping arithmetic: a segment's begin may lie below its work index, so deltas are stored modulo 2^n. Input
  // positions end at a supplied end offset and so fit OffsetT; output positions are output_position_t.
  using delta_t           = ::cuda::std::make_unsigned_t<OffsetT>;
  using output_position_t = load_balanced_output_position_t<OffsetT, WorkT>;
  using output_delta_t    = ::cuda::std::make_unsigned_t<output_position_t>;
  // Whether an output segment may hold the last output position, as the work index records. 64-bit output positions
  // do not reach it.
  static constexpr bool output_may_reach_maximum =
    has_output_deltas && sizeof(output_position_t) < sizeof(::cuda::std::uint64_t);
  using pair_scan_op_t = ScanBySegmentOp<ScanOpT>;
  using input_value_t  = it_value_t<InputIteratorT>;
  using wrapped_input_iterator_t =
    ::cuda::std::conditional_t<::cuda::std::is_pointer_v<InputIteratorT>,
                               CacheModifiedInputIterator<policy.load_modifier, input_value_t, OffsetT>,
                               InputIteratorT>;
  using block_load_t       = BlockLoad<AccumT, threads_per_block, items_per_thread, policy.load_algorithm>;
  using block_scan_t       = BlockScan<flag_value_t, threads_per_block, policy.scan_algorithm>;
  using block_store_t      = BlockStore<AccumT, threads_per_block, items_per_thread, policy.store_algorithm>;
  using block_count_scan_t = BlockScan<int, threads_per_block, policy.scan_algorithm>;
  using block_reduce_t     = BlockReduce<AccumT, threads_per_block, BLOCK_REDUCE_WARP_REDUCTIONS>;
  using scan_tile_state_t  = ReduceByKeyScanTileState<AccumT, int>;
  using prefix_callback_t  = TilePrefixCallbackOp<flag_value_t, pair_scan_op_t, scan_tile_state_t>;

  // Whether the BlockLoad algorithm uses shared memory to load or exchange the data
  static constexpr bool loads_via_smem =
    !(policy.load_algorithm == BLOCK_LOAD_DIRECT || policy.load_algorithm == BLOCK_LOAD_STRIPED
      || policy.load_algorithm == BLOCK_LOAD_VECTORIZE);

  // The look-back runs while no tile is in flight, so its storage shares the tile's.
  union tile_storage_t
  {
    typename block_load_t::TempStorage load;
    typename block_scan_t::TempStorage scan;
    typename block_store_t::TempStorage store;
    typename block_reduce_t::TempStorage reduce;
    typename prefix_callback_t::TempStorage prefix;
  };

  struct tile_deltas_t
  {
    Uninitialized<delta_t[delta_table_size]> input;
    Uninitialized<output_delta_t[has_output_deltas ? delta_table_size : 1]> output;
    typename block_count_scan_t::TempStorage count_scan;
    delta_t carried_input;
    output_delta_t carried_output;
    delta_t uniform_input;
    output_delta_t uniform_output;
    bool input_is_uniform;
    bool output_is_uniform;
    bool input_is_contiguous;
    bool output_is_contiguous;
    bool output_reaches_maximum;
  };

  // Entry 0 holds the look-back's result and, with one entry, the inclusive pair of the tile's last valid item. With
  // one entry per thread, entry t holds the inclusive pair of thread t's last valid item.
  template <int TailCount>
  struct storage_t
  {
    tile_storage_t tile;
    Uninitialized<unsigned char[items_per_tile]> head_flags;
    Uninitialized<flag_value_t[TailCount]> tails;
    tile_deltas_t deltas;
    OffsetT segment_cursor;
    work_t next_head;
  };

  // An exclusive scan keeps one tail per thread, from which each thread's first output follows, when the kernel's
  // shared memory holds them. Otherwise it scans exclusively in the block and keeps one tail.
  static constexpr bool per_thread_tails =
    !is_inclusive
    && sizeof(storage_t<threads_per_block>) + sizeof(boundary_record<OffsetT, work_t>) + sizeof(range_split<work_t>)
         <= max_smem_per_block;

  using _TempStorage = storage_t<per_thread_tails ? threads_per_block : 1>;

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

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static output_delta_t make_output_delta(OffsetT begin, work_t work)
  {
    return static_cast<output_delta_t>(static_cast<output_position_t>(begin)) - static_cast<output_delta_t>(work);
  }

  // The position of work under delta, of type PositionT, whose unsigned type is DeltaT.
  template <typename PositionT, typename DeltaT>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE static PositionT position(work_t work, DeltaT delta)
  {
    return static_cast<PositionT>(static_cast<DeltaT>(work) + delta);
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

  _CCCL_DEVICE _CCCL_FORCEINLINE void
  initialize_cursor([[maybe_unused]] work_t range_begin, [[maybe_unused]] OffsetT num_segments, OffsetT first_segment)
  {
    auto& shared_head_flags = temp_storage.head_flags.Alias();
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < items_per_thread; ++item)
    {
      shared_head_flags[static_cast<int>(threadIdx.x) * items_per_thread + item] = 0;
    }
    if (threadIdx.x == 0)
    {
      _CCCL_ASSERT(first_segment < num_segments, "the first segment of a range exists");
      const work_t head = offsets[first_segment];
      _CCCL_ASSERT(head <= range_begin, "a range begins in its first segment");

      auto& deltas                = temp_storage.deltas;
      temp_storage.segment_cursor = first_segment;
      temp_storage.next_head      = head;
      if (!deltas.input_is_contiguous)
      {
        deltas.carried_input = make_delta(static_cast<OffsetT>(addressing.input_begin[first_segment]), head);
      }
      if constexpr (has_output_deltas)
      {
        if (!deltas.output_is_contiguous)
        {
          deltas.carried_output = make_output_delta(static_cast<OffsetT>(addressing.output_begin[first_segment]), head);
        }
      }
    }
    __syncthreads();
  }

  // Run by the first warp. Flags the head of every nonempty segment in [tile_begin, tile_end) and numbers the heads
  // from 1 in segment order, recording each head's deltas under its number where asked. Each round reads the work
  // index of one segment per lane. Returns the number of heads to every lane.
  _CCCL_DEVICE _CCCL_FORCEINLINE int
  discover_heads(work_t tile_begin, work_t tile_end, OffsetT num_segments, bool records_input, bool records_output)
  {
    constexpr unsigned int warp_mask   = 0xffffffffu;
    constexpr work_t past_last_segment = ::cuda::std::numeric_limits<work_t>::max();
    if (temp_storage.next_head >= tile_end)
    {
      return 0;
    }

    const int lane                  = static_cast<int>(threadIdx.x);
    const unsigned int lanes_before = (1u << lane) - 1u;
    auto& head_flags                = temp_storage.head_flags.Alias();
    auto& deltas                    = temp_storage.deltas;
    OffsetT batch                   = temp_storage.segment_cursor;
    int entry_count                 = 0;
    while (true)
    {
      const bool valid      = static_cast<OffsetT>(lane) < num_segments - batch;
      const OffsetT segment = valid ? batch + static_cast<OffsetT>(lane) : batch;
      work_t head           = past_last_segment;
      work_t next           = past_last_segment;
      OffsetT input_begin{};
      [[maybe_unused]] OffsetT output_begin{};
      if (valid)
      {
        head = offsets[segment];
        next = offsets[segment + OffsetT{1}];
        // Read before it is known whether the segment's head lies in the tile.
        if (records_input)
        {
          input_begin = static_cast<OffsetT>(addressing.input_begin[segment]);
        }
        if constexpr (has_output_deltas)
        {
          if (records_output)
          {
            output_begin = static_cast<OffsetT>(addressing.output_begin[segment]);
          }
        }
      }

      const bool before_end    = head < tile_end;
      const bool is_head       = before_end && head >= tile_begin && next != head;
      const unsigned int heads = __ballot_sync(warp_mask, is_head);
      if (is_head)
      {
        const int entry                                 = entry_count + __popc(heads & lanes_before) + 1;
        head_flags[static_cast<int>(head - tile_begin)] = 1;
        if (records_input)
        {
          deltas.input.Alias()[entry] = make_delta(input_begin, head);
        }
        if constexpr (has_output_deltas)
        {
          if (records_output)
          {
            deltas.output.Alias()[entry] = make_output_delta(output_begin, head);
          }
        }
      }
      entry_count += __popc(heads);

      // Heads do not decrease, so the lanes before the tile's end are a prefix of the warp.
      const unsigned int lanes_before_end = __ballot_sync(warp_mask, before_end);
      if (lanes_before_end != warp_mask)
      {
        const int stop         = __popc(lanes_before_end);
        const work_t stop_head = ShuffleIndex<detail::warp_threads>(head, stop, warp_mask);
        if (lane == 0)
        {
          temp_storage.segment_cursor = batch + static_cast<OffsetT>(stop);
          temp_storage.next_head      = stop_head;
        }
        return entry_count;
      }

      batch += static_cast<OffsetT>(detail::warp_threads);
      const work_t run_head = ShuffleIndex<detail::warp_threads>(head, 0, warp_mask);
      if (run_head == ShuffleIndex<detail::warp_threads>(next, detail::warp_threads - 1, warp_mask))
      {
        // The batch lies in a run of empty segments: skip to the run's last segment.
        const OffsetT remaining = num_segments - batch + OffsetT{1};
        batch += warp_kary_upper_bound_offset(offsets + batch, remaining, run_head) - OffsetT{1};
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void finish_tile_deltas(int entry_count, work_t tile_begin, work_t tile_end)
  {
    const int lane        = static_cast<int>(threadIdx.x);
    auto& deltas          = temp_storage.deltas;
    const int first_entry = (entry_count > 0 && temp_storage.head_flags.Alias()[0] != 0) ? 1 : 0;

    const auto check = [&](const auto* table, auto& uniform_delta, bool& is_uniform, auto& carried) {
      const auto reference = table[first_entry];
      bool mismatch        = false;
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
    if (!deltas.input_is_contiguous)
    {
      check(deltas.input.Alias(), deltas.uniform_input, deltas.input_is_uniform, deltas.carried_input);
    }
    if constexpr (has_output_deltas)
    {
      if (!deltas.output_is_contiguous)
      {
        check(deltas.output.Alias(), deltas.uniform_output, deltas.output_is_uniform, deltas.carried_output);
        if constexpr (output_may_reach_maximum)
        {
          // Output deltas are uniform modulo 2^n for n-bit positions, but iterator addition does not wrap: a tile
          // whose output positions pass the last one is stored item by item, each from its own position.
          if (lane == 0 && deltas.output_reaches_maximum && deltas.output_is_uniform)
          {
            const output_delta_t positions_after_start =
              static_cast<output_delta_t>(::cuda::std::numeric_limits<output_position_t>::max())
              - static_cast<output_delta_t>(position<output_position_t>(tile_begin, deltas.uniform_output));
            deltas.output_is_uniform = positions_after_start >= static_cast<output_delta_t>(tile_end - tile_begin - 1);
          }
        }
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void
  discover_noncontiguous_heads(work_t tile_begin, work_t tile_end, OffsetT num_segments)
  {
    if (threadIdx.x >= detail::warp_threads)
    {
      return;
    }
    auto& deltas              = temp_storage.deltas;
    const bool records_input  = !deltas.input_is_contiguous;
    const bool records_output = has_output_deltas && !deltas.output_is_contiguous;
    if (threadIdx.x == 0)
    {
      if (records_input)
      {
        deltas.input.Alias()[0] = deltas.carried_input;
      }
      if constexpr (has_output_deltas)
      {
        if (records_output)
        {
          deltas.output.Alias()[0] = deltas.carried_output;
        }
      }
    }
    const int entry_count = discover_heads(tile_begin, tile_end, num_segments, records_input, records_output);
    __syncwarp();
    finish_tile_deltas(entry_count, tile_begin, tile_end);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void load_noncontiguous_tile(
    work_t tile_begin,
    work_t tile_end,
    OffsetT num_segments,
    flag_value_t (&items)[items_per_thread],
    int (&entries)[items_per_thread])
  {
    const int valid_items = static_cast<int>(tile_end - tile_begin);
    discover_noncontiguous_heads(tile_begin, tile_end, num_segments);
    __syncthreads();

    auto& shared_head_flags = temp_storage.head_flags.Alias();
    int head_counts[items_per_thread];
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int item = 0; item < items_per_thread; ++item)
    {
      const int local_index          = static_cast<int>(threadIdx.x) * items_per_thread + item;
      head_counts[item]              = shared_head_flags[local_index];
      shared_head_flags[local_index] = 0;
    }

    auto& deltas             = temp_storage.deltas;
    const delta_t tile_delta = deltas.uniform_input;
    const bool gather        = !deltas.input_is_uniform;
    const bool needs_entries = gather || (has_output_deltas && !deltas.output_is_uniform);
    const OffsetT tile_start = position<OffsetT>(tile_begin, tile_delta);
    AccumT values[items_per_thread];
    if (needs_entries)
    {
      block_count_scan_t(deltas.count_scan).InclusiveSum(head_counts, entries);
    }
    if (gather)
    {
      const delta_t* input_deltas = deltas.input.Alias();
      // Pads the items past the end of a partial tile.
      AccumT filler{};
      if ((static_cast<int>(threadIdx.x) + 1) * items_per_thread > valid_items)
      {
        filler = d_in[tile_start];
      }
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = 0; item < items_per_thread; ++item)
      {
        const int local_index = static_cast<int>(threadIdx.x) * items_per_thread + item;
        values[item] =
          local_index < valid_items
            ? static_cast<AccumT>(
                d_in[position<OffsetT>(tile_begin + static_cast<work_t>(local_index), input_deltas[entries[item]])])
            : filler;
      }
    }
    else if (valid_items == items_per_tile)
    {
      block_load_t(temp_storage.tile.load).Load(d_in + tile_start, values);
    }
    else
    {
      block_load_t(temp_storage.tile.load).Load(d_in + tile_start, values, valid_items, d_in[tile_start]);
    }
    // A gathered tile uses no tile storage.
    if (loads_via_smem && !gather)
    {
      __syncthreads();
    }

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
    auto& deltas = temp_storage.deltas;
    if constexpr (has_output_deltas)
    {
      store_items<output_position_t>(
        tile_begin,
        valid_items,
        output_items,
        entries,
        deltas.output_is_uniform,
        deltas.uniform_output,
        deltas.output.Alias());
    }
    else
    {
      store_items<OffsetT>(
        tile_begin,
        valid_items,
        output_items,
        entries,
        deltas.input_is_uniform,
        deltas.uniform_input,
        deltas.input.Alias());
    }
  }

  template <typename PositionT, typename DeltaT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void store_items(
    work_t tile_begin,
    int valid_items,
    AccumT (&output_items)[items_per_thread],
    const int (&entries)[items_per_thread],
    bool is_uniform,
    DeltaT delta,
    const DeltaT* table)
  {
    if (!is_uniform)
    {
      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = 0; item < items_per_thread; ++item)
      {
        const int local_index = static_cast<int>(threadIdx.x) * items_per_thread + item;
        if (local_index < valid_items)
        {
          d_out[position<PositionT>(tile_begin + static_cast<work_t>(local_index), table[entries[item]])] =
            output_items[item];
        }
      }
      return;
    }
    const PositionT tile_start = position<PositionT>(tile_begin, delta);
    if (valid_items == items_per_tile)
    {
      block_store_t(temp_storage.tile.store).Store(d_out + tile_start, output_items);
    }
    else
    {
      block_store_t(temp_storage.tile.store).Store(d_out + tile_start, output_items, valid_items);
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void load_tile(
    work_t tile_begin,
    work_t tile_end,
    OffsetT num_segments,
    flag_value_t (&items)[items_per_thread],
    int (&entries)[items_per_thread])
  {
    const auto& deltas = temp_storage.deltas;
    if (deltas.input_is_contiguous && (!has_output_deltas || deltas.output_is_contiguous))
    {
      load_contiguous_tile(tile_begin, tile_end, num_segments, items, deltas.uniform_input);
      return;
    }
    load_noncontiguous_tile(tile_begin, tile_end, num_segments, items, entries);
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void load_contiguous_tile(
    work_t tile_begin, work_t tile_end, OffsetT num_segments, flag_value_t (&items)[items_per_thread], delta_t delta)
  {
    const int valid_items    = static_cast<int>(tile_end - tile_begin);
    const OffsetT tile_start = position<OffsetT>(tile_begin, delta);
    AccumT values[items_per_thread];

    if (valid_items == items_per_tile)
    {
      block_load_t(temp_storage.tile.load).Load(d_in + tile_start, values);
    }
    else
    {
      block_load_t(temp_storage.tile.load).Load(d_in + tile_start, values, valid_items, d_in[tile_start]);
    }

    if (threadIdx.x < detail::warp_threads)
    {
      static_cast<void>(discover_heads(tile_begin, tile_end, num_segments, false, false));
    }
    __syncthreads();

    auto& shared_head_flags = temp_storage.head_flags.Alias();

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

  // Reduces a range that lies inside first_segment and does not begin at its head. Only thread 0 holds the result.
  _CCCL_DEVICE _CCCL_FORCEINLINE flag_value_t reduce_range(work_t range_begin, work_t range_end, OffsetT first_segment)
  {
    const delta_t delta =
      make_delta(static_cast<OffsetT>(addressing.input_begin[first_segment]), offsets[first_segment]);

    AccumT range_aggregate{};
    for (work_t tile_begin = range_begin; tile_begin < range_end;)
    {
      const work_t tile_size   = (::cuda::std::min) (range_end - tile_begin, static_cast<work_t>(items_per_tile));
      const work_t tile_end    = tile_begin + tile_size;
      const int valid_items    = static_cast<int>(tile_size);
      const OffsetT tile_start = position<OffsetT>(tile_begin, delta);
      AccumT values[items_per_thread];
      AccumT tile_aggregate{};
      if (valid_items == items_per_tile)
      {
        block_load_t(temp_storage.tile.load).Load(d_in + tile_start, values);
        if constexpr (loads_via_smem)
        {
          __syncthreads();
        }

        tile_aggregate = block_reduce_t(temp_storage.tile.reduce).Reduce(values, scan_op);
      }
      else
      {
        block_load_t(temp_storage.tile.load).Load(d_in + tile_start, values, valid_items, d_in[tile_start]);
        if constexpr (loads_via_smem)
        {
          __syncthreads();
        }

        const int thread_items = valid_items - static_cast<int>(threadIdx.x) * items_per_thread;
        AccumT partial         = values[0];
        _CCCL_PRAGMA_UNROLL_FULL()
        for (int item = 1; item < items_per_thread; ++item)
        {
          if (item < thread_items)
          {
            partial = scan_op(partial, values[item]);
          }
        }
        const int valid_threads = (valid_items + items_per_thread - 1) / items_per_thread;
        tile_aggregate          = block_reduce_t(temp_storage.tile.reduce).Reduce(partial, scan_op, valid_threads);
      }

      if (threadIdx.x == 0)
      {
        range_aggregate = tile_begin == range_begin ? tile_aggregate : scan_op(range_aggregate, tile_aggregate);
      }
      __syncthreads();
      tile_begin = tile_end;
    }
    return flag_value_t{0, range_aggregate};
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE flag_value_t
  obtain_prefix(flag_value_t range_aggregate, int tile_idx, scan_tile_state_t& tile_state)
  {
    _CCCL_ASSERT(tile_idx > 0, "the first block begins at a segment head");

    prefix_callback_t prefix_op(tile_state, temp_storage.tile.prefix, pair_scan_op, tile_idx);
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

    prefix_callback_t prefix_op(tile_state, temp_storage.tile.prefix, pair_scan_op, tile_idx);
    if (threadIdx.x < detail::warp_threads)
    {
      using status_word_t       = typename prefix_callback_t::StatusWord;
      using delay_constructor_t = detail::default_delay_constructor_t<flag_value_t>;
      int predecessor_idx       = tile_idx - static_cast<int>(threadIdx.x) - 1;
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
        detail::uninitialized_copy_single(&temp_storage.tails.Alias()[0], exclusive_prefix);
      }
    }
    __syncthreads();
    return temp_storage.tails.Alias()[0];
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
      [[maybe_unused]] bool head_flags[items_per_thread];
      load_tile(tile_begin, tile_end, num_segments, items, entries);

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int item = 0; item < items_per_thread; ++item)
      {
        head_flags[item] = items[item].key != 0;
      }

      // With one tail, the thread holding the tile's last valid item records its inclusive pair for the next tile.
      const int last_thread                = (valid_items - 1) / items_per_thread;
      [[maybe_unused]] const int last_item = valid_items - 1 - last_thread * items_per_thread;
      const bool full_tile                 = valid_items == items_per_tile;
      auto& tails                          = temp_storage.tails.Alias();
      if constexpr (is_inclusive || per_thread_tails)
      {
        flag_value_t ignored_aggregate;
        block_scan_t(temp_storage.tile.scan).InclusiveScan(items, items, pair_scan_op, ignored_aggregate);

        _CCCL_PRAGMA_UNROLL_FULL()
        for (int item = 0; item < items_per_thread; ++item)
        {
          if (items[item].key == 0)
          {
            items[item] = pair_scan_op(running_prefix, items[item]);
          }
        }

        if constexpr (per_thread_tails)
        {
          const int local_begin = static_cast<int>(threadIdx.x) * items_per_thread;
          if (local_begin < valid_items)
          {
            const int thread_items     = valid_items - local_begin;
            const int thread_last_item = (thread_items < items_per_thread ? thread_items : items_per_thread) - 1;
            tails[threadIdx.x]         = full_tile ? items[items_per_thread - 1] : select_item(items, thread_last_item);
          }
        }
        else if (static_cast<int>(threadIdx.x) == last_thread)
        {
          tails[0] = full_tile ? items[items_per_thread - 1] : select_item(items, last_item);
        }
      }
      else
      {
        const flag_value_t tail_input = full_tile ? items[items_per_thread - 1] : select_item(items, last_item);
        // Leaves the first item of thread 0 undefined; it takes running_prefix.
        block_scan_t(temp_storage.tile.scan).ExclusiveScan(items, items, pair_scan_op);

        _CCCL_PRAGMA_UNROLL_FULL()
        for (int item = 0; item < items_per_thread; ++item)
        {
          if (item == 0 && threadIdx.x == 0)
          {
            items[item] = running_prefix;
          }
          else if (items[item].key == 0)
          {
            items[item] = pair_scan_op(running_prefix, items[item]);
          }
        }

        if (static_cast<int>(threadIdx.x) == last_thread)
        {
          tails[0] = pair_scan_op(full_tile ? items[items_per_thread - 1] : select_item(items, last_item), tail_input);
        }
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
          else if constexpr (per_thread_tails)
          {
            output_items[item] =
              head_flags[item]
                ? static_cast<AccumT>(init_value)
                : (item == 0 ? (threadIdx.x == 0 ? running_prefix.value : tails[threadIdx.x - 1].value)
                             : items[item - 1].value);
          }
          else
          {
            output_items[item] = head_flags[item] ? static_cast<AccumT>(init_value) : items[item].value;
          }
        }
      }

      running_prefix = tails[per_thread_tails ? last_thread : 0];
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
    const flag_value_t range_aggregate = reduce_range(range_begin, range_end, first_segment);
    const flag_value_t range_prefix    = obtain_prefix(range_aggregate, tile_idx, tile_state);
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

  // Run by thread 0 once per block, before the first barrier: records the contiguity flags and the uniform deltas
  // they imply, which hold for every range the block scans.
  _CCCL_DEVICE _CCCL_FORCEINLINE void initialize_addressing(OffsetT num_segments)
  {
    // Reads every origin it may need before testing the flags.
    const work_t contiguity         = offsets[num_segments + OffsetT{1}];
    const OffsetT input_origin      = static_cast<OffsetT>(addressing.input_begin[0]);
    const bool input_is_contiguous  = (contiguity & contiguous_input_flag) != 0;
    const bool output_is_contiguous = (contiguity & contiguous_output_flag) != 0;

    auto& deltas                  = temp_storage.deltas;
    deltas.input_is_contiguous    = input_is_contiguous;
    deltas.output_is_contiguous   = output_is_contiguous;
    deltas.output_reaches_maximum = (contiguity & output_reaches_maximum_flag) != 0;
    if (input_is_contiguous)
    {
      deltas.uniform_input    = make_delta(input_origin, work_t{0});
      deltas.input_is_uniform = true;
    }
    if constexpr (has_output_deltas)
    {
      const OffsetT output_origin = static_cast<OffsetT>(addressing.output_begin[0]);
      if (output_is_contiguous)
      {
        deltas.uniform_output    = make_output_delta(output_origin, work_t{0});
        deltas.output_is_uniform = true;
      }
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void consume_suffix_first_range(
    const boundary_record<OffsetT, work_t>& range,
    work_t range_end,
    OffsetT num_segments,
    int tile_idx,
    scan_tile_state_t& tile_state)
  {
    const work_t range_begin    = range.offset;
    const OffsetT first_segment = range.first_segment;
    _CCCL_ASSERT(range_begin < range_end, "load-balanced ranges must not be empty");

    const bool has_independent_suffix = range.is_head || range.tail_begin > range_begin;
    if (!has_independent_suffix)
    {
      consume_forward_range(range_begin, range_end, num_segments, first_segment, tile_idx, tile_state);
      return;
    }

    const flag_value_t suffix_aggregate =
      scan_range(range.tail_begin, range_end, num_segments, flag_value_t{}, range.tail_segment);
    if (threadIdx.x == 0)
    {
      tile_state.SetInclusive(tile_idx, suffix_aggregate);
    }

    if (range.tail_begin == range_begin)
    {
      return;
    }

    const flag_value_t range_prefix = range.is_head ? flag_value_t{} : obtain_predecessor_prefix(tile_idx, tile_state);
    static_cast<void>(scan_range(range_begin, range.tail_begin, num_segments, range_prefix, first_segment));
  }
};

// Static shared memory of device_segmented_scan_load_balanced_kernel under the policy that PolicyGetter returns
template <typename PolicyGetter,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename WorkT,
          typename ScanOpT,
          typename InitValueT,
          typename AccumT,
          bool ForceInclusive,
          typename AddressingT>
inline constexpr ::cuda::std::size_t load_balanced_kernel_smem_bytes =
  sizeof(typename agent_segmented_scan_load_balanced<
         PolicyGetter,
         InputIteratorT,
         OutputIteratorT,
         OffsetT,
         WorkT,
         ScanOpT,
         InitValueT,
         AccumT,
         ForceInclusive,
         AddressingT>::TempStorage)
  + sizeof(boundary_record<OffsetT, WorkT>) + sizeof(range_split<WorkT>);

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
//! @tparam WorkT
//!   Unsigned integer type for logical work positions
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
//!   Specialization of segment_addressing
//!
template <typename LoadBalancedPolicySelector,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename WorkT,
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
    const WorkT* work_index,
    OffsetT num_segments,
    ScanOpT scan_op,
    InitValueT init_value,
    int resident_blocks,
    int capacity,
    ReduceByKeyScanTileState<AccumT, int> tile_state,
    AddressingT addressing)
{
  using work_t                 = WorkT;
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
    WorkT,
    ScanOpT,
    ActualInitValueT,
    AccumT,
    ForceInclusive,
    AddressingT>;

  __shared__ typename agent_t::TempStorage temp_storage;
  __shared__ boundary_record<OffsetT, work_t> begin_record;
  __shared__ range_split<work_t> split;
  static_assert(
    load_balanced_kernel_smem_bytes<
      policy_getter,
      InputIteratorT,
      OutputIteratorT,
      OffsetT,
      WorkT,
      ScanOpT,
      ActualInitValueT,
      AccumT,
      ForceInclusive,
      AddressingT>
      <= max_smem_per_block,
    "The load-balanced segmented scan's shared memory exceeds max_smem_per_block: reduce the accumulator "
    "size, the policy's tile or the offset width");

  _CCCL_ASSERT(num_segments < ::cuda::std::numeric_limits<OffsetT>::max(), "the work index is indexed by OffsetT");

  const ActualInitValueT actual_init_value = init_value;
  agent_t agent(temp_storage, d_in, d_out, work_index, scan_op, actual_init_value, addressing);

  constexpr int tile_size = policy.threads_per_block * policy.items_per_thread;
  if (threadIdx.x == 0)
  {
    agent.initialize_addressing(num_segments);
    split.total = work_index[num_segments];
    split.virtual_blocks =
      (::cuda::std::min) (load_balanced_block_count(split.total, static_cast<work_t>(tile_size), resident_blocks),
                          capacity);
  }

  const int block = static_cast<int>(blockIdx.x);
  if (threadIdx.x < detail::warp_threads)
  {
    __syncwarp();
    if (block < split.virtual_blocks)
    {
      const work_t begin = even_split_boundary(split.total, split.virtual_blocks, block);
      const work_t end   = even_split_boundary(split.total, split.virtual_blocks, block + 1);
      const auto record  = make_even_boundary_record(work_index, num_segments, begin, end);
      if (threadIdx.x == 0)
      {
        begin_record = record;
      }
    }
  }
  __syncthreads();

  const int virtual_blocks = split.virtual_blocks;
  if (block >= virtual_blocks)
  {
    return;
  }

  const boundary_record<OffsetT, work_t> begin = begin_record;
  const work_t end                             = even_split_boundary(split.total, virtual_blocks, block + 1);
  agent.consume_suffix_first_range(begin, end, num_segments, block, tile_state);
}
} // namespace detail::segmented_scan

CUB_NAMESPACE_END
