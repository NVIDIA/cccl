// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved. SPDX-License-Identifier:
// Apache-2.0 WITH LLVM-exception

//! @file
//! cub::DeviceTopK provides device-wide, parallel operations for finding the K largest (or smallest) items from
//! sequences of unordered data items residing within device-accessible memory.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/choose_offset.cuh>
#include <cub/detail/segmented_params.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/device/dispatch/kernels/kernel_batched_topk.cuh>
#include <cub/device/dispatch/tuning/tuning_batched_topk.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/util_type.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__iterator/counting_iterator.h>
#include <cuda/__iterator/transform_iterator.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

namespace detail::batched_topk
{
// -----------------------------------------------------------------------------
// Segmented Top-K-Specific Parameter Types
// -----------------------------------------------------------------------------

// ------------ SELECTION DIRECTION PARAMETER TYPES ------------

// Selection direction known at compile time, same value applies to all segments
template <detail::topk::select SelectDirection>
using select_direction_static = params::uniform_discrete_param<detail::topk::select, SelectDirection>;

// Selection direction is a runtime value, same value applies to all segments
using select_direction_uniform =
  params::uniform_discrete_param<detail::topk::select, detail::topk::select::max, detail::topk::select::min>;

// Per-segment selection direction via iterator
template <typename SelectionDirectionIt, detail::topk::select... SelectDirectionOptions>
using select_direction_per_segment =
  params::per_segment_discrete_param<SelectionDirectionIt, detail::topk::select, SelectDirectionOptions...>;

// ------------ SEGMENT SIZE PARAMETER TYPES ------------

// Segment size known at compile time, same value applies to all segments
template <::cuda::std::int64_t SegmentSize>
using segment_size_static = params::static_constant_param<::cuda::std::int64_t, SegmentSize>;

// Segment size is a runtime value, same value applies to all segments
template <::cuda::std::int64_t MinSegmentSize = 0,
          ::cuda::std::int64_t MaxSegmentSize = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max()>
using segment_size_uniform = params::uniform_param<::cuda::std::int64_t, MinSegmentSize, MaxSegmentSize>;

// Segment size via iterator
template <typename SegmentSizesItT,
          ::cuda::std::int64_t MinSegmentSize = 1,
          ::cuda::std::int64_t MaxSegmentSize = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max()>
using segment_size_per_segment =
  params::per_segment_param<SegmentSizesItT, ::cuda::std::int64_t, MinSegmentSize, MaxSegmentSize>;

// ------------ K PARAMETER TYPES ------------

// K known at compile time, same value applies to all segments
template <::cuda::std::int64_t K>
using k_static = params::static_constant_param<::cuda::std::int64_t, K>;

// K is a runtime value, same value applies to all segments
template <::cuda::std::int64_t MinK = 1,
          ::cuda::std::int64_t MaxK = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max()>
using k_uniform = params::uniform_param<::cuda::std::int64_t, MinK, MaxK>;

// K via iterator
template <typename KItT,
          ::cuda::std::int64_t MinK = 1,
          ::cuda::std::int64_t MaxK = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max()>
using k_per_segment = params::per_segment_param<KItT, ::cuda::std::int64_t, MinK, MaxK>;

// ------------ TOTAL NUMBER OF SEGMENTS ------------
// Number of segments known at compile time
template <::cuda::std::int64_t StaticNumSegments>
using num_segments_static = params::static_constant_param<::cuda::std::int64_t, StaticNumSegments>;

// Number of segments is a runtime value
template <::cuda::std::int64_t MinNumSegments = 1,
          ::cuda::std::int64_t MaxNumSegments = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max()>
using num_segments_uniform = params::uniform_param<::cuda::std::int64_t, MinNumSegments, MaxNumSegments>;

// Number of segments via iterator
template <typename NumSegmentsItT,
          ::cuda::std::int64_t MinNumSegments = 1,
          ::cuda::std::int64_t MaxNumSegments = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max()>
using num_segments_indirect =
  params::per_segment_param<NumSegmentsItT, ::cuda::std::int64_t, MinNumSegments, MaxNumSegments>;

// ------------ TOTAL NUMBER OF ITEMS PARAMETER TYPES ------------

// Number of items guarantee
template <::cuda::std::int64_t MinNumItems = 1,
          ::cuda::std::int64_t MaxNumItems = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max()>
struct total_num_items_guarantee
{
  using value_type                                 = ::cuda::std::int64_t;
  static constexpr value_type static_min_num_items = MinNumItems;
  static constexpr value_type static_max_num_items = MaxNumItems;

  value_type min_num_items = MinNumItems;
  value_type max_num_items = MaxNumItems;

  // Create default ctor, 1 param ctor taking min, 2 param ctor taking min/max
  total_num_items_guarantee() = default;

  _CCCL_HOST_DEVICE total_num_items_guarantee(value_type num_items)
      : min_num_items(num_items)
      , max_num_items(num_items)
  {}

  _CCCL_HOST_DEVICE total_num_items_guarantee(value_type min_items, value_type max_items)
      : min_num_items(min_items)
      , max_num_items(max_items)
  {}
};

// -----------------------------------------------------------------------------
// Helper: turn a segment ID into the number of large-segment-agent tiles needed
// to cover that segment. Wrapped in a transform_iterator, this produces the
// per-segment tile counts that we exclusive-scan to obtain per-segment tile
// offsets.
// -----------------------------------------------------------------------------
template <class SegmentSizeParameterT, class TotalNumItemsValueType>
struct segment_size_to_tile_count_op
{
  SegmentSizeParameterT segment_sizes;
  int large_segment_agent_tile_size;

  template <typename SegmentIndexT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr TotalNumItemsValueType operator()(SegmentIndexT segment_id) const
  {
    return static_cast<TotalNumItemsValueType>(
      ::cuda::ceil_div(segment_sizes.get_param(segment_id), large_segment_agent_tile_size));
  }
};

// -----------------------------------------------------------------------------
// Segmented Top-K Dispatch
// -----------------------------------------------------------------------------

//! @param d_temp_storage Device-accessible allocation of temporary storage. When `nullptr`, the required allocation
//!        size is written to `temp_storage_bytes` and no work is done.
//! @param temp_storage_bytes Reference to size in bytes of `d_temp_storage` allocation
//! @param d_key_segments_it d_key_segments_it[segment_index] -> iterator to the input sequence of key data for segment
//!        `segment_index`
//! @param d_key_segments_out_it d_key_segments_out_it[segment_index] -> iterator to the output sequence of key data for
//!        segment `segment_index`
//! @param d_value_segments_it d_value_segments_it[segment_index] -> iterator to the input sequence of associated value
//!        items for segment `segment_index`. When cub::NullType**, only keys are provided.
//! @param d_value_segments_out_it d_value_segments_out_it[segment_index] -> iterator to the output sequence of
//!        associated value items for segment `segment_index`
//! @param segment_sizes Parameter providing segment sizes for each segment
//! @param k Parameter providing K for each segment
//! @param select_directions Parameter providing the selection direction for each segment
//! @param num_segments Number of segments
//! @param total_num_items_guarantee Allows the user to provide a guarantee on the upper bound of the total number of
//!        items
template <typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT,
          typename TotalNumItemsGuaranteeT,
          typename PolicySelector = policy_selector_from_types<it_value_t<it_value_t<KeyInputItItT>>,
                                                               it_value_t<it_value_t<ValueInputItItT>>,
                                                               ::cuda::std::int64_t,
                                                               params::static_max_value_v<KParameterT>>>
#if _CCCL_HAS_CONCEPTS()
  requires batched_topk_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  ValueInputItItT d_value_segments_it,
  ValueOutputItItT d_value_segments_out_it,
  SegmentSizeParameterT segment_sizes,
  KParameterT k,
  SelectDirectionParameterT select_directions,
  NumSegmentsParameterT num_segments,
  [[maybe_unused]] TotalNumItemsGuaranteeT total_num_items_guarantee,
  cudaStream_t stream                             = nullptr,
  [[maybe_unused]] PolicySelector policy_selector = {})
{
  using large_segment_tile_offset_t = typename TotalNumItemsGuaranteeT::value_type;
  // Helper that determines (a) whether there's any one-worker-per-segment policy supporting the range of segment
  // sizes and k, and (b) if so, which set of one-worker-per-segment policies to use
  constexpr auto policy = find_smallest_covering_policy<
    PolicySelector,
    SegmentSizeParameterT,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    SegmentSizeParameterT,
    KParameterT,
    SelectDirectionParameterT,
    NumSegmentsParameterT,
    large_segment_tile_offset_t>::policy;
  constexpr worker_policy worker_per_segment_policy             = policy.worker_per_segment_policy;
  constexpr multi_worker_policy multi_worker_per_segment_policy = policy.multi_worker_per_segment_policy;

  static constexpr int worker_per_segment_tile_size =
    worker_per_segment_policy.threads_per_block * worker_per_segment_policy.items_per_thread;
  static constexpr bool any_small_segments =
    params::static_min_value_v<SegmentSizeParameterT> <= worker_per_segment_tile_size;
  static constexpr bool only_small_segments =
    params::static_max_value_v<SegmentSizeParameterT> <= worker_per_segment_tile_size;

  // Allocation layout:
  //   only_small_segments: [0] dummy.
  //   any_small_segments && !only_small_segments (mixed): [0] tile offsets, [1] counters struct,
  //                                                       [2] large-segment ids.
  //   !any_small_segments (large-only): [0] tile offsets, [1] segment-size transform-scan temp storage.
  static constexpr int allocations_array_size     = only_small_segments ? 1 : (any_small_segments ? 3 : 2);
  size_t allocation_sizes[allocations_array_size] = {1};

  using num_segments_val_t         = typename NumSegmentsParameterT::value_type;
  using counters_t                 = batched_topk_counters<num_segments_val_t>;
  using segment_size_scan_offset_t = detail::choose_offset_t<num_segments_val_t>;
  using segment_size_scan_input_op_t =
    segment_size_to_tile_count_op<SegmentSizeParameterT, large_segment_tile_offset_t>;
  static constexpr auto multi_worker_per_segment_tile_size =
    multi_worker_per_segment_policy.threads_per_block * multi_worker_per_segment_policy.items_per_thread;
  const segment_size_scan_input_op_t segment_size_scan_input_op{segment_sizes, multi_worker_per_segment_tile_size};
  // Transform iterator over [0, num_segments) producing the tile-count for each segment.
  [[maybe_unused]] const auto segment_size_scan_input_it = ::cuda::transform_iterator(
    ::cuda::counting_iterator<num_segments_val_t>{num_segments_val_t{0}}, segment_size_scan_input_op);

  if constexpr (!only_small_segments)
  {
    const auto num_segments_val = num_segments.get_param(0);
    // Scan output
    allocation_sizes[0] = num_segments_val * sizeof(large_segment_tile_offset_t);
    if constexpr (any_small_segments)
    {
      allocation_sizes[1] = sizeof(counters_t);
      // Large segment ids for indirectly accessing the large segment parameters
      allocation_sizes[2] = num_segments_val * sizeof(num_segments_val_t);
    }
    else
    {
      // Query the temporary storage requirement of the segment-size transform-scan.
      if (const auto error = CubDebug(detail::scan::dispatch(
            nullptr,
            allocation_sizes[1],
            segment_size_scan_input_it,
            static_cast<large_segment_tile_offset_t*>(nullptr),
            ::cuda::std::plus<>{},
            detail::InputValue<large_segment_tile_offset_t>(large_segment_tile_offset_t{0}),
            static_cast<segment_size_scan_offset_t>(num_segments_val),
            stream)))
      {
        return error;
      }
    }
  }

  // Compute allocation pointers into the single storage blob (or compute the necessary size of the blob)
  void* allocations[allocations_array_size] = {};
  if (const auto error =
        CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
  {
    return error;
  }

  if (d_temp_storage == nullptr)
  {
    return cudaSuccess;
  }

  // TODO (elstehle): support number of segments provided by device-accessible iterator
  // Only uniform number of segments are supported (i.e., we need to resolve the number of segments on the host)
  static_assert(!params::is_per_segment_param_v<NumSegmentsParameterT>,
                "Only uniform segment sizes are currently supported.");

  if constexpr (any_small_segments)
  {
    if constexpr (!only_small_segments)
    {
      // Zero-initialize the counters struct that holds the large-segment queue length and the block retirement
      // counter; both are read by the agent's atomic operations and must start at 0.
      if (const auto error = CubDebug(cudaMemsetAsync(allocations[1], 0, sizeof(counters_t), stream)))
      {
        return error;
      }
    }
    const int grid_dim      = static_cast<int>(num_segments.get_param(0));
    constexpr int block_dim = worker_per_segment_policy.threads_per_block;
    if (const auto error = CubDebug(
          THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(grid_dim, block_dim, 0, stream)
            .doit(
              device_segmented_topk_kernel<
                PolicySelector,
                KeyInputItItT,
                KeyOutputItItT,
                ValueInputItItT,
                ValueOutputItItT,
                SegmentSizeParameterT,
                KParameterT,
                SelectDirectionParameterT,
                NumSegmentsParameterT,
                large_segment_tile_offset_t>,
              d_key_segments_it,
              d_key_segments_out_it,
              d_value_segments_it,
              d_value_segments_out_it,
              segment_sizes,
              k,
              select_directions,
              num_segments,
              only_small_segments ? nullptr : static_cast<counters_t*>(allocations[1]),
              only_small_segments ? nullptr : static_cast<num_segments_val_t*>(allocations[2]),
              only_small_segments ? nullptr : static_cast<large_segment_tile_offset_t*>(allocations[0]))))
    {
      return error;
    }
  }
  else
  {
    // No small segments: the small-kernel epilogue (which would otherwise produce the per-segment tile offsets) does
    // not run. Compute the per-segment tile offsets directly via a transform-scan over all segment sizes.
    // The large segment agent will either consume these offsets directly (segment_id -> tile offset) or, when going
    // through the large-segment queue, via a transform iterator over `d_large_segments_ids` (level of indirection).
    if (const auto error = CubDebug(detail::scan::dispatch(
          allocations[1],
          allocation_sizes[1],
          segment_size_scan_input_it,
          static_cast<large_segment_tile_offset_t*>(allocations[0]),
          ::cuda::std::plus<>{},
          detail::InputValue<large_segment_tile_offset_t>(large_segment_tile_offset_t{0}),
          static_cast<segment_size_scan_offset_t>(num_segments.get_param(0)),
          stream)))
    {
      return error;
    }
  }

  if constexpr (!only_small_segments)
  {
    // TODO (elstehle): support larger number of segments through multiple kernel launches
    // Depending on any_small_segments, we need to either:
    // - Indirectly get the large segment parameters via the queued large segment IDs
    // - Directly take the segment parameters since all segments are large
  }
  return CubDebug(detail::DebugSyncStream(stream));
}
} // namespace detail::batched_topk

CUB_NAMESPACE_END
