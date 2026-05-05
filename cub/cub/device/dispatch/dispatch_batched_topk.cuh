// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved. SPDX-License-Identifier:
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

#include <cub/detail/segmented_params.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/kernels/kernel_batched_topk.cuh>
#include <cub/device/dispatch/tuning/tuning_batched_topk.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_temporary_storage.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

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
using num_segments_per_segment =
  params::per_segment_param<NumSegmentsItT, ::cuda::std::int64_t, MinNumSegments, MaxNumSegments>;

// ------------ TOTAL NUMBER OF ITEMS PARAMETER TYPES ------------

// Number of items guarantee
template <::cuda::std::int64_t MinNumItemsT = 1,
          ::cuda::std::int64_t MaxNumItems  = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max()>
struct total_num_items_guarantee
{
  static constexpr ::cuda::std::int64_t static_min_num_items = MinNumItemsT;
  static constexpr ::cuda::std::int64_t static_max_num_items = MaxNumItems;

  ::cuda::std::int64_t min_num_items = MinNumItemsT;
  ::cuda::std::int64_t max_num_items = MaxNumItems;

  // Create default ctor, 1 param ctor taking min, 2 param ctor taking min/max
  total_num_items_guarantee() = default;

  _CCCL_HOST_DEVICE total_num_items_guarantee(::cuda::std::int64_t num_items)
      : min_num_items(num_items)
      , max_num_items(num_items)
  {}

  _CCCL_HOST_DEVICE total_num_items_guarantee(::cuda::std::int64_t min_items, ::cuda::std::int64_t max_items)
      : min_num_items(min_items)
      , max_num_items(max_items)
  {}
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
  // Helper that determines (a) whether there's any one-worker-per-segment policy supporting the range of segment
  // sizes and k, and (b) if so, which set of one-worker-per-segment policies to use
  constexpr worker_policy selected = find_smallest_covering_policy<
    PolicySelector,
    SegmentSizeParameterT,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    SegmentSizeParameterT,
    KParameterT,
    SelectDirectionParameterT,
    NumSegmentsParameterT>::policy;

  if (d_temp_storage == nullptr)
  {
    temp_storage_bytes = 1;
    return cudaSuccess;
  }

  // TODO (elstehle): support number of segments provided by device-accessible iterator
  // Only uniform number of segments are supported (i.e., we need to resolve the number of segments on the host)
  static_assert(!params::is_per_segment_param_v<NumSegmentsParameterT>,
                "Only uniform segment sizes are currently supported.");

  // TODO (elstehle): support larger number of segments through multiple kernel launches
  const int grid_dim      = static_cast<int>(num_segments.get_param(0));
  constexpr int block_dim = selected.block_threads;

  if (const auto error = CubDebug(
        THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(grid_dim, block_dim, 0, stream)
          .doit(device_segmented_topk_kernel<PolicySelector,
                                             KeyInputItItT,
                                             KeyOutputItItT,
                                             ValueInputItItT,
                                             ValueOutputItItT,
                                             SegmentSizeParameterT,
                                             KParameterT,
                                             SelectDirectionParameterT,
                                             NumSegmentsParameterT>,
                d_key_segments_it,
                d_key_segments_out_it,
                d_value_segments_it,
                d_value_segments_out_it,
                segment_sizes,
                k,
                select_directions,
                num_segments)))
  {
    return error;
  }

  return CubDebug(detail::DebugSyncStream(stream));
}
} // namespace detail::batched_topk

CUB_NAMESPACE_END
