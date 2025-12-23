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

#include <cub/agent/agent_segmented_topk.cuh>
#include <cub/detail/segmented_params.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_topk.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_temporary_storage.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_topk
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
struct k_uniform : public params::uniform_param<::cuda::std::int64_t, MinK, MaxK>
{};

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
template <::cuda::std::int64_t MaxNumItems  = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max(),
          ::cuda::std::int64_t MinNumItemsT = 1>
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
// One-worker-per-segment policy selection
// -----------------------------------------------------------------------------
template <typename PoliciesT,
          ::cuda::std::int64_t Index,
          ::cuda::std::int64_t Count,
          template <typename...> class WorkerPerSegmentAgentT,
          typename... AgentParamsT>
struct find_valid_policy_impl;

// Base case: End of policy chain reached: If we reach Index == Count, it means we checked all with no match
template <typename PoliciesT,
          ::cuda::std::int64_t Count,
          template <typename...> class WorkerPerSegmentAgentT,
          typename... AgentParamsT>
struct find_valid_policy_impl<PoliciesT, Count, Count, WorkerPerSegmentAgentT, AgentParamsT...>
{
  using policy_t                         = void;
  static constexpr bool has_valid_policy = false;
};

template <typename PoliciesT,
          ::cuda::std::int64_t Index,
          ::cuda::std::int64_t Count,
          template <typename...> class WorkerPerSegmentAgentT,
          typename... AgentParamsT>
struct find_valid_policy_impl
{
  // Inspect the current policy
  using current_policy_t = ::cuda::std::tuple_element_t<Index, PoliciesT>;

  // Instantiate agent to check temporary storage size
  using current_agent_t      = WorkerPerSegmentAgentT<current_policy_t, AgentParamsT...>;
  static constexpr bool fits = (sizeof(typename current_agent_t::TempStorage) <= 48 * 1024);

  // The 'next' policy in the chain
  using next_step = find_valid_policy_impl<PoliciesT, Index + 1, Count, WorkerPerSegmentAgentT, AgentParamsT...>;

  // Select result:
  // If 'fits' is true, we stop here.
  // If 'fits' is false, we take the result from 'next_step'.
  using policy_t = ::cuda::std::conditional_t<fits, current_policy_t, typename next_step::policy_t>;

  // Whether there's a valid policy that we can instantiate the agent with such that the agent's shared memory doesn't
  // exceed the static shared memory limimt
  static constexpr bool has_valid_policy = fits ? true : next_step::has_valid_policy;
};

template <typename SegmentedTopKPolicy, template <typename...> class WorkerPerSegmentAgentT, typename... AgentParamsT>
struct find_valid_policy
{
  // The list of policies for the one-worker-per-segment approach
  using worker_per_segment_policies = typename SegmentedTopKPolicy::worker_per_segment_policies;

  // Helper to find a valid policy that we can successfully instantiate the agent with
  using find_valid_policy_impl_t =
    find_valid_policy_impl<worker_per_segment_policies,
                           0,
                           ::cuda::std::tuple_size<worker_per_segment_policies>::value,
                           WorkerPerSegmentAgentT,
                           AgentParamsT...>;

  // Whether there's a valid policy for one-worker-per-segment approach
  static constexpr bool supports_one_worker_per_segment = find_valid_policy_impl_t::has_valid_policy;

  // Policy selected for one-worker-per-segment approach, if there is a valid policy
  using worker_per_segment_policy_t = typename find_valid_policy_impl_t::policy_t;

  // Agent for the one-worker-per-segment approach, if there is a valid policy
  using worker_per_segment_agent_t =
    ::cuda::std::conditional_t<supports_one_worker_per_segment,
                               WorkerPerSegmentAgentT<worker_per_segment_policy_t, AgentParamsT...>,
                               void>;
};

// -----------------------------------------------------------------------------
// Global Kernel Entry Point
// -----------------------------------------------------------------------------
template <typename ChainedPolicyT,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT>
__launch_bounds__(int()) __global__ void DeviceSegmentedTopKKernel(
  KeyInputItItT d_key_segments_it,
  KeyOutputItItT d_key_segments_out_it,
  ValueInputItItT d_value_segments_it,
  ValueOutputItItT d_value_segments_out_it,
  SegmentSizeParameterT segment_sizes,
  KParameterT k,
  SelectDirectionParameterT select_directions,
  NumSegmentsParameterT num_segments)
{
  using active_policy_t = typename ChainedPolicyT::ActivePolicy;

  using find_valid_policy_t = find_valid_policy<
    active_policy_t,
    AgentSegmentedTopkWorkerPerSegment,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    SegmentSizeParameterT,
    KParameterT,
    SelectDirectionParameterT,
    NumSegmentsParameterT>;

  using agent_t = typename find_valid_policy_t::worker_per_segment_agent_t;

  // Static Assertions (Constraints)
  static_assert(agent_t::tile_size >= params::static_max_value_v<SegmentSizeParameterT>,
                "Block size exceeds maximum segment size supported by SegmentSizeParameterT");
  static_assert(sizeof(agent_t::TempStorage) <= 48 * 1024,
                "Static shared memory per block must not exceed 48KB limit.");

  // Temporary storage allocation
  __shared__ typename agent_t::TempStorage temp_storage;

  // Instantiate agent
  agent_t agent(
    temp_storage,
    d_key_segments_it,
    d_key_segments_out_it,
    d_value_segments_it,
    d_value_segments_out_it,
    segment_sizes,
    k,
    select_directions,
    num_segments);

  // Process segments
  agent.Process();
}

// -----------------------------------------------------------------------------
// Segmented Top-K Dispatch
// -----------------------------------------------------------------------------
template <typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT,
          typename TotalNumItemsGuaranteeT,
          typename SelectedPolicy = policy_hub<it_value_t<it_value_t<KeyInputItItT>>,
                                               it_value_t<it_value_t<ValueInputItItT>>,
                                               ::cuda::std::int64_t,
                                               params::static_max_value_v<KParameterT>>>
struct DispatchSegmentedTopK
{
  using offset_t = ::cuda::std::int64_t;

  /// Device-accessible allocation of temporary storage.
  /// When `nullptr`, the required allocation size is written to `temp_storage_bytes` and no work is done.
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /// d_key_segments_it[segment_index] -> iterator to the input sequence of key data for segment `segment_index`
  KeyInputItItT d_key_segments_it;

  /// d_key_segments_out_it[segment_index] -> iterator to the output sequence of key data for segment `segment_index`
  KeyOutputItItT d_key_segments_out_it;

  /// d_value_segments_it[segment_index] -> iterator to the input sequence of associated value items for segment
  /// `segment_index`
  ValueInputItItT d_value_segments_it;

  /// d_value_segments_out_it[segment_index] -> iterator to the output sequence of associated value items for segment
  /// `segment_index`
  ValueOutputItItT d_value_segments_out_it;

  /// Parameter providing segment sizes for each segment
  SegmentSizeParameterT segment_sizes;

  /// Parameter providing K for each segment
  KParameterT k;

  /// Parameter providing the selection direction for each segment
  SelectDirectionParameterT select_directions;

  /// Number of segments
  NumSegmentsParameterT num_segments;

  // Allows the user to provide a guarantee on the upper bound of the total number of items
  TotalNumItemsGuaranteeT total_num_items_guarantee;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  // We pass ValueInputItItT itself as cub::NullType** when only keys are processed
  static constexpr bool keys_only = ::cuda::std::is_same_v<ValueInputItItT, NullType**>;

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchSegmentedTopK(
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
    TotalNumItemsGuaranteeT total_num_items_guarantee,
    cudaStream_t stream,
    int ptx_version)
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_key_segments_it(d_key_segments_it)
      , d_key_segments_out_it(d_key_segments_out_it)
      , d_value_segments_it(d_value_segments_it)
      , d_value_segments_out_it(d_value_segments_out_it)
      , segment_sizes(segment_sizes)
      , k(k)
      , select_directions(select_directions)
      , num_segments(num_segments)
      , total_num_items_guarantee(total_num_items_guarantee)
      , stream(stream)
      , ptx_version(ptx_version)
  {}

  template <typename ActiveWorkerPerSegmentPolicyTPolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t InvokeFixedSegmentSize()
  {
    using max_policy_t = typename SelectedPolicy::max_policy;

    // Currently, only uniform segment sizes are supported
    static_assert(!params::is_per_segment_param_v<SegmentSizeParameterT>,
                  "Only uniform segment sizes are currently supported.");

    // Instantiate the kernel with the selected policy and check shared memory requirements
    using topk_policy_t = ActiveWorkerPerSegmentPolicyTPolicyT;

    constexpr int block_dim = topk_policy_t::BLOCK_THREADS;

    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    // TODO (elstehle): support number of segments provided by device-accessible iterator
    // Only uniform number of segments are supported (i.e., we  need to resolve the number of segments on the host)
    static_assert(!params::is_per_segment_param_v<NumSegmentsParameterT>,
                  "Only uniform segment sizes are currently supported.");
    int grid_dim = resolve_param(num_segments, 0);

    THRUST_NS_QUALIFIER::cuda_cub::detail::triple_chevron(grid_dim, block_dim, 0, stream)
      .doit(
        DeviceSegmentedTopKKernel<max_policy_t,
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
        num_segments);

    // Check for failure to launch
    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }

    return cudaSuccess;
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    using max_policy_t = typename SelectedPolicy::max_policy;

    // Helper that determines (a) whether there's any one-worker-per-segment policy supporting the range of segment
    // sizes and k, and (b) if so, which set of one-worker-per-segment policies to use
    using find_valid_policy_t = find_valid_policy<
      ActivePolicyT,
      AgentSegmentedTopkWorkerPerSegment,
      KeyInputItItT,
      KeyOutputItItT,
      ValueInputItItT,
      ValueOutputItItT,
      SegmentSizeParameterT,
      KParameterT,
      SelectDirectionParameterT,
      NumSegmentsParameterT>;

    // Currently, we only support fixed-size segments that fit into shared memory
    // TODO (elstehle): extend support for variable-size segments
    if constexpr (!params::is_per_segment_param_v<SegmentSizeParameterT>
                  && find_valid_policy_t::supports_one_worker_per_segment)
    {
      return InvokeFixedSegmentSize<typename find_valid_policy_t::worker_per_segment_policy_t>();
    }
    else
    {
      return cudaErrorNotSupported;
    }
  }

  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
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
    TotalNumItemsGuaranteeT total_num_items_guarantee,
    cudaStream_t stream)
  {
    using max_policy_t = typename SelectedPolicy::max_policy;

    int ptx_version = 0;
    if (cudaError_t error = CubDebug(PtxVersion(ptx_version)))
    {
      return error;
    }

    DispatchSegmentedTopK dispatch{
      d_temp_storage,
      temp_storage_bytes,
      d_key_segments_it,
      d_key_segments_out_it,
      d_value_segments_it,
      d_value_segments_out_it,
      segment_sizes,
      k,
      select_directions,
      num_segments,
      total_num_items_guarantee,
      stream,
      ptx_version};

    return CubDebug(max_policy_t::Invoke(ptx_version, dispatch));
  }
};
} // namespace detail::segmented_topk

CUB_NAMESPACE_END
