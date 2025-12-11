// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved. SPDX-License-Identifier:
// Apache-2.0 WITH LLVM-exception

//! @file
//! cub::DeviceTopK provides device-wide, parallel operations for finding the K largest (or smallest) items from
//! sequences of unordered data items residing within device-accessible memory.

#pragma once

#include <cub/config.cuh>

#include "cub/block/block_store.cuh"
#include "cuda/__iterator/counting_iterator.h"
#include "cuda/std/__limits/numeric_limits.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/agent/agent_segmented_topk.cuh>
#include <cub/agent/agent_topk.cuh>
#include <cub/block/block_radix_sort.cuh>
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

namespace detail::params
{
// -----------------------------------------------------------------------------
// [PARAMETER MIXINS AND HELPERS]
// -----------------------------------------------------------------------------

// Allows providing constrains on parameter values at compile time
template <typename T, T Min = ::cuda::std::numeric_limits<T>::min(), T Max = ::cuda::std::numeric_limits<T>::max()>
struct static_bounds_mixin
{
  static_assert(Min <= Max, "Min must be <= Max");

  static constexpr T static_min_value = Min;
  static constexpr T static_max_value = Max;

  // Indicates that there's only one possible value
  static constexpr bool is_exact = (Min == Max);
};

// Allows specifying a list of supported options for a parameter. E.g., the orders (ascending, descending) that are
// supported by a sorting algorithm.
template <typename T, T... Options>
struct supported_options
{
  static constexpr size_t count = sizeof...(Options);
};

// Helper that translates a runtime parameter value into a compile-time constant by matching against a list of supported
// options.
template <typename T, T... Opts, typename Param, typename Functor>
_CCCL_HOST_DEVICE bool dispatch_impl(Param p, supported_options<T, Opts...>, Functor&& f)
{
  // Fold expression over the supported options.
  // This generates code equivalent to:
  // if (p.value == Opt1) f(integral_constant<Opt1>);
  // else if (p.value == Opt2) f(integral_constant<Opt2>);
  // ...

  bool match_found = ((p.value == Opts ? (f(::cuda::std::integral_constant<T, Opts>{}), true) : false) || ...);

  // Optional: Handling cases where the runtime value was not in the supported
  // list. In a release build, we assume the user respected the contract.
  return match_found;
}

// Dispatcher that matches a runtime parameter value against a list of supported options and invokes a functor with the
// matched option as a compile-time constant.
template <typename ParamT, typename Functor>
_CCCL_HOST_DEVICE void dispatch_discrete(ParamT p, Functor&& f)
{
  using supported_list = typename ParamT::supported_options_t;
  dispatch_impl(p, supported_list{}, ::cuda::std::forward<Functor>(f));
}

// -----------------------------------------------------------------------------
// [FUNDAMENTAL PARAMETER TYPES]
// -----------------------------------------------------------------------------

struct tag_static
{};
struct tag_uniform
{};
struct tag_per_segment
{};

// A compile-time constant
template <typename T, T Value>
struct static_constant_param : public static_bounds_mixin<T, Value, Value>
{
  using value_type = T;
  using param_tag  = tag_static;
};
// -----------------------------------------------------------------------------
// 1. Uniform Param
// -----------------------------------------------------------------------------
// Added default template args so CTAD can deduce T and default Min/Max
template <typename T, T Min = ::cuda::std::numeric_limits<T>::min(), T Max = ::cuda::std::numeric_limits<T>::max()>
struct uniform_param : public static_bounds_mixin<T, Min, Max>
{
  using value_type = T;
  using param_tag  = tag_uniform;

  T value;

  _CCCL_HOST_DEVICE constexpr uniform_param(T v)
      : value(v)
  {}

  uniform_param() = default;
};

// Deduction Guide:
// Allows: uniform_param{5} -> uniform_param<int, INT_MIN, INT_MAX>
template <typename T>
uniform_param(T) -> uniform_param<T>;

// -----------------------------------------------------------------------------
// 2. Per-Segment Param
// -----------------------------------------------------------------------------
// Added defaults for T, Min, and Max based on the Iterator's value_type
template <typename IteratorT,
          typename T = typename ::cuda::std::iterator_traits<IteratorT>::value_type,
          T Min      = ::cuda::std::numeric_limits<T>::min(),
          T Max      = ::cuda::std::numeric_limits<T>::max()>
struct per_segment_param : public static_bounds_mixin<T, Min, Max>
{
  using iterator_type = IteratorT;
  using value_type    = T;
  using param_tag     = tag_per_segment;

  IteratorT iterator;
  T min_value = Min;
  T max_value = Max;

  // Constructor 1: Implicit bounds (from template args)
  _CCCL_HOST_DEVICE constexpr per_segment_param(IteratorT iter)
      : iterator(iter)
  {}

  // Constructor 2: Explicit runtime bounds
  _CCCL_HOST_DEVICE constexpr per_segment_param(IteratorT iter, T min_v, T max_v)
      : iterator(iter)
      , min_value(min_v)
      , max_value(max_v)
  {}

  per_segment_param() = default;
};

// Deduction Guide:
// Allows: per_segment_param{iter} -> per_segment_param<IteratorT, ValueT, Min,
// Max>
template <typename IteratorT>
per_segment_param(IteratorT) -> per_segment_param<IteratorT>;

// -----------------------------------------------------------------------------
// 3. Uniform Discrete Param
// -----------------------------------------------------------------------------
// Note: CTAD is not provided for Options... because specific integer values
// cannot be deduced from a runtime constructor argument.
template <typename T, T... Options>
struct uniform_discrete_param
{
  using value_type          = T;
  using param_tag           = tag_uniform;
  using supported_options_t = supported_options<T, Options...>;

  T value;

  _CCCL_HOST_DEVICE constexpr uniform_discrete_param(T v)
      : value(v)
  {}

  uniform_discrete_param() = default;
};

// -----------------------------------------------------------------------------
// 4. Per-Segment Discrete Param
// -----------------------------------------------------------------------------
template <typename IteratorT, typename T, T... Options>
struct per_segment_discrete_param
{
  using iterator_type       = IteratorT;
  using value_type          = T;
  using param_tag           = tag_per_segment;
  using supported_options_t = supported_options<T, Options...>;

  IteratorT iterator;

  _CCCL_HOST_DEVICE constexpr per_segment_discrete_param(IteratorT iter)
      : iterator(iter)
  {}

  per_segment_discrete_param() = default;
};

// -----------------------------------------------------------------------------
// [PARAMETER TYPE TRAITS AND HELPERS]
// -----------------------------------------------------------------------------

template <typename T>
using is_static_param = ::cuda::std::is_same<typename T::param_tag, tag_static>;

template <typename T>
inline constexpr bool is_static_param_v = is_static_param<T>::value;

template <typename T>
using is_uniform_param = ::cuda::std::is_same<typename T::param_tag, tag_uniform>;

template <typename T>
inline constexpr bool is_uniform_param_v = is_uniform_param<T>::value;

template <typename T>
using is_per_segment_param = ::cuda::std::is_same<typename T::param_tag, tag_per_segment>;

template <typename T>
inline constexpr bool is_per_segment_param_v = is_per_segment_param<T>::value;

// Helper function to statically determine if a parameter always is below a
// given threshold
template <typename ParamT, typename T>
constexpr _CCCL_HOST_DEVICE bool max_le(const ParamT&, T threshold)
{
  return ParamT::static_max_value <= threshold;
}

// Helper function to statically determine if a parameter always is above a
// given threshold
template <typename ParamT, typename T>
constexpr _CCCL_HOST_DEVICE bool min_ge(const ParamT&, T threshold)
{
  return ParamT::static_min_value >= threshold;
}

// Get max value (works for all types inheriting bounds_mixin)
template <typename T>
inline constexpr auto static_max_value_v = T::static_max_value;

// Get min value
template <typename T>
inline constexpr auto static_min_value_v = T::static_min_value;

//! Resolve parameter value for a given segment index
template <typename ParamT, typename SegmentIndexT>
constexpr _CCCL_HOST_DEVICE auto resolve_param(ParamT const& p, [[maybe_unused]] SegmentIndexT segment_id)
{
  if constexpr (is_static_param_v<ParamT>)
  {
    // Case 1: Compile-time constant.
    return ParamT::static_value;
  }
  else if constexpr (is_uniform_param_v<ParamT>)
  {
    // Case 2: Runtime uniform.
    return p.value;
  }
  else
  {
    // Case 3: Per-segment.
    static_assert(is_per_segment_param_v<ParamT>, "Unknown parameter type");
    return p.iterator[segment_id];
  }
}
} // namespace detail::params

namespace detail::segmented_topk
{
// ------------ Helper to create segment iterators ------------

template <typename SegmentSizeT, typename ItT>
struct segment_index_to_offset_op
{
  ItT base_ptr;
  SegmentSizeT segment_size;

  template <typename SegmentIndexT>
  _CCCL_HOST_DEVICE ItT operator()(SegmentIndexT segment_index)
  {
    return base_ptr + (segment_index * segment_size);
  }
};

template <typename SegmentIndexT, typename SegmentSizeT, typename ItT>
auto make_segment_iterator(ItT raw_ptr, SegmentSizeT segment_size)
{
  auto counting_it = ::cuda::make_counting_iterator(SegmentIndexT{0});

  // We transform that count into a pointer
  segment_index_to_offset_op<SegmentSizeT, ItT> functor{raw_ptr, segment_size};

  return cuda::make_transform_iterator(counting_it, functor);
}

// -----------------------------------------------------------------------------
// [ALGORITHM-SPECIFIC PARAMETER TYPES]
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
template <::cuda::std::int64_t StaticK>
using k_static = params::static_constant_param<::cuda::std::int64_t, StaticK>;

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
  total_num_items_guarantee(::cuda::std::int64_t num_items)
      : min_num_items(num_items)
      , max_num_items(num_items)
  {}
  total_num_items_guarantee(::cuda::std::int64_t min_items, ::cuda::std::int64_t max_items)
      : min_num_items(min_items)
      , max_num_items(max_items)
  {}
};

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

// segment_sizes: static bounds are limiting to invocations of BlockTopK or ClusterTopK
// -> true: Specialize Agent to be of AgentSegmentedTopKOneWorkerPerSegment
// -> else: Use AgentSegmentedTopK
// segment_sizes: runtime bounds are limiting to invocations of BlockTopK or ClusterTopK
// -> AgentSegmentedTopK: TBD

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

  using TopKPolicyT = typename find_valid_policy_t::worker_per_segment_policy_t;

  using AgentT = typename find_valid_policy_t::worker_per_segment_agent_t;

  // 3. Static Assertions (Constraints)
  static_assert(AgentT::tile_size >= params::static_max_value_v<SegmentSizeParameterT>,
                "Block size exceeds maximum segment size supported by SegmentSizeParameterT");
  static_assert(sizeof(AgentT::TempStorage) <= 48 * 1024, "Static shared memory per block must not exceed 48KB limit.");

  __shared__ typename AgentT::TempStorage temp_storage;

  // Instantiate agent
  AgentT agent(
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

  DispatchSegmentedTopK(
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
