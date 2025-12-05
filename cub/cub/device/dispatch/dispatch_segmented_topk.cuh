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

#include <cub/agent/agent_topk.cuh>
#include <cub/block/block_radix_sort.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/tuning/tuning_topk.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_temporary_storage.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>

#include <cuda/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

namespace detail::topk
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
template <typename Param, typename Functor>
_CCCL_HOST_DEVICE void dispatch_discrete(Param p, Functor&& f)
{
  using supported_list = typename Param::supported_options_t;
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

// -----------------------------------------------------------------------------
// [ALGORITHM-SPECIFIC PARAMETER TYPES]
// -----------------------------------------------------------------------------

// ------------ SELECTION DIRECTION PARAMETER TYPES ------------

// Selection direction known at compile time, same value applies to all segments
template <select SelectDirection>
using select_direction_static = uniform_discrete_param<select, SelectDirection>;

// Selection direction is a runtime value, same value applies to all segments
using select_direction_uniform = uniform_discrete_param<select, select::max, select::min>;

// Per-segment selection direction via iterator
template <typename SelectionDirectionIt, select... SelectDirectionOptions>
using select_direction_per_segment =
  per_segment_discrete_param<SelectionDirectionIt, select, SelectDirectionOptions...>;

// ------------ SEGMENT SIZE PARAMETER TYPES ------------

// Segment size known at compile time, same value applies to all segments
template <::cuda::std::int64_t SegmentSize>
using segment_size_static = static_constant_param<::cuda::std::int64_t, SegmentSize>;

// Segment size is a runtime value, same value applies to all segments
template <::cuda::std::int64_t MinSegmentSize = 0,
          ::cuda::std::int64_t MaxSegmentSize = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max()>
using segment_size_uniform = uniform_param<::cuda::std::int64_t, MinSegmentSize, MaxSegmentSize>;

// Segment size via iterator
template <typename SegmentSizesItT,
          ::cuda::std::int64_t MinSegmentSize = 1,
          ::cuda::std::int64_t MaxSegmentSize = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max()>
using segment_size_per_segment =
  per_segment_param<SegmentSizesItT, ::cuda::std::int64_t, MinSegmentSize, MaxSegmentSize>;

// ------------ K PARAMETER TYPES ------------

// K known at compile time, same value applies to all segments
template <::cuda::std::int64_t StaticK>
using k_static = static_constant_param<::cuda::std::int64_t, StaticK>;

// K is a runtime value, same value applies to all segments
template <::cuda::std::int64_t MinK = 1,
          ::cuda::std::int64_t MaxK = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max()>
struct k_uniform : public uniform_param<::cuda::std::int64_t, MinK, MaxK>
{};

// K via iterator
template <typename KItT,
          ::cuda::std::int64_t MinK = 1,
          ::cuda::std::int64_t MaxK = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max()>
using k_per_segment = per_segment_param<KItT, ::cuda::std::int64_t, MinK, MaxK>;

// ------------ TOTAL NUMBER OF SEGMENTS ------------
// Number of segments known at compile time
template <::cuda::std::int64_t StaticNumSegments>
using num_segments_static = static_constant_param<::cuda::std::int64_t, StaticNumSegments>;

// Number of segments is a runtime value
template <::cuda::std::int64_t MinNumSegments = 1,
          ::cuda::std::int64_t MaxNumSegments = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max()>
using num_segments_uniform = uniform_param<::cuda::std::int64_t, MinNumSegments, MaxNumSegments>;

// Number of segments via iterator
template <typename NumSegmentsItT,
          ::cuda::std::int64_t MinNumSegments = 1,
          ::cuda::std::int64_t MaxNumSegments = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max()>
using num_segments_per_segment =
  per_segment_param<NumSegmentsItT, ::cuda::std::int64_t, MinNumSegments, MaxNumSegments>;

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

template <typename KeyT, int BLOCK_DIM_X, int items_per_thread, typename ValueT = cub::NullType>
class BlockTopK
{
private:
  // Internal CUB primitive
  using BlockRadixSortT = cub::BlockRadixSort<KeyT, BLOCK_DIM_X, items_per_thread, ValueT>;

public:
  // Expose TempStorage requirements
  struct TempStorage
  {
    typename BlockRadixSortT::TempStorage sort_storage;
  };

private:
  TempStorage& temp_storage;
  int linear_tid;

public:
  __device__ __forceinline__ BlockTopK(TempStorage& temp_storage)
      : temp_storage(temp_storage)
      , linear_tid(threadIdx.x)
  {}

  /**
   * @brief Sorts the block such that the Top K elements are in the first K
   * positions.
   * * After this call:
   * - The data across all threads is sorted.
   * - The item at BlockRank `i` is located at:
   * Thread `i / items_per_thread`, Register index `i % items_per_thread`
   * - Valid Top-K items are those where BlockRank < K.
   *
   * @param keys           [In/Out] Thread-local array of keys
   * @param values         [In/Out] Thread-local array of values
   * @param k              [In] Number of top elements to select
   * @param is_descending  [In] If true, largest elements are first (default:
   * true)
   * @param valid_items    [In] Number of valid items in the block (default:
   * full block)
   * @param begin_bit      [In] Least significant bit index for radix sort
   * (default: 0)
   * @param end_bit        [In] Most significant bit index for radix sort
   * (default: sizeof(KeyT)*8)
   */
  __device__ __forceinline__ void Select(
    KeyT (&keys)[items_per_thread],
    ValueT (&values)[items_per_thread],
    int k,
    bool is_descending = true,
    int valid_items    = BLOCK_DIM_X * items_per_thread,
    int begin_bit      = 0,
    int end_bit        = sizeof(KeyT) * 8)
  {
    // Delegate to CUB BlockRadixSort
    // Note: BlockRadixSort produces a BLOCKED arrangement.
    // Thread 0 has items [0 .. IPT-1], Thread 1 has [IPT .. 2*IPT-1], etc.

    if (is_descending)
    {
      // Sort Descending: Largest items move to Rank 0 (Thread 0)
      BlockRadixSortT(temp_storage.sort_storage).SortDescending(keys, values, begin_bit, end_bit);
    }
    else
    {
      // Sort Ascending: Smallest items move to Rank 0 (Thread 0)
      BlockRadixSortT(temp_storage.sort_storage).Sort(keys, values, begin_bit, end_bit);
    }

    // Logic barrier implicit in CUB sort
  }

  // Overload for Keys only
  __device__ __forceinline__ void Select(
    KeyT (&keys)[items_per_thread],
    int k,
    bool is_descending = true,
    int valid_items    = BLOCK_DIM_X * items_per_thread,
    int begin_bit      = 0,
    int end_bit        = sizeof(KeyT) * 8)
  {
    if (is_descending)
    {
      BlockRadixSortT(temp_storage.sort_storage).SortDescending(keys, begin_bit, end_bit);
    }
    else
    {
      BlockRadixSortT(temp_storage.sort_storage).Sort(keys, begin_bit, end_bit);
    }
  }
};

// get_next_block_id()

// segment_sizes: static bounds are limiting to invocations of BlockTopK or ClusterTopK
// -> true: Specialize Agent to be of AgentSegmentedTopKOneWorkerPerSegment
// -> else: Use AgentSegmentedTopK
// segment_sizes: runtime bounds are limiting to invocations of BlockTopK or ClusterTopK
// -> AgentSegmentedTopK: TBD

template <typename ActivePolicyT,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT>
struct AgentSegmentedTopkWorkerPerSegment
{
  // -------------------------------------------------------------------------
  // Types and Constants
  // -------------------------------------------------------------------------

  // Derive inner types from Iterator of Iterators
  using key_it_t   = typename ::cuda::std::iterator_traits<KeyInputItItT>::value_type;
  using value_it_t = typename ::cuda::std::iterator_traits<ValueInputItItT>::value_type;

  using key_t   = typename ::cuda::std::iterator_traits<key_it_t>::value_type;
  using value_t = typename ::cuda::std::iterator_traits<value_it_t>::value_type;

  static constexpr int block_threads    = ActivePolicyT::block_threads;
  static constexpr int items_per_thread = ActivePolicyT::items_per_thread;
  static constexpr int tile_size        = block_threads * items_per_thread;

  // Check if we are dealing with Keys-Only or Keys-Values
  static constexpr bool is_keys_only = ::cuda::std::is_same<value_t, cub::NullType>::value;

  // -------------------------------------------------------------------------
  // Primitive Types
  // -------------------------------------------------------------------------

  using BlockLoadKeysT = BlockLoad<key_t, block_threads, items_per_thread, BLOCK_LOAD_WARP_TRANSPOSE>;
  using BlockLoadValsT = BlockLoad<value_t, block_threads, items_per_thread, BLOCK_LOAD_WARP_TRANSPOSE>;

  using BlockTopkT = BlockTopK<key_t, block_threads, items_per_thread, value_t>;

  using BlockStoreKeysT = BlockStore<key_t, block_threads, items_per_thread, BLOCK_STORE_WARP_TRANSPOSE>;
  using BlockStoreValsT = BlockStore<value_t, block_threads, items_per_thread, BLOCK_STORE_WARP_TRANSPOSE>;

  // -------------------------------------------------------------------------
  // Shared Memory Storage
  // -------------------------------------------------------------------------

  struct TempStorage
  {
    union
    {
      typename BlockLoadKeysT::TempStorage load_keys;
      typename BlockLoadValsT::TempStorage load_vals;
      typename BlockTopkT::TempStorage topk;
      typename BlockStoreKeysT::TempStorage store_keys;
      typename BlockStoreValsT::TempStorage store_vals;
    };
  };

  // -------------------------------------------------------------------------
  // Members
  // -------------------------------------------------------------------------

  TempStorage& temp_storage;
  KeyInputItItT d_key_segments_it;
  KeyOutputItItT d_key_segments_out_it;
  ValueInputItItT d_value_segments_it;
  ValueOutputItItT d_value_segments_out_it;
  SegmentSizeParameterT segment_sizes;
  KParameterT k_param;
  SelectDirectionParameterT select_directions;
  NumSegmentsParameterT num_segments;

  // -------------------------------------------------------------------------
  // Constructor
  // -------------------------------------------------------------------------

  __device__ __forceinline__ AgentSegmentedTopkWorkerPerSegment(
    TempStorage& temp_storage,
    KeyInputItItT d_key_segments_it,
    KeyOutputItItT d_key_segments_out_it,
    ValueInputItItT d_value_segments_it,
    ValueOutputItItT d_value_segments_out_it,
    SegmentSizeParameterT segment_sizes,
    KParameterT k_param,
    SelectDirectionParameterT select_directions,
    NumSegmentsParameterT num_segments)
      : temp_storage(temp_storage)
      , d_key_segments_it(d_key_segments_it)
      , d_key_segments_out_it(d_key_segments_out_it)
      , d_value_segments_it(d_value_segments_it)
      , d_value_segments_out_it(d_value_segments_out_it)
      , segment_sizes(segment_sizes)
      , k_param(k_param)
      , select_directions(select_directions)
      , num_segments(num_segments)
  {}

  // -------------------------------------------------------------------------
  // Processing Logic
  // -------------------------------------------------------------------------

  __device__ __forceinline__ void Process()
  {
    // 1. Identify Segment
    int segment_id = blockIdx.x;

    // Boundary check
    // Note: Using resolve_param to handle various parameter types safely
    if (segment_id >= resolve_param(num_segments, 0))
    {
      return;
    }

    // 2. Resolve Segment Parameters
    auto segment_size = resolve_param(segment_sizes, segment_id);
    auto k            = resolve_param(k_param, segment_id);
    auto direction    = resolve_param(select_directions, segment_id);

    // Determine padding key based on direction (Max-K needs Lowest(), Min-K needs Max())
    // Assuming 'select' enum has ::max or ::min. Adjust as per your specific Enum definition.
    key_t padding_key = (direction == select::max) ? ::cuda::std::numeric_limits<key_t>::lowest()
                                                   : ::cuda::std::numeric_limits<key_t>::max();

    // 3. Load Keys
    key_t thread_keys[items_per_thread];

    // Dereference iterator-of-iterators to get the segment specific iterator
    auto block_keys_in = d_key_segments_it[segment_id];

    BlockLoadKeysT(temp_storage.load_keys).Load(block_keys_in, thread_keys, segment_size, padding_key);

    // 4. Load Values (if applicable)
    value_t thread_values[items_per_thread];

    if constexpr (!is_keys_only)
    {
      __syncthreads(); // Barrier for smem reuse
      auto block_vals_in = d_value_segments_it[segment_id];

      BlockLoadValsT(temp_storage.load_vals).Load(block_vals_in, thread_values, segment_size);
    }

    // 5. Perform Block Top-K
    __syncthreads(); // Barrier for smem reuse

    if constexpr (!is_keys_only)
    {
      // Pass both keys and values
      BlockTopkT(temp_storage.topk)
        .Select(thread_keys,
                thread_values,
                k,
                (direction == select::max), // is_descending
                segment_size);
    }
    else
    {
      // Keys only
      BlockTopkT(temp_storage.topk)
        .Select(thread_keys,
                k,
                (direction == select::max), // is_descending
                segment_size);
    }

    // 6. Store Results
    __syncthreads(); // Barrier for smem reuse

    auto block_keys_out = d_key_segments_out_it[segment_id];

    BlockStoreKeysT(temp_storage.store_keys)
      .Store(block_keys_out,
             thread_keys,
             k // Only store K items
      );

    if constexpr (!is_keys_only)
    {
      __syncthreads(); // Barrier for smem reuse
      auto block_vals_out = d_value_segments_out_it[segment_id];

      BlockStoreValsT(temp_storage.store_vals).Store(block_vals_out, thread_values, k);
    }
  }
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
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::topk_policy_t::block_threads)) __global__
  void DeviceSegmentedTopKKernel(
    KeyInputItItT d_key_segments_it,
    KeyOutputItItT d_key_segments_out_it,
    ValueInputItItT d_value_segments_it,
    ValueOutputItItT d_value_segments_out_it,
    SegmentSizeParameterT segment_sizes,
    KParameterT k,
    SelectDirectionParameterT select_directions,
    NumSegmentsParameterT num_segments)
{
  using ActivePolicyT = typename ChainedPolicyT::ActivePolicy;
  using TopKPolicyT   = typename ActivePolicyT::topk_policy_t;

  using AgentT = AgentSegmentedTopkWorkerPerSegment<
    TopKPolicyT,
    KeyInputItItT,
    KeyOutputItItT,
    ValueInputItItT,
    ValueOutputItItT,
    SegmentSizeParameterT,
    KParameterT,
    SelectDirectionParameterT,
    NumSegmentsParameterT>;

  // 3. Static Assertions (Constraints)
  static_assert(AgentT::tile_size >= static_max_value_v<SegmentSizeParameterT>,
                "Block size exceeds maximum segment size supported by SegmentSizeParameterT");

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
          typename SelectedPolicy = policy_hub<it_value_t<KeyInputItItT>, ::cuda::std::int64_t>>
struct DispatchSegmentedTopK
{
  using offset_t = ::cuda::std::int64_t;

  // TODO (elstehle): consider making this part of the env-based API
  // The algorithm allocates a double-buffer for intermediate results of size
  // num_items/coefficient_for_candidate_buffer
  static constexpr offset_t coefficient_for_candidate_buffer = 128;

  /// Device-accessible allocation of temporary storage.
  /// When `nullptr`, the required allocation size is written to
  /// `temp_storage_bytes` and no work is done.
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /// Pointer to the input sequence of data items
  KeyInputItItT d_key_segments_it;

  /// Pointer to the K output sequence of key data
  KeyOutputItItT d_key_segments_out_it;

  /// Pointer to the input sequence of associated value items
  ValueInputItItT d_value_segments_it;

  /// Pointer to the output sequence of associated value items
  ValueOutputItItT d_value_segments_out_it;

  /// Segment sizes
  SegmentSizeParameterT segment_sizes;

  /// The K value
  KParameterT k;

  /// The selection direction for each segment
  SelectDirectionParameterT select_directions;

  /// Number of segments
  NumSegmentsParameterT num_segments;

  // Allows the user to provide a guarantee on the upper bound of the total number of items
  TotalNumItemsGuaranteeT total_num_items_guarantee;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  using key_in_t =
    typename ::cuda::std::iterator_traits<typename ::cuda::std::iterator_traits<KeyInputItItT>::value_type>::value_type;

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

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t InvokeFixedSegmentSize()
  {
    using max_policy_t = typename SelectedPolicy::max_policy;

    // Only unfiorm segment sizes are supported here
    static_assert(!is_per_segment_param_v<SegmentSizeParameterT>,
                  "Only uniform segment sizes are currently supported.");

    // Instantiate the kernel with the selected policy and check shared memory
    // requirements
    using topk_policy_t = typename ActivePolicyT::topk_policy_t;

    constexpr int block_dim = topk_policy_t::block_threads;

    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1; // TODO (elstehle): calculate real storage size, if needed
      return cudaSuccess;
    }

    // TODO (elstehle): support number of segments provided by device-accessible iterator
    // Only uniform number of segments are supported currently (i.e., we
    // need to be able to resolve the number of segments on the host side)
    static_assert(!is_per_segment_param_v<NumSegmentsParameterT>,
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

    // Currently, we only support fixed-size segments that fit into shared memory
    // TODO (elstehle): extend support for variable-size segments
    if constexpr (!is_per_segment_param_v<SegmentSizeParameterT>)
    {
      return InvokeFixedSegmentSize<ActivePolicyT>();
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
} // namespace detail::topk

CUB_NAMESPACE_END
