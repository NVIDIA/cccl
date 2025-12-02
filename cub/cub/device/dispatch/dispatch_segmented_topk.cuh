// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! cub::DeviceTopK provides device-wide, parallel operations for finding the K largest (or smallest) items
//! from sequences of unordered data items residing within device-accessible memory.

#pragma once

#include <cub/config.cuh>

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

CUB_NAMESPACE_BEGIN

namespace detail::topk
{

#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <type_traits>

// Allows constraining static bounds for parameters
template <typename T, T Min = ::cuda::std::numeric_limits<T>::min(), T Max = ::cuda::std::numeric_limits<T>::max()>
struct static_bounds_mixin
{
  static_assert(Min <= Max, "Min must be <= Max");

  static constexpr T static_min_value = Min;
  static constexpr T static_max_value = Max;

  // Indicates that there's only one possible value
  static constexpr bool is_exact = (Min == Max);
};

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

// Runtime value applicable to all segments
template <typename T, T Min, T Max>
struct uniform_param : public static_bounds_mixin<T, Min, Max>
{
  using value_type = T;
  using param_tag  = tag_uniform;

  T value;
};

// Per-segment value (provided by an iterator)
template <typename IteratorT, typename T, T Min, T Max>
struct per_segment_param : public static_bounds_mixin<T, Min, Max>
{
  using iterator_type = IteratorT;
  using value_type    = T;
  using param_tag     = tag_per_segment;

  IteratorT iterator;
  T min_value = Min;
  T max_value = Max;
};


template<typename T, T... Options>
struct supported_options
{
  static constexpr size_t count = sizeof...(Options);
};

// Runtime value (from a discrete set of values) applicable to all segments
template <typename T, T... Options>
struct uniform_discrete_param
{
  using value_type = T;
  using param_tag  = tag_uniform;
  using supported_options_t = supported_options<T, Options...>;

  T value;
};

// Per-segment value (provided by an iterator) of discrete values
template <typename IteratorT, typename T, T... Options>
struct per_segment_discrete_param
{
  using iterator_type = IteratorT;
  using value_type    = T;
  using param_tag     = tag_per_segment;
  using supported_options_t = supported_options<T, Options...>;

  IteratorT iterator;
};


// Helper to peel the supported_values pack
template <typename T, T... Opts, typename Param, typename Functor>
_CCCL_HOST_DEVICE
void dispatch_impl(Param p, supported_options<T, Opts...>, Functor&& f) {
    
    // Fold expression over the supported options.
    // This generates code equivalent to:
    // if (p.value == Opt1) f(integral_constant<Opt1>);
    // else if (p.value == Opt2) f(integral_constant<Opt2>);
    // ...
    
    bool match_found = ((p.value == Opts ? (f(::cuda::std::integral_constant<T, Opts>{}), true) : false) || ...);
    
    // Optional: Handling cases where the runtime value was not in the supported list.
    // In a release build, we assume the user respected the contract.
    // (void)match_found; 
}

template <typename Param, typename Functor>
_CCCL_HOST_DEVICE
void dispatch_discrete(Param p, Functor&& f) {
    using supported_list = typename Param::supported_options_t;
    dispatch_impl(p, supported_list{}, ::cuda::std::forward<Functor>(f));
}

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

// Helper function to statically determine if a parameter always is below a given threshold
template <typename ParamT, typename T>
constexpr _CCCL_HOST_DEVICE bool max_le(const ParamT&, T threshold)
{
  return ParamT::max_value <= threshold;
}

// Helper function to statically determine if a parameter always is above a given threshold
template <typename ParamT, typename T>
constexpr _CCCL_HOST_DEVICE bool min_ge(const ParamT&, T threshold)
{
  return ParamT::min_value >= threshold;
}

// Get max value (works for all types inheriting bounds_mixin)
template <typename T>
inline constexpr auto static_max_value_v = T::max_value;

// Get min value
template <typename T>
inline constexpr auto static_min_value_v = T::min_value;

template <typename ParamT, typename SegmentIndexT>
constexpr _CCCL_HOST_DEVICE auto resolve_param(ParamT const& p, SegmentIndexT segment_id)
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
    // Case 3: Per-segment (per_segment).
    static_assert(is_per_segment_param_v<ParamT>, "Unknown parameter type");
    return p.iterator[segment_id];
  }
}

// Selection direction known at compile time, same value applies to all segments
template <select SelectDirection>
struct select_direction_static
{
  static constexpr select select_direction = SelectDirection;
};

// Selection direction is a runtime value, same value applies to all segments
struct select_direction_uniform
{
  select select_direction;
};

// Per-segment selection direction via iterator
template <typename SelectionDirectionIt>
struct select_direction_per_segment
{
  SelectionDirectionIt select_directions;
};

// Segment size known at compile time, same value applies to all segments
template <::cuda::std::int64_t SegmentSize>
struct segment_size_static
{
  static constexpr ::cuda::std::int64_t segment_size = SegmentSize;
};

// Segment size is a runtime value, same value applies to all segments
template <::cuda::std::int64_t MaxSegmentSize = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max(),
          ::cuda::std::int64_t MinSegmentSize = 0>
struct segment_size_uniform
{
  ::cuda::std::int64_t value;
  static constexpr ::cuda::std::int64_t min_segment_size = MinSegmentSize;
  static constexpr ::cuda::std::int64_t max_segment_size = MaxSegmentSize;
};

// Segment size via iterator
template <typename SegmentSizesItT,
          ::cuda::std::int64_t MaxSegmentSize = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max(),
          ::cuda::std::int64_t MinSegmentSize = 0>
struct segment_size_per_segment
{
  SegmentSizesItT segment_size_it;
  static constexpr ::cuda::std::int64_t min_segment_size = MinSegmentSize;
  static constexpr ::cuda::std::int64_t max_segment_size = MaxSegmentSize;
};

// K known at compile time, same value applies to all segments
template <::cuda::std::int64_t StaticK>
struct k_static
{
  static constexpr ::cuda::std::int64_t K = StaticK;
};

// K is a runtime value, same value applies to all segments
template <::cuda::std::int64_t MaxK = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max(),
          ::cuda::std::int64_t MinK = 1>
struct k_uniform
{
  static constexpr ::cuda::std::int64_t min_k = MinK;
  static constexpr ::cuda::std::int64_t max_k = MaxK;

  ::cuda::std::int64_t value;
};

// K via iterator
// TODO (elstehle): should we consider moving the iterator template parameter the end, as it may be implicit from CTAD?
template <typename KItT,
          ::cuda::std::int64_t MaxK = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max(),
          ::cuda::std::int64_t MinK = 1>
struct k_per_segment
{
  static constexpr ::cuda::std::int64_t static_min_k = MinK;
  static constexpr ::cuda::std::int64_t static_max_k = MaxK;

  KItT k_it;
  static constexpr ::cuda::std::int64_t min_k = MinK;
  static constexpr ::cuda::std::int64_t max_k = MaxK;
};

// Number of items guarantee
template <::cuda::std::int64_t MaxNumItems  = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max(),
          ::cuda::std::int64_t MinNumItemsT = 1>
struct total_num_items_guarantee
{
  static constexpr ::cuda::std::int64_t static_min_num_items = MinNumItemsT;
  static constexpr ::cuda::std::int64_t static_max_num_items = MaxNumItems;

  static constexpr ::cuda::std::int64_t min_num_items = MinNumItemsT;
  static constexpr ::cuda::std::int64_t max_num_items = MaxNumItems;
};

template <
    typename KeyT,
    int BLOCK_DIM_X,
    int ITEMS_PER_THREAD,
    typename ValueT = cub::NullType
>
class BlockTopK
{
private:
    // Internal CUB primitive
    using BlockRadixSortT = cub::BlockRadixSort<KeyT, BLOCK_DIM_X, ITEMS_PER_THREAD, ValueT>;

public:
    // Expose TempStorage requirements
    struct TempStorage {
        typename BlockRadixSortT::TempStorage sort_storage;
    };

private:
    TempStorage& temp_storage;
    int linear_tid;

public:
    __device__ __forceinline__ BlockTopK(TempStorage& temp_storage)
        : temp_storage(temp_storage),
          linear_tid(threadIdx.x)
    {}

    /**
     * @brief Sorts the block such that the Top K elements are in the first K positions.
     * * After this call:
     * - The data across all threads is sorted.
     * - The item at BlockRank `i` is located at:
     * Thread `i / ITEMS_PER_THREAD`, Register index `i % ITEMS_PER_THREAD`
     * - Valid Top-K items are those where BlockRank < K.
     *
     * @param keys           [In/Out] Thread-local array of keys
     * @param values         [In/Out] Thread-local array of values
     * @param k              [In] Number of top elements to select
     * @param is_descending  [In] If true, largest elements are first (default: true)
     * @param valid_items    [In] Number of valid items in the block (default: full block)
     * @param begin_bit      [In] Least significant bit index for radix sort (default: 0)
     * @param end_bit        [In] Most significant bit index for radix sort (default: sizeof(KeyT)*8)
     */
    __device__ __forceinline__ void Select(
        KeyT (&keys)[ITEMS_PER_THREAD],
        ValueT (&values)[ITEMS_PER_THREAD],
        int k,
        bool is_descending = true,
        int valid_items = BLOCK_DIM_X * ITEMS_PER_THREAD,
        int begin_bit = 0,
        int end_bit = sizeof(KeyT) * 8
    ) {
        // Delegate to CUB BlockRadixSort
        // Note: BlockRadixSort produces a BLOCKED arrangement.
        // Thread 0 has items [0 .. IPT-1], Thread 1 has [IPT .. 2*IPT-1], etc.
        
        if (is_descending) {
            // Sort Descending: Largest items move to Rank 0 (Thread 0)
            BlockRadixSortT(temp_storage.sort_storage).SortDescending(
                keys, values, begin_bit, end_bit, valid_items
            );
        } else {
            // Sort Ascending: Smallest items move to Rank 0 (Thread 0)
            BlockRadixSortT(temp_storage.sort_storage).Sort(
                keys, values, begin_bit, end_bit, valid_items
            );
        }

        // Logic barrier implicit in CUB sort
    }

    // Overload for Keys only
    __device__ __forceinline__ void Select(
        KeyT (&keys)[ITEMS_PER_THREAD],
        int k,
        bool is_descending = true,
        int valid_items = BLOCK_DIM_X * ITEMS_PER_THREAD,
        int begin_bit = 0,
        int end_bit = sizeof(KeyT) * 8
    ) {
        if (is_descending) {
            BlockRadixSortT(temp_storage.sort_storage).SortDescending(
                keys, begin_bit, end_bit, valid_items
            );
        } else {
            BlockRadixSortT(temp_storage.sort_storage).Sort(
                keys, begin_bit, end_bit, valid_items
            );
        }
    }
};

template <typename ChainedPolicyT,
          typename KeyInputItItT,
          typename KeyOutputItItT,
          typename ValueInputItItT,
          typename ValueOutputItItT,
          typename SegmentSizeParameterT,
          typename KParameterT,
          typename SelectDirectionParameterT,
          typename NumSegmentsParameterT>
__launch_bounds__(int(ChainedPolicyT::ActivePolicy::topk_policy_t::block_threads))
  CUB_DETAIL_KERNEL_ATTRIBUTES void DeviceSegmentedTopKKernel(
    KeyInputItItT d_key_segments_it,
    KeyOutputItItT d_key_segments_out_it,
    ValueInputItItT d_value_segments_it,
    ValueOutputItItT d_value_segments_out_it,
    SegmentSizeParameterT segment_sizes,
    KParameterT k_it,
    SelectDirectionParameterT select_directions,
    NumSegmentsParameterT num_segments,
    cudaStream_t stream)
{
  using active_policy_t = typename ChainedPolicyT::ActivePolicy;
  using topk_policy_t   = typename active_policy_t::topk_policy_t;

  // TODO (elstehle): infer the offset types
  using segment_size_t   = ::cuda::std::int64_t;
  using k_index_t   = ::cuda::std::int64_t;
  using segment_index_t = ::cuda::std::int64_t;

  constexpr int block_threads    = topk_policy_t::block_threads;
  constexpr int items_per_thread = topk_policy_t::items_per_thread;

  using key_t = it_value_t<it_value_t<KeyInputItItT>>;
  using value_t = it_value_t<it_value_t<ValueInputItItT>>;

  // Use Striped loading for non-deterministic results
  using block_load_keys_t = cub::BlockLoad<key_t, block_threads, items_per_thread, cub::BLOCK_LOAD_STRIPED>;
  using block_load_vals_t = cub::BlockLoad<value_t, block_threads, items_per_thread, cub::BLOCK_LOAD_STRIPED>;
  // Use your custom BlockTopK
  using block_topk_t = BlockTopK<key_t, block_threads, items_per_thread, value_t>;

  // Use Striped storage to write the top-k results coalesced
  using block_store_keys_t = cub::BlockStore<key_t, block_threads, items_per_thread, cub::BLOCK_STORE_STRIPED>;
  using block_store_vals_t = cub::BlockStore<value_t, block_threads, items_per_thread, cub::BLOCK_STORE_STRIPED>;

  // Shared memory union to save space
  __shared__ union
  {
    typename block_load_keys_t::TempStorage load_keys;
    typename block_load_vals_t::TempStorage load_vals;
    typename block_topk_t::TempStorage topk;
    typename block_store_keys_t::TempStorage store_keys;
    typename block_store_vals_t::TempStorage store_vals;
  } temp_storage;

  key_t thread_keys[items_per_thread];

  // Calculate Segment Offsets
  int segment_id = blockIdx.x;
  if (segment_id >= num_segments)
  {
    return;
  }

  segment_size_t segment_size = resolve_param(segment_sizes, segment_id);
  block_load_keys_t(temp_storage.load_keys).Load(d_key_segments_it[segment_id], thread_keys, segment_size);

  value_t thread_values[items_per_thread];
  // if constexpr ()
  // {
  //   // Make sure we can reuse shared memory for value loading
  //   __syncthreads();

  //   block_load_vals_t(temp_storage.load_vals).Load(d_value_segments_it[segment_id], thread_values, segment_size);
  // }

  // Make sure we can reuse shared memory for top-k selection
  __syncthreads();

  // This sorts the *entire* block in registers.
  // The top k items will end up in the first K positions
  k_index_t k               = resolve_param(k_it, segment_id);
  block_topk_t(temp_storage.topk).Select(thread_keys, thread_values, k, 
                                          resolve_param(select_directions, segment_id) == select::max,
                                          segment_size);

  // Make sure we can reuse shared memory for storing back the results
  __syncthreads();

  // We only want to write the first K items.
  // BlockStore accepts a 'valid_items' count. It will write the first
  // 'valid_items' logical ranks to the output pointer.
  block_store_keys_t(temp_storage.store_keys)
    .Store(d_key_segments_out_it[segment_id],
           thread_keys,
           k
    );

  __syncthreads();

  // if constexpr ()
  // {
  //   block_store_vals_t(temp_storage.store_vals)
  //     .Store(d_value_segments_out_it[segment_id],
  //            thread_values,
  //            k
  //     );
  // }
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
  // The algorithm allocates a double-buffer for intermediate results of size num_items/coefficient_for_candidate_buffer
  static constexpr offset_t coefficient_for_candidate_buffer = 128;

  /// Device-accessible allocation of temporary storage.
  /// When `nullptr`, the required allocation size is written to `temp_storage_bytes` and no work is done.
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
    KParameterT k_it,
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
      , k(k_it)
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

    static_assert(!is_per_segment_param_v<SegmentSizeParameterT>,
                  "Only uniform segment sizes are currently supported.");

    // Instantiate the kernel with the selected policy and check shared memory requirements
    using topk_policy_t = typename ActivePolicyT::topk_policy_t;
  }

  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke()
  {
    using max_policy_t = typename SelectedPolicy::max_policy;

    // Currently, we only support fixed-size segments that fit into shared memory
    // TODO (elstehle): extend support for variable-size segments
    if constexpr (is_per_segment_param_v<SegmentSizeParameterT>)
    {
      InvokeFixedSegmentSize<ActivePolicyT>();
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
    KParameterT k_it,
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
      k_it,
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
