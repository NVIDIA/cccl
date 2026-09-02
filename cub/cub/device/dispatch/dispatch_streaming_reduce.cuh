// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/dispatch/dispatch_reduce.cuh>
#include <cub/device/dispatch/tuning/tuning_reduce.cuh>
#include <cub/iterator/arg_index_input_iterator.cuh>

#include <cuda/__iterator/tabulate_output_iterator.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/__functional/identity.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/limits>

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

CUB_NAMESPACE_BEGIN

namespace detail::reduce
{
template <typename GlobalAccumT, typename PromoteToGlobalOpT, typename GlobalReductionOpT, typename FinalResultOutIteratorT>
struct accumulating_transform_output_op
{
  bool first_partition;
  bool last_partition;

  // We use a double-buffer to make assignment idempotent (i.e., allow potential repeated assignment)
  GlobalAccumT* d_previous_aggregate;
  GlobalAccumT* d_aggregate_out;

  // Output iterator to which the final result of type `GlobalAccumT` across all partitions will be assigned
  FinalResultOutIteratorT d_out;

  // Unary promotion operator type that is used to transform a per-partition result to a global result
  PromoteToGlobalOpT promote_op;

  // Reduction operation
  GlobalReductionOpT reduce_op;

  template <typename IndexT, typename AccumT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(IndexT, AccumT per_partition_aggregate)
  {
    // Add this partitions aggregate to the global aggregate
    if (first_partition)
    {
      *d_aggregate_out = promote_op(per_partition_aggregate);
    }
    else
    {
      *d_aggregate_out = reduce_op(*d_previous_aggregate, promote_op(per_partition_aggregate));
    }

    // If this is the last partition, we write the global aggregate to the user-provided iterator
    if (last_partition)
    {
      *d_out = *d_aggregate_out;
    }
  }

  /**
   * This is a helper function that's invoked after a partition has been fully processed
   */
  template <typename GlobalOffsetT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void advance(GlobalOffsetT partition_size, bool next_partition_is_the_last)
  {
    promote_op.advance(partition_size);
    using ::cuda::std::swap;
    swap(d_previous_aggregate, d_aggregate_out);
    first_partition = false;
    last_partition  = next_partition_is_the_last;
  }
};

/**
 * Unary "promotion" operator type that is used to transform a per-partition result to a global result
 */
template <typename GlobalOffsetT>
struct local_to_global_op
{
  // The current partition's offset to be factored into this partition's index
  GlobalOffsetT current_partition_offset;

  /**
   * This helper function is invoked after a partition has been fully processed, in preparation for the next partition.
   */
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void advance(GlobalOffsetT partition_size)
  {
    current_partition_offset += partition_size;
  }

  /**
   * Unary operator called to transform the per-partition aggregate of a partition to a global aggregate type (i.e., one
   * that is used to reduce across partitions).
   */
  template <typename PerPartitionOffsetT, typename AccumT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE KeyValuePair<GlobalOffsetT, AccumT>
  operator()(KeyValuePair<PerPartitionOffsetT, AccumT> partition_aggregate)
  {
    return KeyValuePair<GlobalOffsetT, AccumT>{
      current_partition_offset + static_cast<GlobalOffsetT>(partition_aggregate.key), partition_aggregate.value};
  }
};

template <typename ExtremumOutIteratorT, typename IndexOutIteratorT>
struct unzip_and_write_arg_extremum_op
{
  ExtremumOutIteratorT result_out_it;
  IndexOutIteratorT index_out_it;

  template <typename IndexT, typename KeyValuePairT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(IndexT, KeyValuePairT reduced_result)
  {
    *result_out_it = reduced_result.value;
    *index_out_it  = reduced_result.key;
  }
};

// transform the KeyValuePair<OffsetT, T> produced by ArgIndexInputIterator to argminmax_accum_t<T, OffsetT>
struct kvp_to_argminmax_accum
{
  template <typename T, typename OffsetT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto operator()(KeyValuePair<OffsetT, T> kv) const
    -> argminmax_accum_t<T, OffsetT>
  {
    return {kv.value, kv.value, kv.key, kv.key};
  }
};

// Local-to-global promotion for ArgMinMax: adds the partition offset to both min_index and max_index
template <typename GlobalOffsetT>
struct local_to_global_minmax_op
{
  GlobalOffsetT current_partition_offset;

  _CCCL_HOST_DEVICE void advance(GlobalOffsetT partition_size)
  {
    current_partition_offset += partition_size;
  }

  template <typename T, typename PerPartitionOffsetT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE detail::argminmax_accum_t<T, GlobalOffsetT>
  operator()(const detail::argminmax_accum_t<T, PerPartitionOffsetT>& p) const
  {
    return {p.min_value,
            p.max_value,
            current_partition_offset + static_cast<GlobalOffsetT>(p.min_index),
            current_partition_offset + static_cast<GlobalOffsetT>(p.max_index)};
  }
};

template <typename MinExtremumOutT, typename MinIndexOutT, typename MaxExtremumOutT, typename MaxIndexOutT>
struct write_arg_minmax_result_op
{
  MinExtremumOutT min_out;
  MinIndexOutT min_index_out;
  MaxExtremumOutT max_out;
  MaxIndexOutT max_index_out;

  template <typename IndexT, typename T, typename GlobalOffsetT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(IndexT, const argminmax_accum_t<T, GlobalOffsetT>& result)
  {
    *min_out       = result.min_value;
    *min_index_out = result.min_index;
    *max_out       = result.max_value;
    *max_index_out = result.max_index;
  }
};

/******************************************************************************
 * Single-problem streaming reduction dispatch
 *****************************************************************************/

template <typename PerPartitionAccumT,
          typename GlobalAccumT,
          typename PerPartitionOffsetT,
          typename ArgIndexInputIteratorT,
          typename InputIteratorT,
          typename ResultOutIteratorT,
          typename GlobalOffsetT,
          typename ReductionOpT,
          typename TransformOpT,
          typename InitValueT,
          typename PromoteToGlobalOpT,
          typename TuningEnvT>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_streaming_arg_reduce_impl(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  ResultOutIteratorT d_result_out,
  GlobalOffsetT num_items,
  ReductionOpT reduce_op,
  TransformOpT transform_op,
  InitValueT init_value,
  PromoteToGlobalOpT promote_op,
  cudaStream_t stream,
  const TuningEnvT& = {})
{
  // Resolve the tuning policy from the (optional) tuning environment, defaulting to the type-derived policy
  using default_policy_selector_t = policy_selector_from_types<PerPartitionAccumT, PerPartitionOffsetT, ReductionOpT>;
  using policy_selector_t =
    ::cuda::std::execution::__query_result_or_t<TuningEnvT, ReducePolicy, default_policy_selector_t>;

#  if _CCCL_HAS_CONCEPTS()
  static_assert(reduce_policy_selector<policy_selector_t>);
#  endif // _CCCL_HAS_CONCEPTS()

  // Upper bound at which we want to cut the input into multiple partitions. Align to 4096 bytes for performance reasons
  static constexpr PerPartitionOffsetT max_offset_size = ::cuda::std::numeric_limits<PerPartitionOffsetT>::max();
  static constexpr PerPartitionOffsetT max_partition_size =
    max_offset_size - (max_offset_size % PerPartitionOffsetT{4096});

  // Whether the given number of items fits into a single partition
  const bool is_single_partition =
    static_cast<GlobalOffsetT>(max_partition_size) >= static_cast<GlobalOffsetT>(num_items);

  // The largest partition size ever encountered
  const auto largest_partition_size =
    is_single_partition ? static_cast<PerPartitionOffsetT>(num_items) : max_partition_size;

  // The current partition's input iterator is an ArgIndex iterator that generates indices relative to the beginning of
  // the current partition, i.e., [0, partition_size), offset by the current partition's offset
  ArgIndexInputIteratorT d_indexed_in(d_in);

  // Reduction operator that enables accumulating per-partition results to a global reduction result
  auto accumulating_out_op =
    accumulating_transform_output_op<GlobalAccumT, PromoteToGlobalOpT, ReductionOpT, ResultOutIteratorT>{
      true, is_single_partition, nullptr, nullptr, d_result_out, promote_op, reduce_op};

  // Query temporary storage requirements for per-partition reduction
  void* allocations[2]       = {nullptr, nullptr};
  size_t allocation_sizes[2] = {0, 2 * sizeof(GlobalAccumT)};
  if (const auto error = CubDebug(reduce::dispatch<PerPartitionAccumT>(
        nullptr,
        allocation_sizes[0],
        d_indexed_in,
        ::cuda::make_tabulate_output_iterator(accumulating_out_op),
        static_cast<PerPartitionOffsetT>(largest_partition_size),
        reduce_op,
        init_value,
        stream,
        transform_op,
        policy_selector_t{})))
  {
    return error;
  }

  // Alias the temporary allocations from the single storage blob (or compute the necessary size of the blob)
  if (const auto error = CubDebug(alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
  {
    return error;
  }

  // Return if the caller is simply requesting the size of the storage allocation
  if (d_temp_storage == nullptr)
  {
    return cudaSuccess;
  }

  // Pointer to the double-buffer of global accumulators, which aggregate cross-partition results
  GlobalAccumT* const d_global_aggregates = static_cast<GlobalAccumT*>(allocations[1]);

  accumulating_out_op.d_previous_aggregate = d_global_aggregates;
  accumulating_out_op.d_aggregate_out      = d_global_aggregates + 1;

  for (GlobalOffsetT current_partition_offset = 0; current_partition_offset < static_cast<GlobalOffsetT>(num_items);
       current_partition_offset += static_cast<GlobalOffsetT>(max_partition_size))
  {
    const GlobalOffsetT remaining_items = (num_items - current_partition_offset);
    const GlobalOffsetT current_num_items =
      (remaining_items < max_partition_size) ? remaining_items : max_partition_size;

    d_indexed_in = ArgIndexInputIteratorT(d_in + current_partition_offset);

    if (const auto error = CubDebug(reduce::dispatch<PerPartitionAccumT>(
          d_temp_storage,
          temp_storage_bytes,
          d_indexed_in,
          ::cuda::make_tabulate_output_iterator(accumulating_out_op),
          static_cast<PerPartitionOffsetT>(current_num_items),
          reduce_op,
          init_value,
          stream,
          transform_op,
          policy_selector_t{})))
    {
      return error;
    }

    // Whether the next partition will be the last partition
    const bool next_partition_is_last =
      (remaining_items - current_num_items) <= static_cast<GlobalOffsetT>(max_partition_size);
    accumulating_out_op.advance(current_num_items, next_partition_is_last);
  }

  return cudaSuccess;
}

// Internal dispatch routine for computing a device-wide argument extremum, like `ArgMin` and `ArgMax`.
// Streaming, here, refers to the approach used for large number of items that are processed in multiple partitions.
//
// @tparam PerPartitionOffsetT
//   Offset type used as the index to access items within one partition, i.e., the offset type used within the kernel
//   template specialization
//
// @tparam InputIteratorT
//   Random-access input iterator type for reading input items @iterator
//
// @tparam OutputIteratorT
//   Output iterator type for writing the result of the (index, extremum)-key-value-pair
//
// @tparam GlobalOffsetT
//   Offset type used as the index to access items within the total input range, i.e., in the range [d_in, d_in +
// num_items)
//
// @tparam ReductionOpT
//   Binary reduction functor type having a member function that returns the selected extremum of two input items.
//   The streaming reduction requires two overloads, one used for selecting the extremum within one partition and one
//   for selecting the extremum across partitions.
//
// @tparam TuningEnvT
//   Tuning environment Environment type
//
template <typename PerPartitionOffsetT,
          typename InputIteratorT,
          typename ExtremumOutIteratorT,
          typename IndexOutIteratorT,
          typename GlobalOffsetT,
          typename ReductionOpT,
          typename TuningEnvT>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_streaming_arg_reduce(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  ExtremumOutIteratorT d_min_out,
  IndexOutIteratorT d_index_out,
  GlobalOffsetT num_items,
  ReductionOpT reduce_op,
  cudaStream_t stream,
  const TuningEnvT& tuning_env = {})
{
  using input_value_t = detail::it_value_t<InputIteratorT>;
  // TODO(bgruber): we should use the input_value_t in the accumulator and for comparison, and only covert when writing
  // the final result
  using output_extremum_t     = detail::non_void_value_t<ExtremumOutIteratorT, input_value_t>;
  using per_partition_accum_t = KeyValuePair<PerPartitionOffsetT, output_extremum_t>;
  using global_accum_t        = KeyValuePair<GlobalOffsetT, output_extremum_t>;

  // Wrapped input iterator to produce index-value tuples, i.e., <PerPartitionOffsetT, InputT>-tuples
  using arg_index_input_iterator_t = ArgIndexInputIterator<InputIteratorT, PerPartitionOffsetT, output_extremum_t>;

  // Tabulate output iterator that unzips the result and writes it to the user-provided output iterators
  auto d_result_out = ::cuda::make_tabulate_output_iterator(
    detail::reduce::unzip_and_write_arg_extremum_op<ExtremumOutIteratorT, IndexOutIteratorT>{d_min_out, d_index_out});

  // Initial value for empty problems, according to documented contract
  const auto empty_problem_extremum = static_cast<output_extremum_t>([] {
    if constexpr (::cuda::std::is_same_v<ReductionOpT, arg_min>
                  && ::cuda::std::numeric_limits<input_value_t>::is_specialized)
    {
      return ::cuda::std::numeric_limits<input_value_t>::max();
    }
    else if constexpr (::cuda::std::is_same_v<ReductionOpT, arg_max>
                       && ::cuda::std::numeric_limits<input_value_t>::is_specialized)
    {
      return ::cuda::std::numeric_limits<input_value_t>::lowest();
    }
    else
    {
      return input_value_t{};
    }
  }());
  auto initial_value = empty_problem_init_t<per_partition_accum_t>{{PerPartitionOffsetT{1}, empty_problem_extremum}};

  return dispatch_streaming_arg_reduce_impl<per_partition_accum_t,
                                            global_accum_t,
                                            PerPartitionOffsetT,
                                            arg_index_input_iterator_t>(
    d_temp_storage,
    temp_storage_bytes,
    d_in,
    d_result_out,
    num_items,
    reduce_op,
    ::cuda::std::identity{},
    initial_value,
    local_to_global_op<GlobalOffsetT>{GlobalOffsetT{0}},
    stream,
    tuning_env);
}

// Internal dispatch routine for computing a device-wide combined argument minimum and maximum in a single pass.
// Streaming, here, refers to the approach used for large number of items that are processed in multiple partitions.
//
// @tparam PerPartitionOffsetT
//   Offset type used as the index to access items within one partition
//
// @tparam InputIteratorT
//   Random-access input iterator type for reading input items @iterator
//
// @tparam MinExtremumOutIteratorT
//   Output iterator type for writing the minimum value
//
// @tparam MinIndexOutIteratorT
//   Output iterator type for writing the index of the minimum value
//
// @tparam MaxExtremumOutIteratorT
//   Output iterator type for writing the maximum value
//
// @tparam MaxIndexOutIteratorT
//   Output iterator type for writing the index of the maximum value
//
// @tparam GlobalOffsetT
//   Offset type used as the index to access items within the total input range
//
// @tparam TuningEnvT
//   Tuning environment type
//
template <typename PerPartitionOffsetT,
          typename InputIteratorT,
          typename MinExtremumOutIteratorT,
          typename MinIndexOutIteratorT,
          typename MaxExtremumOutIteratorT,
          typename MaxIndexOutIteratorT,
          typename GlobalOffsetT,
          typename ReductionOpT,
          typename TuningEnvT>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t dispatch_streaming_arg_minmax(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  MinExtremumOutIteratorT d_min_out,
  MinIndexOutIteratorT d_min_index_out,
  MaxExtremumOutIteratorT d_max_out,
  MaxIndexOutIteratorT d_max_index_out,
  GlobalOffsetT num_items,
  ReductionOpT reduce_op,
  cudaStream_t stream,
  const TuningEnvT& tuning_env = {})
{
  using input_value_t         = it_value_t<InputIteratorT>;
  using per_partition_accum_t = argminmax_accum_t<input_value_t, PerPartitionOffsetT>;
  using global_accum_t        = argminmax_accum_t<input_value_t, GlobalOffsetT>;

  using arg_index_input_iterator_t = ArgIndexInputIterator<InputIteratorT, PerPartitionOffsetT>;

  // output iterator that splits the accumulator and writes to the four user-provided output iterators
  auto d_result_out = ::cuda::make_tabulate_output_iterator(
    write_arg_minmax_result_op<MinExtremumOutIteratorT,
                               MinIndexOutIteratorT,
                               MaxExtremumOutIteratorT,
                               MaxIndexOutIteratorT>{d_min_out, d_min_index_out, d_max_out, d_max_index_out});

  return dispatch_streaming_arg_reduce_impl<per_partition_accum_t,
                                            global_accum_t,
                                            PerPartitionOffsetT,
                                            arg_index_input_iterator_t>(
    d_temp_storage,
    temp_storage_bytes,
    d_in,
    d_result_out,
    num_items,
    reduce_op,
    kvp_to_argminmax_accum{},
    no_init,
    local_to_global_minmax_op<GlobalOffsetT>{GlobalOffsetT{0}},
    stream,
    tuning_env);
}
} // namespace detail::reduce
CUB_NAMESPACE_END

#endif // !_CCCL_DOXYGEN_INVOKED
