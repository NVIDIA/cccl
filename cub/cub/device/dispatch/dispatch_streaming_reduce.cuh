/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * @file cub::DeviceReduce provides device-wide, parallel operations for
 *       computing a reduction across a sequence of data items residing within
 *       device-accessible memory.
 */

#pragma once

#include <cub/config.cuh>

#include "cuda/std/__type_traits/is_assignable.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_CCCL_SUPPRESS_DEPRECATED_PUSH
#include <cuda/std/functional>
_CCCL_SUPPRESS_DEPRECATED_POP

#include <cub/device/dispatch/dispatch_reduce.cuh>
#include <cub/iterator/arg_index_input_iterator.cuh>
#include <cub/iterator/constant_input_iterator.cuh>

#include <thrust/iterator/tabulate_output_iterator.h>

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace reduce
{

template <typename GlobalAccumT, typename PromoteToGlobalOpT, typename GlobalReductionOpT, typename FinalResultOutIteratorT>
class accumulating_transform_output_op
{
private:
  bool first_partition = true;
  bool last_partition  = false;

  // We use a double-buffer to make assignment idempotent (i.e., allow potential repeated assignment)
  GlobalAccumT* d_previous_aggregate = nullptr;
  GlobalAccumT* d_aggregate_out      = nullptr;

  // Output iterator to which the final result of type `GlobalAccumT` across all partitions will be assigned
  FinalResultOutIteratorT d_out;

  // Unary promotion operator type that is used to transform a per-partition result to a global result
  PromoteToGlobalOpT promote_op;

  // Reduction operation
  GlobalReductionOpT reduce_op;

public:
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE accumulating_transform_output_op(
    GlobalAccumT* d_previous_aggregate,
    GlobalAccumT* d_aggregate_out,
    bool is_last_partition,
    FinalResultOutIteratorT d_out,
    PromoteToGlobalOpT promote_op,
    GlobalReductionOpT reduce_op)
      : last_partition(is_last_partition)
      , d_previous_aggregate(d_previous_aggregate)
      , d_aggregate_out(d_aggregate_out)
      , d_out(d_out)
      , promote_op(promote_op)
      , reduce_op(reduce_op)
  {}

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
    ::cuda::std::swap(d_previous_aggregate, d_aggregate_out);
    first_partition = false;
    last_partition  = next_partition_is_the_last;
  };
};

/**
 * Unary promotion operator type that is used to transform a per-partition result to a global result
 */
template <typename GlobalOffsetT>
struct promote_to_global_op
{
  // The current partition's offset to be factored into this partition's index
  GlobalOffsetT current_partition_offset;

  /**
   * This helper function is invoked after a partition has been fully processed, in preparation for the next partition.
   */
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void advance(GlobalOffsetT partition_size)
  {
    current_partition_offset += partition_size;
  };

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

/**
 * Offsets a the given input iterator by a fixed offset, such that when an item at index `i` is accessed, the item
 * `it[*offset_it + i]` is accessed.
 */
template <typename Iterator, typename OffsetItT>
class offset_iterator : public thrust::iterator_adaptor<offset_iterator<Iterator, OffsetItT>, Iterator>
{
public:
  using super_t = thrust::iterator_adaptor<offset_iterator<Iterator, OffsetItT>, Iterator>;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE offset_iterator(const Iterator& it, OffsetItT offset_it)
      : super_t(it)
      , offset_it(offset_it)
  {}

  // befriend thrust::iterator_core_access to allow it access to the private interface below
  friend class thrust::iterator_core_access;

private:
  OffsetItT offset_it;

  _CCCL_DEVICE _CCCL_FORCEINLINE typename super_t::reference dereference() const
  {
    return *(this->base() + (*offset_it));
  }
};

template <typename Iterator, typename OffsetItT>
_CCCL_HOST_DEVICE offset_iterator<Iterator, OffsetItT> make_offset_iterator(Iterator it, OffsetItT offset_it)
{
  return offset_iterator<Iterator, OffsetItT>{it, offset_it};
}

/**
 * Helper transform operation that writes to the (index, extremum)-key-value-pair to the user-provided output iterator.
 * This helper allows "implicit conversion" between KeyValuePair types that have a different key type, which may happen
 * if the user uses a different index type than the global offset type used by the algorithm.
 */
template <typename KeyValuePairOutItT>
struct write_to_user_out_it
{
  KeyValuePairOutItT out_it;

  template <typename IndexT, typename OffsetT, typename ResultT>
  _CCCL_DEVICE _CCCL_FORCEINLINE void operator()(IndexT, cub::KeyValuePair<OffsetT, ResultT> reduced_result)
  {
    using kv_pair_t = ::cuda::std::_If<
      ::cuda::std::is_assignable_v<decltype(*std::declval<KeyValuePairOutItT>()),
                                   cub::KeyValuePair<::cuda::std::uint64_t, ResultT>>,
      cub::KeyValuePair<::cuda::std::uint64_t, ResultT>,
      ::cuda::std::_If<::cuda::std::is_assignable_v<decltype(*std::declval<KeyValuePairOutItT>()),
                                                    cub::KeyValuePair<::cuda::std::int64_t, ResultT>>,
                       cub::KeyValuePair<::cuda::std::int64_t, ResultT>,
                       ::cuda::std::_If<::cuda::std::is_assignable_v<decltype(*std::declval<KeyValuePairOutItT>()),
                                                                     cub::KeyValuePair<::cuda::std::uint32_t, ResultT>>,
                                        cub::KeyValuePair<::cuda::std::uint32_t, ResultT>,
                                        cub::KeyValuePair<::cuda::std::int32_t, ResultT>>>>;

    using index_t = typename kv_pair_t::Key;
    *out_it       = kv_pair_t{static_cast<index_t>(reduced_result.key), reduced_result.value};
  }
};

/******************************************************************************
 * Single-problem streaming reduction dispatch
 *****************************************************************************/

/**
 * @brief Utility class for dispatching the appropriately-tuned kernels for
 *        device-wide reduction
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OutputIteratorT
 *   Output iterator type for writing the result of the (index, extremum)-key-value-pair
 *
 * @tparam PerPartitionOffsetT
 *   Offset type used as the index to access items within one partition, i.e., the offset type used within the kernel
 * template specialization
 *
 * @tparam GlobalOffsetT
 *   Offset type used as the index to access items within the total input range, i.e., in the range [d_in, d_in +
 * num_items)
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member
 *   `auto operator()(const T &a, const U &b)`
 *
 * @tparam InitT
 *   Initial value type
 */
template <typename InputIteratorT,
          typename OutputIteratorT,
          typename PerPartitionOffsetT,
          typename GlobalOffsetT,
          typename ReductionOpT,
          typename InitT>
struct DispatchStreamingArgReduce
{
  /**
   * @brief Internal dispatch routine for computing a device-wide reduction
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no work
   *   is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_in
   *   Pointer to the input sequence of data items
   *
   * @param[out] d_result_out
   *   Iterator to which the  (index, extremum)-key-value-pair is written
   *
   * @param[in] num_items
   *   Total number of input items (i.e., length of `d_in`)
   *
   * @param[in] reduce_op
   *   Reduction operator that returns the (index, extremum)-key-value-pair corresponding to the extremum of the two
   * input items.
   *
   * @param[in] init
   *   The extremum value to be returned for empty problems
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   */
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_result_out,
    GlobalOffsetT num_items,
    ReductionOpT reduce_op,
    InitT init,
    cudaStream_t stream)
  {
    // The input type
    using input_value_t = cub::detail::value_t<InputIteratorT>;

    // Constant iterator to provide the offset of the current partition for the user-provided input iterator
    using constant_offset_it_t = cub::ConstantInputIterator<GlobalOffsetT>;

    // Wrapped input iterator to produce index-value tuples, i.e., <PerPartitionOffsetT, InputT>-tuples
    // We make sure to offset the user-provided input iterator by the current partition's offset
    using arg_index_input_iterator_t =
      ArgIndexInputIterator<offset_iterator<InputIteratorT, constant_offset_it_t>, PerPartitionOffsetT, InitT>;

    // The type used for the aggregate that the user wants to find the extremum for
    using output_aggregate_t = InitT;

    // The output tuple type (i.e., extremum plus index tuples)
    using per_partition_accum_t = KeyValuePair<PerPartitionOffsetT, output_aggregate_t>;
    using global_accum_t        = KeyValuePair<GlobalOffsetT, output_aggregate_t>;

    // Unary promotion operator type that is used to transform a per-partition result to a global result
    // operator()(per_partition_accum_t) -> global_accum_t
    using promote_to_global_op_t = promote_to_global_op<GlobalOffsetT>;

    // Reduction operator type that enables accumulating per-partition results to a global reduction result
    using accumulating_transform_output_op_t =
      accumulating_transform_output_op<global_accum_t, promote_to_global_op_t, ReductionOpT, OutputIteratorT>;

    // The output iterator that implements the logic to accumulate per-partition result to a global aggregate and,
    // eventually, write to the user-provided output iterators
    using accumulating_transform_out_it_t = thrust::tabulate_output_iterator<accumulating_transform_output_op_t>;

    // Empty problem initialization type
    using empty_problem_init_t = empty_problem_init_t<per_partition_accum_t>;

    // Per-partition DispatchReduce template specialization
    using dispatch_reduce_t =
      DispatchReduce<arg_index_input_iterator_t,
                     accumulating_transform_out_it_t,
                     PerPartitionOffsetT,
                     ReductionOpT,
                     empty_problem_init_t,
                     per_partition_accum_t>;

    void* allocations[2]       = {nullptr};
    size_t allocation_sizes[2] = {0, 2 * sizeof(global_accum_t)};

    // The current partition's input iterator is an ArgIndex iterator that generates indexes relative to the beginning
    // of the current partition, i.e., [0, partition_size) along with an OffsetIterator that offsets the user-provided
    // input iterator by the current partition's offset
    arg_index_input_iterator_t d_indexed_offset_in(make_offset_iterator(d_in, constant_offset_it_t{GlobalOffsetT{0}}));

    // Transforms the per-partition result to a global result by adding the current partition's offset to the arg result
    // of a partition
    promote_to_global_op_t promote_to_global_op{GlobalOffsetT{0}};

    accumulating_transform_output_op_t accumulating_out_op(
      nullptr, nullptr, false, d_result_out, promote_to_global_op, reduce_op);

    // Upper bound at which we want to cut the input into multiple partitions
    static constexpr PerPartitionOffsetT max_partition_size = ::cuda::std::numeric_limits<PerPartitionOffsetT>::max();

    // Whether the given number of items fits into a single partition
    const bool is_single_partition =
      static_cast<GlobalOffsetT>(max_partition_size) >= static_cast<GlobalOffsetT>(num_items);

    // The largest partition size ever encountered
    const auto largest_partition_size =
      is_single_partition ? static_cast<PerPartitionOffsetT>(num_items) : max_partition_size;

    empty_problem_init_t initial_value{{PerPartitionOffsetT{1}, init}};

    // Query temporary storage requirements for per-partition reduction
    dispatch_reduce_t::Dispatch(
      nullptr,
      allocation_sizes[0],
      d_indexed_offset_in,
      thrust::make_tabulate_output_iterator(accumulating_out_op),
      static_cast<PerPartitionOffsetT>(largest_partition_size),
      reduce_op,
      initial_value,
      stream);

    // Alias the temporary allocations from the single storage blob (or compute the necessary size
    // of the blob)
    cudaError_t error = cub::AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes);
    if (error != cudaSuccess)
    {
      return error;
    }

    // Return if the caller is simply requesting the size of the storage allocation
    if (d_temp_storage == nullptr)
    {
      return cudaSuccess;
    }

    // Pointer to the double-buffer of global accumulators, which aggregate cross-partition results
    global_accum_t* d_global_aggregates = reinterpret_cast<global_accum_t*>(allocations[1]);

    accumulating_out_op = accumulating_transform_output_op_t{
      d_global_aggregates,
      (d_global_aggregates + 1),
      is_single_partition,
      d_result_out,
      promote_to_global_op,
      reduce_op};

    for (GlobalOffsetT current_partition_offset = 0; current_partition_offset < static_cast<GlobalOffsetT>(num_items);
         current_partition_offset += static_cast<GlobalOffsetT>(max_partition_size))
    {
      const GlobalOffsetT remaining_items = (num_items - current_partition_offset);
      GlobalOffsetT current_num_items = (remaining_items < max_partition_size) ? remaining_items : max_partition_size;

      d_indexed_offset_in = arg_index_input_iterator_t(
        detail::reduce::make_offset_iterator(d_in, constant_offset_it_t{current_partition_offset}));

      error = dispatch_reduce_t::Dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_indexed_offset_in,
        thrust::make_tabulate_output_iterator(accumulating_out_op),
        static_cast<PerPartitionOffsetT>(current_num_items),
        reduce_op,
        initial_value,
        stream);

      // Whether the next partition will be the last partition
      const bool next_partition_is_last =
        (remaining_items - current_num_items) <= static_cast<GlobalOffsetT>(max_partition_size);
      accumulating_out_op.advance(current_num_items, next_partition_is_last);
    }

    return cudaSuccess;
  }
};

} // namespace reduce
} // namespace detail

CUB_NAMESPACE_END
