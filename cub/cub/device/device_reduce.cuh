/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2024, NVIDIA CORPORATION.  All rights reserved.
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

//! @file
//! cub::DeviceReduce provides device-wide, parallel operations for computing a reduction across a sequence of data
//! items residing within device-accessible memory.

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
#include <cub/detail/nvtx.cuh>
#include <cub/device/dispatch/dispatch_reduce.cuh>
#include <cub/device/dispatch/dispatch_reduce_by_key.cuh>
#include <cub/iterator/arg_index_input_iterator.cuh>
#include <cub/util_deprecated.cuh>
#include <cub/util_type.cuh>

#include <thrust/iterator/tabulate_output_iterator.h>


#include <iterator>
#include <limits>

CUB_NAMESPACE_BEGIN

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

  //
  PromoteToGlobalOpT promote_op;

  //
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

template <typename GlobalOffsetT>
struct accumulating_argmin_reduction_op
{
  // The current partition's offset to be factored into this partitions index
  GlobalOffsetT current_partition_offset;

  /**
   * This is a helper function that's invoked after a partition has been fully processed
   */
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE void advance(GlobalOffsetT partition_size)
  {
    current_partition_offset += partition_size;
  };

  /**
   * Unary operator called to "transform" the per-partition aggregate of the very first partition to a global
   * aggregate type (across partitions)
   */
  template <typename PerPartitionOffsetT, typename AccumT>
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE KeyValuePair<GlobalOffsetT, AccumT>
  operator()(KeyValuePair<PerPartitionOffsetT, AccumT> partition_aggregate)
  {
    return KeyValuePair<GlobalOffsetT, AccumT>{
      current_partition_offset + static_cast<GlobalOffsetT>(partition_aggregate.key), partition_aggregate.value};
  }
};

template <typename Iterator, typename OffsetItT>
class OffsetIteratorT : public thrust::iterator_adaptor<OffsetIteratorT<Iterator, OffsetItT>, Iterator>
{
public:
  using super_t = thrust::iterator_adaptor<OffsetIteratorT<Iterator, OffsetItT>, Iterator>;

  __host__ __device__ OffsetIteratorT(const Iterator& it, OffsetItT offset_it)
      : super_t(it)
      , offset_it(offset_it)
  {}

  // befriend thrust::iterator_core_access to allow it access to the private interface below
  friend class thrust::iterator_core_access;

private:
  OffsetItT offset_it;

  __host__ __device__ typename super_t::reference dereference() const
  {
    return *(this->base() + (*offset_it));
  }
};

template <typename Iterator, typename OffsetItT>
_CCCL_HOST_DEVICE OffsetIteratorT<Iterator, OffsetItT> make_offset_iterator(Iterator it, OffsetItT offset_it)
{
  return OffsetIteratorT<Iterator, OffsetItT>{it, offset_it};
}

template <typename AggregateOutIteratorT, typename IndexOutIteratorT>
struct write_arg_result_to_user_iterators_op
{
  AggregateOutIteratorT result_out_it;
  IndexOutIteratorT index_out_it;

  template <typename IndexT, typename KeyValuePairT>
  __host__ __device__ void operator()(IndexT, KeyValuePairT reduced_result)
  {
    *result_out_it = reduced_result.value;
    *index_out_it  = reduced_result.key;
  }
};

//! @rst
//! DeviceReduce provides device-wide, parallel operations for computing
//! a reduction across a sequence of data items residing within
//! device-accessible memory.
//!
//! .. image:: ../../img/reduce_logo.png
//!     :align: center
//!
//! Overview
//! ====================================
//!
//! A `reduction <http://en.wikipedia.org/wiki/Reduce_(higher-order_function)>`_
//! (or *fold*) uses a binary combining operator to compute a single aggregate
//! from a sequence of input elements.
//!
//! Usage Considerations
//! ====================================
//!
//! @cdp_class{DeviceReduce}
//!
//! Performance
//! ====================================
//!
//! @linear_performance{reduction, reduce-by-key, and run-length encode}
//!
//! @endrst
struct DeviceReduce
{
  //! @rst
  //! Computes a device-wide reduction using the specified binary ``reduction_op`` functor and initial value ``init``.
  //!
  //! - Does not support binary reduction operators that are non-commutative.
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - The range ``[d_in, d_in + num_items)`` shall not overlap ``d_out``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates a user-defined min-reduction of a
  //! device vector of ``int`` data elements.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_radix_sort.cuh>
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers for
  //!    // input and output
  //!    int          num_items;  // e.g., 7
  //!    int          *d_in;      // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int          *d_out;     // e.g., [-]
  //!    CustomMin    min_op;
  //!    int          init;       // e.g., INT_MAX
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceReduce::Reduce(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, num_items, min_op, init);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run reduction
  //!    cub::DeviceReduce::Reduce(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_in, d_out, num_items, min_op, init);
  //!
  //!    // d_out <-- [0]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam ReductionOpT
  //!   **[inferred]** Binary reduction functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam T
  //!   **[inferred]** Data element type that is convertible to the `value` type of `InputIteratorT`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of `d_in`)
  //!
  //! @param[in] reduction_op
  //!   Binary reduction functor
  //!
  //! @param[in] init
  //!   Initial value of the reduction
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename ReductionOpT, typename T, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t Reduce(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    NumItemsT num_items,
    ReductionOpT reduction_op,
    T init,
    cudaStream_t stream = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceReduce::Reduce");

    // Signed integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return DispatchReduce<InputIteratorT, OutputIteratorT, OffsetT, ReductionOpT, T>::Dispatch(
      d_temp_storage, temp_storage_bytes, d_in, d_out, static_cast<OffsetT>(num_items), reduction_op, init, stream);
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  template <typename InputIteratorT, typename OutputIteratorT, typename ReductionOpT, typename T>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED CUB_RUNTIME_FUNCTION static cudaError_t Reduce(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    int num_items,
    ReductionOpT reduction_op,
    T init,
    cudaStream_t stream,
    bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Reduce<InputIteratorT, OutputIteratorT, ReductionOpT, T>(
      d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, reduction_op, init, stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

  //! @rst
  //! Computes a device-wide sum using the addition (``+``) operator.
  //!
  //! - Uses ``0`` as the initial value of the reduction.
  //! - Does not support ``+`` operators that are non-commutative..
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - The range ``[d_in, d_in + num_items)`` shall not overlap ``d_out``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the sum-reduction of a device vector
  //! of ``int`` data elements.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/device/device_radix_sort.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for input and output
  //!    int  num_items;      // e.g., 7
  //!    int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_out;         // e.g., [-]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceReduce::Sum(
  //!      d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run sum-reduction
  //!    cub::DeviceReduce::Sum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  //!
  //!    // d_out <-- [38]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Sum(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      NumItemsT num_items,
      cudaStream_t stream = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceReduce::Sum");

    // Signed integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    // The output value type
    using OutputT = cub::detail::non_void_value_t<OutputIteratorT, cub::detail::value_t<InputIteratorT>>;

    using InitT = OutputT;

    return DispatchReduce<InputIteratorT, OutputIteratorT, OffsetT, cub::Sum, InitT>::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      static_cast<OffsetT>(num_items),
      cub::Sum(),
      InitT{}, // zero-initialize
      stream);
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  template <typename InputIteratorT, typename OutputIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED CUB_RUNTIME_FUNCTION static cudaError_t
  Sum(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      int num_items,
      cudaStream_t stream,
      bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Sum<InputIteratorT, OutputIteratorT>(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

  //! @rst
  //! Computes a device-wide minimum using the less-than (``<``) operator.
  //!
  //! - Uses ``std::numeric_limits<T>::max()`` as the initial value of the reduction.
  //! - Does not support ``<`` operators that are non-commutative.
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - The range ``[d_in, d_in + num_items)`` shall not overlap ``d_out``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the min-reduction of a device vector of ``int`` data elements.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_radix_sort.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for input and output
  //!    int  num_items;      // e.g., 7
  //!    int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_out;         // e.g., [-]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceReduce::Min(
  //!      d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run min-reduction
  //!    cub::DeviceReduce::Min(
  //!      d_temp_storage, temp_storage_bytes, d_in, d_out, num_items);
  //!
  //!    // d_out <-- [0]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Min(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      NumItemsT num_items,
      cudaStream_t stream = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceReduce::Min");

    // Signed integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    // The input value type
    using InputT = cub::detail::value_t<InputIteratorT>;

    using InitT = InputT;

    return DispatchReduce<InputIteratorT, OutputIteratorT, OffsetT, cub::Min, InitT>::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      static_cast<OffsetT>(num_items),
      cub::Min(),
      // replace with
      // std::numeric_limits<T>::max() when
      // C++11 support is more prevalent
      Traits<InitT>::Max(),
      stream);
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  template <typename InputIteratorT, typename OutputIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED CUB_RUNTIME_FUNCTION static cudaError_t
  Min(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      int num_items,
      cudaStream_t stream,
      bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Min<InputIteratorT, OutputIteratorT>(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

  //! @rst
  //! Finds the first device-wide minimum using the less-than (``<``) operator, also returning the index of that item.
  //!
  //! - The output value type of ``d_out`` is ``cub::KeyValuePair<int, T>``
  //!   (assuming the value type of ``d_in`` is ``T``)
  //!
  //!   - The minimum is written to ``d_out.value`` and its offset in the input array is written to ``d_out.key``.
  //!   - The ``{1, std::numeric_limits<T>::max()}`` tuple is produced for zero-length inputs
  //!
  //! - Does not support ``<`` operators that are non-commutative.
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - The range ``[d_in, d_in + num_items)`` shall not overlap `d_out`.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the argmin-reduction of a device vector
  //! of ``int`` data elements.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/device/device_radix_sort.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for input and output
  //!    int                      num_items;      // e.g., 7
  //!    int                      *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    KeyValuePair<int, int>   *d_argmin;      // e.g., [{-,-}]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_argmin, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run argmin-reduction
  //!    cub::DeviceReduce::ArgMin(d_temp_storage, temp_storage_bytes, d_in, d_argmin, num_items);
  //!
  //!    // d_argmin <-- [{5, 0}]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items
  //!   (of some type `T`) @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate
  //!   (having value type `cub::KeyValuePair<int, T>`) @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename AggregateOutIteratorT, typename IndexOutIteratorT, typename OffsetT>
  CUB_RUNTIME_FUNCTION static cudaError_t ArgMin(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    AggregateOutIteratorT d_result_out,
    IndexOutIteratorT d_index_out,
    OffsetT num_items,
    cudaStream_t stream = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceReduce::ArgMin");

    // Offset type used within the kernel and to index within one partition
    using PerPartitionOffsetT = int;

    // Offset type used to index within the total input in the range [d_in, d_in + num_items)
    using GlobalOffsetT = ::cuda::std::uint64_t;

    // The input type
    using InputValueT = cub::detail::value_t<InputIteratorT>;

    // The value type used for the accumulator
    using OutputAggregateT = detail::non_void_value_t<AggregateOutIteratorT, InputValueT>;

    // The per-partition accumulator and output tuple type
    using OutputTupleT = KeyValuePair<PerPartitionOffsetT, OutputAggregateT>;
    using GlobalAccumT = KeyValuePair<GlobalOffsetT, OutputAggregateT>;

    // Initial value type
    using InitT = detail::reduce::empty_problem_init_t<OutputTupleT>;

    // Helper iterator to offset the input iterator by the current partition's offset
    using constant_offset_it = cub::ConstantInputIterator<GlobalOffsetT>;

    // Wrapped input iterator to produce index-value tuples, i.e., <PerPartitionOffsetT, InputT>-tuples
    // We make sure to offset the user-provided input iterator by the current partition's offset
    using ArgIndexInputIteratorT =
      ArgIndexInputIterator<OffsetIteratorT<InputIteratorT, constant_offset_it>, PerPartitionOffsetT, InputValueT>;

    // Initial value
    InitT initial_value{OutputTupleT(1, ::cuda::std::numeric_limits<InputValueT>::max())};

    // Accumulator type that accumulates across partitions
    using global_accum_t = KeyValuePair<GlobalOffsetT, OutputAggregateT>;

    // Reduction operator type that accumulates the per-partition ArgMin-reduction results to a global aggrate
    // operator()(OutputAggregateT) -> GlobalOffsetT (invoked after first partition)
    // operator()(GlobalOffsetT, OutputAggregateT) -> GlobalOffsetT (invoked to accumulate per-partition result to
    // the global aggregate)
    using accumulating_argmin_reduction_op_t = accumulating_argmin_reduction_op<GlobalOffsetT>;
    accumulating_argmin_reduction_op_t accumulating_argmin_op_t{GlobalOffsetT{0}};

    // Iterator that "unzips" the KeyValuePair from the global aggregate and assigns key and the value to one of the two
    // output iterators, respectively
    using write_kv_pair_to_distinct_out_its_t =
      thrust::tabulate_output_iterator<write_arg_result_to_user_iterators_op<AggregateOutIteratorT, IndexOutIteratorT>>;
    write_kv_pair_to_distinct_out_its_t d_out = thrust::make_tabulate_output_iterator(
      write_arg_result_to_user_iterators_op<AggregateOutIteratorT, IndexOutIteratorT>{d_result_out, d_index_out});

    // Reduction operator type that enables accumulating per-partition results to a global reduction result
    using accumulating_transform_output_op_t =
      accumulating_transform_output_op<global_accum_t,
                                       accumulating_argmin_reduction_op_t,
                                       cub::ArgMin,
                                       write_kv_pair_to_distinct_out_its_t>;

    // The output iterator that implements the logic to accumulate per-partition result to a global aggregate and,
    // eventually, write to the user-provided output iterators
    using tabulate_output_iterator_t = thrust::tabulate_output_iterator<accumulating_transform_output_op_t>;

    void* allocations[2]       = {nullptr};
    size_t allocation_sizes[2] = {0, 2 * sizeof(GlobalAccumT)};

    // The current partition's input iterator is an ArgIndex iterator that generates indexes relative to the beginning
    // of the current partition, i.e., [0, partition_size) along with an OffsetIterator that offsets the user-provided
    // input iterator by the current partition's offset.
    ArgIndexInputIteratorT d_indexed_offset_in(make_offset_iterator(d_in, constant_offset_it{GlobalOffsetT{0}}));

    //
    accumulating_transform_output_op_t accumulating_out_op(nullptr, nullptr, false, d_out, accumulating_argmin_op_t, cub::ArgMin{});

    // Upper bound at which we want to cut the input into multiple partitions
    static constexpr PerPartitionOffsetT max_partition_size =
      ::cuda::std::numeric_limits<PerPartitionOffsetT>::max();

    // Whether the given number of items fits into a single partition
    const bool is_single_partition =
      static_cast<GlobalOffsetT>(max_partition_size) >= static_cast<GlobalOffsetT>(num_items);

    // The largest partition size ever encountered
    const auto largest_partition_size =
      is_single_partition ? static_cast<PerPartitionOffsetT>(num_items) : max_partition_size;

    // Query temporary storage requirements for per-partition reduction
    DispatchReduce<ArgIndexInputIteratorT,
                   tabulate_output_iterator_t,
                   PerPartitionOffsetT,
                   cub::ArgMin,
                   InitT,
                   OutputTupleT>::Dispatch(nullptr,
                                           allocation_sizes[0],
                                           d_indexed_offset_in,
                                           thrust::make_tabulate_output_iterator(accumulating_out_op),
                                           static_cast<PerPartitionOffsetT>(largest_partition_size),
                                           cub::ArgMin(),
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
    GlobalAccumT* d_global_aggregates = reinterpret_cast<GlobalAccumT*>(allocations[1]);

    accumulating_out_op = accumulating_transform_output_op_t{
      d_global_aggregates, (d_global_aggregates + 1), is_single_partition, d_out, accumulating_argmin_op_t, cub::ArgMin{}};

    for (GlobalOffsetT current_partition_offset = 0;
         current_partition_offset < static_cast<GlobalOffsetT>(num_items);
         current_partition_offset += static_cast<GlobalOffsetT>(max_partition_size))
    {
      const GlobalOffsetT remaining_items = (num_items - current_partition_offset);
      GlobalOffsetT current_num_items = (remaining_items < max_partition_size) ? remaining_items : max_partition_size;

      d_indexed_offset_in =
        ArgIndexInputIteratorT(make_offset_iterator(d_in, constant_offset_it{current_partition_offset}));

      error = DispatchReduce<
        ArgIndexInputIteratorT,
        tabulate_output_iterator_t,
        PerPartitionOffsetT,
        cub::ArgMin,
        InitT,
        OutputTupleT>::Dispatch(d_temp_storage,
                                temp_storage_bytes,
                                d_indexed_offset_in,
                                thrust::make_tabulate_output_iterator(accumulating_out_op),
                                static_cast<PerPartitionOffsetT>(current_num_items),
                                cub::ArgMin(),
                                initial_value,
                                stream);

      // Whether the next partition will be the last partition
      const bool next_partition_is_last =
        (remaining_items - current_num_items) <= static_cast<GlobalOffsetT>(max_partition_size);
      accumulating_out_op.advance(current_num_items, next_partition_is_last);
    }

    return cudaSuccess;
  }

  template <typename InputIteratorT, typename OutputIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t ArgMin(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    int num_items,
    cudaStream_t stream = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceReduce::ArgMin");

    // Signed integer type for global offsets
    using OffsetT = int;

    // The input type
    using InputValueT = cub::detail::value_t<InputIteratorT>;

    // The output tuple type
    using OutputTupleT = cub::detail::non_void_value_t<OutputIteratorT, KeyValuePair<OffsetT, InputValueT>>;

    using AccumT = OutputTupleT;

    using InitT = detail::reduce::empty_problem_init_t<AccumT>;

    // The output value type
    using OutputValueT = typename OutputTupleT::Value;

    // Wrapped input iterator to produce index-value <OffsetT, InputT> tuples
    using ArgIndexInputIteratorT = ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;

    ArgIndexInputIteratorT d_indexed_in(d_in);

    // Initial value
    // TODO Address https://github.com/NVIDIA/cub/issues/651
    InitT initial_value{AccumT(1, Traits<InputValueT>::Max())};

    return DispatchReduce<ArgIndexInputIteratorT, OutputIteratorT, OffsetT, cub::ArgMin, InitT, AccumT>::Dispatch(
      d_temp_storage, temp_storage_bytes, d_indexed_in, d_out, num_items, cub::ArgMin(), initial_value, stream);
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  template <typename InputIteratorT, typename OutputIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED CUB_RUNTIME_FUNCTION static cudaError_t ArgMin(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    int num_items,
    cudaStream_t stream,
    bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return ArgMin<InputIteratorT, OutputIteratorT>(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

  //! @rst
  //! Computes a device-wide maximum using the greater-than (``>``) operator.
  //!
  //! - Uses ``std::numeric_limits<T>::lowest()`` as the initial value of the reduction.
  //! - Does not support ``>`` operators that are non-commutative.
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - The range ``[d_in, d_in + num_items)`` shall not overlap ``d_out``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the max-reduction of a device vector of ``int`` data elements.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh> // or equivalently <cub/device/device_radix_sort.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for input and output
  //!    int  num_items;      // e.g., 7
  //!    int  *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    int  *d_max;         // e.g., [-]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_max, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run max-reduction
  //!    cub::DeviceReduce::Max(d_temp_storage, temp_storage_bytes, d_in, d_max, num_items);
  //!
  //!    // d_max <-- [9]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t
  Max(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      NumItemsT num_items,
      cudaStream_t stream = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceReduce::Max");

    // Signed integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    // The input value type
    using InputT = cub::detail::value_t<InputIteratorT>;

    using InitT = InputT;

    return DispatchReduce<InputIteratorT, OutputIteratorT, OffsetT, cub::Max, InitT>::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      static_cast<OffsetT>(num_items),
      cub::Max(),
      // replace with
      // std::numeric_limits<T>::lowest()
      // when C++11 support is more
      // prevalent
      Traits<InitT>::Lowest(),
      stream);
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  template <typename InputIteratorT, typename OutputIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED CUB_RUNTIME_FUNCTION static cudaError_t
  Max(void* d_temp_storage,
      size_t& temp_storage_bytes,
      InputIteratorT d_in,
      OutputIteratorT d_out,
      int num_items,
      cudaStream_t stream,
      bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return Max<InputIteratorT, OutputIteratorT>(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

  //! @rst
  //! Finds the first device-wide maximum using the greater-than (``>``)
  //! operator, also returning the index of that item
  //!
  //! - The output value type of ``d_out`` is ``cub::KeyValuePair<int, T>``
  //!   (assuming the value type of ``d_in`` is ``T``)
  //!
  //!   - The maximum is written to ``d_out.value`` and its offset in the input
  //!     array is written to ``d_out.key``.
  //!   - The ``{1, std::numeric_limits<T>::lowest()}`` tuple is produced for zero-length inputs
  //!
  //! - Does not support ``>`` operators that are non-commutative.
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - The range ``[d_in, d_in + num_items)`` shall not overlap ``d_out``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the argmax-reduction of a device vector
  //! of `int` data elements.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_reduce.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for input and output
  //!    int                      num_items;      // e.g., 7
  //!    int                      *d_in;          // e.g., [8, 6, 7, 5, 3, 0, 9]
  //!    KeyValuePair<int, int>   *d_argmax;      // e.g., [{-,-}]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceReduce::ArgMax(
  //!      d_temp_storage, temp_storage_bytes, d_in, d_argmax, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run argmax-reduction
  //!    cub::DeviceReduce::ArgMax(
  //!      d_temp_storage, temp_storage_bytes, d_in, d_argmax, num_items);
  //!
  //!    // d_argmax <-- [{6, 9}]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items (of some type `T`) @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate
  //!   (having value type `cub::KeyValuePair<int, T>`) @iterator
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of `d_in`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT, typename OutputIteratorT>
  CUB_RUNTIME_FUNCTION static cudaError_t ArgMax(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    int num_items,
    cudaStream_t stream = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceReduce::ArgMax");

    // Signed integer type for global offsets
    using OffsetT = int;

    // The input type
    using InputValueT = cub::detail::value_t<InputIteratorT>;

    // The output tuple type
    using OutputTupleT = cub::detail::non_void_value_t<OutputIteratorT, KeyValuePair<OffsetT, InputValueT>>;

    using AccumT = OutputTupleT;

    // The output value type
    using OutputValueT = typename OutputTupleT::Value;

    using InitT = detail::reduce::empty_problem_init_t<AccumT>;

    // Wrapped input iterator to produce index-value <OffsetT, InputT> tuples
    using ArgIndexInputIteratorT = ArgIndexInputIterator<InputIteratorT, OffsetT, OutputValueT>;

    ArgIndexInputIteratorT d_indexed_in(d_in);

    // Initial value
    // TODO Address https://github.com/NVIDIA/cub/issues/651
    InitT initial_value{AccumT(1, Traits<InputValueT>::Lowest())};

    return DispatchReduce<ArgIndexInputIteratorT, OutputIteratorT, OffsetT, cub::ArgMax, InitT, AccumT>::Dispatch(
      d_temp_storage, temp_storage_bytes, d_indexed_in, d_out, num_items, cub::ArgMax(), initial_value, stream);
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  template <typename InputIteratorT, typename OutputIteratorT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED CUB_RUNTIME_FUNCTION static cudaError_t ArgMax(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    int num_items,
    cudaStream_t stream,
    bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return ArgMax<InputIteratorT, OutputIteratorT>(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

  //! @rst
  //! Fuses transform and reduce operations
  //!
  //! - Does not support binary reduction operators that are non-commutative.
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - The range ``[d_in, d_in + num_items)`` shall not overlap ``d_out``.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates a user-defined min-reduction of a
  //! device vector of `int` data elements.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_reduce.cuh>
  //!
  //!    thrust::device_vector<int> in = { 1, 2, 3, 4 };
  //!    thrust::device_vector<int> out(1);
  //!
  //!    std::size_t temp_storage_bytes = 0;
  //!    std::uint8_t *d_temp_storage = nullptr;
  //!
  //!    const int init = 42;
  //!
  //!    cub::DeviceReduce::TransformReduce(
  //!      d_temp_storage,
  //!      temp_storage_bytes,
  //!      in.begin(),
  //!      out.begin(),
  //!      in.size(),
  //!      cub::Sum{},
  //!      square_t{},
  //!      init);
  //!
  //!    thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  //!    d_temp_storage = temp_storage.data().get();
  //!
  //!    cub::DeviceReduce::TransformReduce(
  //!      d_temp_storage,
  //!      temp_storage_bytes,
  //!      in.begin(),
  //!      out.begin(),
  //!      in.size(),
  //!      cub::Sum{},
  //!      square_t{},
  //!      init);
  //!
  //!    // out[0] <-- 72
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input items @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Output iterator type for recording the reduced aggregate @iterator
  //!
  //! @tparam ReductionOpT
  //!   **[inferred]** Binary reduction functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam TransformOpT
  //!   **[inferred]** Unary reduction functor type having member `auto operator()(const T &a)`
  //!
  //! @tparam T
  //!   **[inferred]** Data element type that is convertible to the `value` type of `InputIteratorT`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_in
  //!   Pointer to the input sequence of data items
  //!
  //! @param[out] d_out
  //!   Pointer to the output aggregate
  //!
  //! @param[in] num_items
  //!   Total number of input items (i.e., length of `d_in`)
  //!
  //! @param[in] reduction_op
  //!   Binary reduction functor
  //!
  //! @param[in] transform_op
  //!   Unary transform functor
  //!
  //! @param[in] init
  //!   Initial value of the reduction
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename ReductionOpT,
            typename TransformOpT,
            typename T,
            typename NumItemsT>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformReduce(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    NumItemsT num_items,
    ReductionOpT reduction_op,
    TransformOpT transform_op,
    T init,
    cudaStream_t stream = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceReduce::TransformReduce");

    using OffsetT = detail::choose_offset_t<NumItemsT>;

    return DispatchTransformReduce<InputIteratorT, OutputIteratorT, OffsetT, ReductionOpT, TransformOpT, T>::Dispatch(
      d_temp_storage,
      temp_storage_bytes,
      d_in,
      d_out,
      static_cast<OffsetT>(num_items),
      reduction_op,
      init,
      stream,
      transform_op);
  }

  //! @rst
  //! Reduces segments of values, where segments are demarcated by corresponding runs of identical keys.
  //!
  //! This operation computes segmented reductions within ``d_values_in`` using the specified binary ``reduction_op``
  //! functor. The segments are identified by "runs" of corresponding keys in `d_keys_in`, where runs are maximal
  //! ranges of consecutive, identical keys. For the *i*\ :sup:`th` run encountered, the first key of the run and
  //! the corresponding value aggregate of that run are written to ``d_unique_out[i]`` and ``d_aggregates_out[i]``,
  //! respectively. The total number of runs encountered is written to ``d_num_runs_out``.
  //!
  //! - The ``==`` equality operator is used to determine whether keys are equivalent
  //! - Provides "run-to-run" determinism for pseudo-associative reduction
  //!   (e.g., addition of floating point types) on the same GPU device.
  //!   However, results for pseudo-associative reduction may be inconsistent
  //!   from one device to a another device of a different compute-capability
  //!   because CUB can employ different tile-sizing for different architectures.
  //! - Let ``out`` be any of
  //!   ``[d_unique_out, d_unique_out + *d_num_runs_out)``
  //!   ``[d_aggregates_out, d_aggregates_out + *d_num_runs_out)``
  //!   ``d_num_runs_out``. The ranges represented by ``out`` shall not overlap
  //!   ``[d_keys_in, d_keys_in + num_items)``,
  //!   ``[d_values_in, d_values_in + num_items)`` nor ``out`` in any way.
  //! - @devicestorage
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates the segmented reduction of ``int`` values grouped by runs of
  //! associated ``int`` keys.
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_reduce.cuh>
  //!
  //!    // CustomMin functor
  //!    struct CustomMin
  //!    {
  //!        template <typename T>
  //!        __host__ __device__ __forceinline__
  //!        T operator()(const T &a, const T &b) const {
  //!            return (b < a) ? b : a;
  //!        }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    // for input and output
  //!    int          num_items;          // e.g., 8
  //!    int          *d_keys_in;         // e.g., [0, 2, 2, 9, 5, 5, 5, 8]
  //!    int          *d_values_in;       // e.g., [0, 7, 1, 6, 2, 5, 3, 4]
  //!    int          *d_unique_out;      // e.g., [-, -, -, -, -, -, -, -]
  //!    int          *d_aggregates_out;  // e.g., [-, -, -, -, -, -, -, -]
  //!    int          *d_num_runs_out;    // e.g., [-]
  //!    CustomMin    reduction_op;
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceReduce::ReduceByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_unique_out, d_values_in,
  //!      d_aggregates_out, d_num_runs_out, reduction_op, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run reduce-by-key
  //!    cub::DeviceReduce::ReduceByKey(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_keys_in, d_unique_out, d_values_in,
  //!      d_aggregates_out, d_num_runs_out, reduction_op, num_items);
  //!
  //!    // d_unique_out      <-- [0, 2, 9, 5, 8]
  //!    // d_aggregates_out  <-- [0, 1, 6, 2, 4]
  //!    // d_num_runs_out    <-- [5]
  //!
  //! @endrst
  //!
  //! @tparam KeysInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input keys @iterator
  //!
  //! @tparam UniqueOutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing unique output keys @iterator
  //!
  //! @tparam ValuesInputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input values @iterator
  //!
  //! @tparam AggregatesOutputIterator
  //!   **[inferred]** Random-access output iterator type for writing output value aggregates @iterator
  //!
  //! @tparam NumRunsOutputIteratorT
  //!   **[inferred]** Output iterator type for recording the number of runs encountered @iterator
  //!
  //! @tparam ReductionOpT
  //!   **[inferred]** Binary reduction functor type having member `T operator()(const T &a, const T &b)`
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in] d_keys_in
  //!   Pointer to the input sequence of keys
  //!
  //! @param[out] d_unique_out
  //!   Pointer to the output sequence of unique keys (one key per run)
  //!
  //! @param[in] d_values_in
  //!   Pointer to the input sequence of corresponding values
  //!
  //! @param[out] d_aggregates_out
  //!   Pointer to the output sequence of value aggregates
  //!   (one aggregate per run)
  //!
  //! @param[out] d_num_runs_out
  //!   Pointer to total number of runs encountered
  //!   (i.e., the length of `d_unique_out`)
  //!
  //! @param[in] reduction_op
  //!   Binary reduction functor
  //!
  //! @param[in] num_items
  //!   Total number of associated key+value pairs
  //!   (i.e., the length of `d_in_keys` and `d_in_values`)
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename KeysInputIteratorT,
            typename UniqueOutputIteratorT,
            typename ValuesInputIteratorT,
            typename AggregatesOutputIteratorT,
            typename NumRunsOutputIteratorT,
            typename ReductionOpT,
            typename NumItemsT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t ReduceByKey(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIteratorT d_keys_in,
    UniqueOutputIteratorT d_unique_out,
    ValuesInputIteratorT d_values_in,
    AggregatesOutputIteratorT d_aggregates_out,
    NumRunsOutputIteratorT d_num_runs_out,
    ReductionOpT reduction_op,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceReduce::ReduceByKey");

    // Signed integer type for global offsets
    using OffsetT = detail::choose_offset_t<NumItemsT>;

    // FlagT iterator type (not used)

    // Selection op (not used)

    // Default == operator
    using EqualityOp = Equality;

    return DispatchReduceByKey<
      KeysInputIteratorT,
      UniqueOutputIteratorT,
      ValuesInputIteratorT,
      AggregatesOutputIteratorT,
      NumRunsOutputIteratorT,
      EqualityOp,
      ReductionOpT,
      OffsetT>::Dispatch(d_temp_storage,
                         temp_storage_bytes,
                         d_keys_in,
                         d_unique_out,
                         d_values_in,
                         d_aggregates_out,
                         d_num_runs_out,
                         EqualityOp(),
                         reduction_op,
                         static_cast<OffsetT>(num_items),
                         stream);
  }

#ifndef DOXYGEN_SHOULD_SKIP_THIS // Do not document
  template <typename KeysInputIteratorT,
            typename UniqueOutputIteratorT,
            typename ValuesInputIteratorT,
            typename AggregatesOutputIteratorT,
            typename NumRunsOutputIteratorT,
            typename ReductionOpT>
  CUB_DETAIL_RUNTIME_DEBUG_SYNC_IS_NOT_SUPPORTED CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t ReduceByKey(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    KeysInputIteratorT d_keys_in,
    UniqueOutputIteratorT d_unique_out,
    ValuesInputIteratorT d_values_in,
    AggregatesOutputIteratorT d_aggregates_out,
    NumRunsOutputIteratorT d_num_runs_out,
    ReductionOpT reduction_op,
    int num_items,
    cudaStream_t stream,
    bool debug_synchronous)
  {
    CUB_DETAIL_RUNTIME_DEBUG_SYNC_USAGE_LOG

    return ReduceByKey<KeysInputIteratorT,
                       UniqueOutputIteratorT,
                       ValuesInputIteratorT,
                       AggregatesOutputIteratorT,
                       NumRunsOutputIteratorT,
                       ReductionOpT>(
      d_temp_storage,
      temp_storage_bytes,
      d_keys_in,
      d_unique_out,
      d_values_in,
      d_aggregates_out,
      d_num_runs_out,
      reduction_op,
      num_items,
      stream);
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS
};

CUB_NAMESPACE_END
