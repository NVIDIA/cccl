// SPDX-FileCopyrightText: Copyright (c) 2011-2021, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

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
#include <cub/detail/env_dispatch.cuh>
#include <cub/detail/type_traits.cuh>
#include <cub/device/dispatch/dispatch_adjacent_difference.cuh>
#include <cub/util_namespace.cuh>

#include <cuda/__functional/call_or.h>
#include <cuda/__stream/get_stream.h>
#include <cuda/std/__execution/env.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

//! @rst
//! DeviceAdjacentDifference provides device-wide, parallel operations for
//! computing the differences of adjacent elements residing within
//! device-accessible memory.
//!
//! Overview
//! ++++++++++++++++++++++++++
//!
//! - DeviceAdjacentDifference calculates the differences of adjacent elements in
//!   d_input. Because the binary operation could be noncommutative, there
//!   are two sets of methods. Methods named SubtractLeft subtract left element
//!   ``*(i - 1)`` of input sequence from current element ``*i``.
//!   Methods named ``SubtractRight`` subtract current element ``*i`` from the
//!   right one ``*(i + 1)``:
//!
//!   .. code-block:: c++
//!
//!      int *d_values; // [1, 2, 3, 4]
//!      //...
//!      int *d_subtract_left_result  <-- [  1,  1,  1,  1 ]
//!      int *d_subtract_right_result <-- [ -1, -1, -1,  4 ]
//!
//! - For SubtractLeft, if the left element is out of bounds, the iterator is
//!   assigned to ``*(result + (i - first))`` without modification.
//! - For SubtractRight, if the right element is out of bounds, the iterator is
//!   assigned to ``*(result + (i - first))`` without modification.
//!
//! Snippet
//! ++++++++++++++++++++++++++
//!
//! The code snippet below illustrates how to use ``DeviceAdjacentDifference`` to
//! compute the left difference between adjacent elements.
//!
//! .. code-block:: c++
//!
//!    #include <cub/cub.cuh>
//!    // or equivalently <cub/device/device_adjacent_difference.cuh>
//!
//!    // Declare, allocate, and initialize device-accessible pointers
//!    int  num_items;       // e.g., 8
//!    int  *d_values;       // e.g., [1, 2, 1, 2, 1, 2, 1, 2]
//!    //...
//!
//!    // Determine temporary device storage requirements
//!    void     *d_temp_storage = nullptr;
//!    size_t   temp_storage_bytes = 0;
//!
//!    cub::DeviceAdjacentDifference::SubtractLeft(
//!      d_temp_storage, temp_storage_bytes, d_values, num_items);
//!
//!    // Allocate temporary storage
//!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
//!
//!    // Run operation
//!    cub::DeviceAdjacentDifference::SubtractLeft(
//!      d_temp_storage, temp_storage_bytes, d_values, num_items);
//!
//!    // d_values <-- [1, 1, -1, 1, -1, 1, -1, 1]
//!
//! @endrst
struct DeviceAdjacentDifference
{
  //! @rst
  //! Subtracts the left element of each adjacent pair of elements residing within device-accessible memory
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! - Calculates the differences of adjacent elements in ``d_input``.
  //!   That is, ``*d_input`` is assigned to ``*d_output``, and, for each iterator ``i`` in the
  //!   range ``[d_input + 1, d_input + num_items)``, the result of
  //!   ``difference_op(*i, *(i - 1))`` is assigned to ``*(d_output + (i - d_input))``.
  //! - Note that the behavior is undefined if the input and output ranges
  //!   overlap in any way.
  //!
  //! Snippet
  //! ++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates how to use ``DeviceAdjacentDifference``
  //! to compute the difference between adjacent elements.
  //!
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_adjacent_difference.cuh>
  //!
  //!    struct CustomDifference
  //!    {
  //!      template <typename DataType>
  //!      __host__ DataType operator()(DataType &lhs, DataType &rhs)
  //!      {
  //!        return lhs - rhs;
  //!      }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    int  num_items;      // e.g., 8
  //!    int  *d_input;       // e.g., [1, 2, 1, 2, 1, 2, 1, 2]
  //!    int  *d_output;
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!
  //!    cub::DeviceAdjacentDifference::SubtractLeftCopy(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_input, d_output,
  //!      num_items, CustomDifference());
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run operation
  //!    cub::DeviceAdjacentDifference::SubtractLeftCopy(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_input, d_output,
  //!      num_items, CustomDifference());
  //!
  //!    // d_input  <-- [1, 2, 1, 2, 1, 2, 1, 2]
  //!    // d_output <-- [1, 1, -1, 1, -1, 1, -1, 1]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   @rst
  //!   is a model of `Input Iterator <https://en.cppreference.com/w/cpp/iterator/input_iterator>`_,
  //!   and ``x`` and ``y`` are objects of ``InputIteratorT``'s ``value_type``, then
  //!   ``x - y`` is defined, and ``InputIteratorT``'s ``value_type`` is convertible to
  //!   a type in ``OutputIteratorT``'s set of ``value_types``, and the return type
  //!   of ``x - y`` is convertible to a type in ``OutputIteratorT``'s set of
  //!   ``value_types``.
  //!   @endrst
  //!
  //! @tparam OutputIteratorT
  //!   @rst
  //!   is a model of `Output Iterator <https://en.cppreference.com/w/cpp/iterator/output_iterator>`_.
  //!   @endrst
  //!
  //! @tparam DifferenceOpT
  //!   Its `result_type` is convertible to a type in `OutputIteratorT`'s set of `value_types`.
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
  //! @param[in] d_input
  //!   Beginning of the input sequence
  //!
  //! @param[out] d_output
  //!   Beginning of the output sequence
  //!
  //! @param[in] num_items
  //!   Number of items in the input sequence
  //!
  //! @param[in] difference_op
  //!   The binary function used to compute differences
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename DifferenceOpT = ::cuda::std::minus<>,
            typename NumItemsT     = uint32_t>
  static CUB_RUNTIME_FUNCTION cudaError_t SubtractLeftCopy(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_input,
    OutputIteratorT d_output,
    NumItemsT num_items,
    DifferenceOpT difference_op = {},
    cudaStream_t stream         = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceAdjacentDifference::SubtractLeftCopy");
    using OffsetT = detail::choose_offset_t<NumItemsT>;
    return detail::adjacent_difference::dispatch<MayAlias::No, ReadOption::Left>(
      d_temp_storage, temp_storage_bytes, d_input, d_output, static_cast<OffsetT>(num_items), difference_op, stream);
  }

  //! @rst
  //! Subtracts the left element of each adjacent pair of elements residing within device-accessible memory.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Calculates the differences of adjacent elements in ``d_input``. That is, for
  //! each iterator ``i`` in the range ``[d_input + 1, d_input + num_items)``, the
  //! result of ``difference_op(*i, *(i - 1))`` is assigned to
  //! ``*(d_input + (i - d_input))``.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates how to use ``DeviceAdjacentDifference``
  //! to compute the difference between adjacent elements.
  //!
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_adjacent_difference.cuh>
  //!
  //!    struct CustomDifference
  //!    {
  //!      template <typename DataType>
  //!      __host__ DataType operator()(DataType &lhs, DataType &rhs)
  //!      {
  //!        return lhs - rhs;
  //!      }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    int  num_items;     // e.g., 8
  //!    int  *d_data;       // e.g., [1, 2, 1, 2, 1, 2, 1, 2]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void     *d_temp_storage = nullptr;
  //!    size_t   temp_storage_bytes = 0;
  //!    cub::DeviceAdjacentDifference::SubtractLeft(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, num_items, CustomDifference());
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run operation
  //!    cub::DeviceAdjacentDifference::SubtractLeft(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_data, num_items, CustomDifference());
  //!
  //!    // d_data <-- [1, 1, -1, 1, -1, 1, -1, 1]
  //!
  //! @endrst
  //!
  //! @tparam RandomAccessIteratorT
  //!   @rst
  //!   is a model of `Random Access Iterator <https://en.cppreference.com/w/cpp/iterator/random_access_iterator>`_,
  //!   ``RandomAccessIteratorT`` is mutable. If ``x`` and ``y`` are objects of
  //!   ``RandomAccessIteratorT``'s ``value_type``, and ``x - y`` is defined, then the
  //!   return type of ``x - y`` should be convertible to a type in
  //!   ``RandomAccessIteratorT``'s set of ``value_types``.
  //!   @endrst
  //!
  //! @tparam DifferenceOpT
  //!   Its `result_type` is convertible to a type in `RandomAccessIteratorT`'s
  //!   set of `value_types`.
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of `num_items`
  //!
  //! @param[in] d_temp_storage
  //!   Device-accessible allocation of temporary storage. When `nullptr`, the
  //!   required allocation size is written to `temp_storage_bytes` and no work
  //!   is done.
  //!
  //! @param[in,out] temp_storage_bytes
  //!   Reference to size in bytes of `d_temp_storage` allocation
  //!
  //! @param[in,out] d_input
  //!   Beginning of the input sequence and the result
  //!
  //! @param[in] num_items
  //!   Number of items in the input sequence
  //!
  //! @param[in] difference_op
  //!   The binary function used to compute differences
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename RandomAccessIteratorT, typename DifferenceOpT = ::cuda::std::minus<>, typename NumItemsT = uint32_t>
  static CUB_RUNTIME_FUNCTION cudaError_t SubtractLeft(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RandomAccessIteratorT d_input,
    NumItemsT num_items,
    DifferenceOpT difference_op = {},
    cudaStream_t stream         = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceAdjacentDifference::SubtractLeft");
    using OffsetT = detail::choose_offset_t<NumItemsT>;
    return detail::adjacent_difference::dispatch<MayAlias::Yes, ReadOption::Left>(
      d_temp_storage, temp_storage_bytes, d_input, d_input, static_cast<OffsetT>(num_items), difference_op, stream);
  }

  //! @rst
  //! Subtracts the right element of each adjacent pair of elements residing within device-accessible memory.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! - Calculates the right differences of adjacent elements in ``d_input``.
  //!   That is, ``*(d_input + num_items - 1)`` is assigned to
  //!   ``*(d_output + num_items - 1)``, and, for each iterator ``i`` in the range
  //!   ``[d_input, d_input + num_items - 1)``, the result of
  //!   ``difference_op(*i, *(i + 1))`` is assigned to
  //!   ``*(d_output + (i - d_input))``.
  //! - Note that the behavior is undefined if the input and output ranges
  //!   overlap in any way.
  //!
  //! Snippet
  //! ++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates how to use ``DeviceAdjacentDifference``
  //! to compute the difference between adjacent elements.
  //!
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_adjacent_difference.cuh>
  //!
  //!    struct CustomDifference
  //!    {
  //!      template <typename DataType>
  //!      __host__ DataType operator()(DataType &lhs, DataType &rhs)
  //!      {
  //!        return lhs - rhs;
  //!      }
  //!    };
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    int  num_items;     // e.g., 8
  //!    int  *d_input;      // e.g., [1, 2, 1, 2, 1, 2, 1, 2]
  //!    int  *d_output;
  //!    ..
  //!
  //!    // Determine temporary device storage requirements
  //!    void *d_temp_storage = nullptr;
  //!    size_t temp_storage_bytes = 0;
  //!    cub::DeviceAdjacentDifference::SubtractRightCopy(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_input, d_output, num_items, CustomDifference());
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run operation
  //!    cub::DeviceAdjacentDifference::SubtractRightCopy(
  //!      d_temp_storage, temp_storage_bytes,
  //!      d_input, d_output, num_items, CustomDifference());
  //!
  //!    // d_input <-- [1, 2, 1, 2, 1, 2, 1, 2]
  //!    // d_data  <-- [-1, 1, -1, 1, -1, 1, -1, 2]
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   @rst
  //!   is a model of `Input Iterator <https://en.cppreference.com/w/cpp/iterator/input_iterator>`_,
  //!   and ``x`` and ``y`` are objects of ``InputIteratorT``'s ``value_type``, then
  //!   ``x - y`` is defined, and ``InputIteratorT``'s ``value_type`` is convertible to
  //!   a type in ``OutputIteratorT``'s set of ``value_types``, and the return type
  //!   of ``x - y`` is convertible to a type in ``OutputIteratorT``'s set of
  //!   ``value_types``.
  //!   @endrst
  //!
  //! @tparam OutputIteratorT
  //!   @rst
  //!   is a model of `Output Iterator <https://en.cppreference.com/w/cpp/iterator/output_iterator>`_.
  //!   @endrst
  //!
  //! @tparam DifferenceOpT
  //!   Its `result_type` is convertible to a type in `RandomAccessIteratorT`'s
  //!   set of `value_types`.
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
  //! @param[in] d_input
  //!   Beginning of the input sequence
  //!
  //! @param[out] d_output
  //!   Beginning of the output sequence
  //!
  //! @param[in] num_items
  //!   Number of items in the input sequence
  //!
  //! @param[in] difference_op
  //!   The binary function used to compute differences.
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename DifferenceOpT = ::cuda::std::minus<>,
            typename NumItemsT     = uint32_t>
  static CUB_RUNTIME_FUNCTION cudaError_t SubtractRightCopy(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_input,
    OutputIteratorT d_output,
    NumItemsT num_items,
    DifferenceOpT difference_op = {},
    cudaStream_t stream         = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceAdjacentDifference::SubtractRightCopy");
    using OffsetT = detail::choose_offset_t<NumItemsT>;
    return detail::adjacent_difference::dispatch<MayAlias::No, ReadOption::Right>(
      d_temp_storage, temp_storage_bytes, d_input, d_output, static_cast<OffsetT>(num_items), difference_op, stream);
  }

  //! @rst
  //! Subtracts the right element of each adjacent pair of elements residing within device-accessible memory.
  //!
  //! .. versionadded:: 2.2.0
  //!    First appears in CUDA Toolkit 12.3.
  //!
  //! Overview
  //! ++++++++++++++++++++++++++
  //!
  //! Calculates the right differences of adjacent elements in ``d_input``.
  //! That is, for each iterator ``i`` in the range
  //! ``[d_input, d_input + num_items - 1)``, the result of
  //! ``difference_op(*i, *(i + 1))`` is assigned to ``*(d_input + (i - d_input))``.
  //!
  //! Snippet
  //! ++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates how to use ``DeviceAdjacentDifference``
  //! to compute the difference between adjacent elements.
  //!
  //!
  //! .. code-block:: c++
  //!
  //!    #include <cub/cub.cuh>
  //!    // or equivalently <cub/device/device_adjacent_difference.cuh>
  //!
  //!    // Declare, allocate, and initialize device-accessible pointers
  //!    int  num_items;    // e.g., 8
  //!    int  *d_data;      // e.g., [1, 2, 1, 2, 1, 2, 1, 2]
  //!    ...
  //!
  //!    // Determine temporary device storage requirements
  //!    void *d_temp_storage = nullptr;
  //!    size_t temp_storage_bytes = 0;
  //!    cub::DeviceAdjacentDifference::SubtractRight(
  //!      d_temp_storage, temp_storage_bytes, d_data, num_items);
  //!
  //!    // Allocate temporary storage
  //!    cudaMalloc(&d_temp_storage, temp_storage_bytes);
  //!
  //!    // Run operation
  //!    cub::DeviceAdjacentDifference::SubtractRight(
  //!      d_temp_storage, temp_storage_bytes, d_data, num_items);
  //!
  //!    // d_data  <-- [-1, 1, -1, 1, -1, 1, -1, 2]
  //!
  //! @endrst
  //!
  //! @tparam RandomAccessIteratorT
  //!   @rst
  //!   is a model of `Random Access Iterator <https://en.cppreference.com/w/cpp/iterator/random_access_iterator>`_,
  //!   ``RandomAccessIteratorT`` is mutable. If ``x`` and ``y`` are objects of
  //!   ``RandomAccessIteratorT``'s `value_type`, and ``x - y`` is defined, then the
  //!   return type of ``x - y`` should be convertible to a type in
  //!   ``RandomAccessIteratorT``'s set of ``value_types``.
  //!   @endrst
  //!
  //! @tparam DifferenceOpT
  //!   Its `result_type` is convertible to a type in `RandomAccessIteratorT`'s
  //!   set of `value_types`.
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
  //! @param[in,out] d_input
  //!   Beginning of the input sequence
  //!
  //! @param[in] num_items
  //!   Number of items in the input sequence
  //!
  //! @param[in] difference_op
  //!   The binary function used to compute differences
  //!
  //! @param[in] stream
  //!   @rst
  //!   **[optional]** CUDA stream to launch kernels within. Default is stream\ :sub:`0`.
  //!   @endrst
  template <typename RandomAccessIteratorT, typename DifferenceOpT = ::cuda::std::minus<>, typename NumItemsT = uint32_t>
  static CUB_RUNTIME_FUNCTION cudaError_t SubtractRight(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    RandomAccessIteratorT d_input,
    NumItemsT num_items,
    DifferenceOpT difference_op = {},
    cudaStream_t stream         = nullptr)
  {
    _CCCL_NVTX_RANGE_SCOPE_IF(d_temp_storage, "cub::DeviceAdjacentDifference::SubtractRight");
    using OffsetT = detail::choose_offset_t<NumItemsT>;
    return detail::adjacent_difference::dispatch<MayAlias::Yes, ReadOption::Right>(
      d_temp_storage, temp_storage_bytes, d_input, d_input, static_cast<OffsetT>(num_items), difference_op, stream);
  }

  //! @rst
  //! Subtracts the left element of each adjacent pair of elements residing within device-accessible memory.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! This is an environment-based API that allows customization of:
  //!
  //! - Stream: Query via ``cuda::get_stream``
  //!
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! - Calculates the differences of adjacent elements in ``d_input``.
  //!   That is, ``*d_input`` is assigned to ``*d_output``, and, for each iterator ``i`` in the
  //!   range ``[d_input + 1, d_input + num_items)``, the result of
  //!   ``difference_op(*i, *(i - 1))`` is assigned to ``*(d_output + (i - d_input))``.
  //! - Note that the behavior is undefined if the input and output ranges
  //!   overlap in any way.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates how to use ``SubtractLeftCopy`` with a custom stream
  //! via an environment.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_adjacent_difference_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin subtract-left-copy-env-stream
  //!     :end-before: example-end subtract-left-copy-env-stream
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input elements @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing output elements @iterator
  //!
  //! @tparam DifferenceOpT
  //!   **[inferred]** Binary function object type used to compute differences
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!   Supports customization of stream via ``cuda::get_stream``.
  //!
  //! @param[in] d_input
  //!   Beginning of the input sequence
  //!
  //! @param[out] d_output
  //!   Beginning of the output sequence
  //!
  //! @param[in] num_items
  //!   Number of items in the input sequence
  //!
  //! @param[in] difference_op
  //!   The binary function used to compute differences
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename DifferenceOpT,
            typename NumItemsT,
            typename EnvT = ::cuda::std::execution::env<>,
            ::cuda::std::enable_if_t<!::cuda::std::is_same_v<InputIteratorT, void*>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t SubtractLeftCopy(
    InputIteratorT d_input, OutputIteratorT d_output, NumItemsT num_items, DifferenceOpT difference_op, EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceAdjacentDifference::SubtractLeftCopy");

    using OffsetT                 = detail::choose_offset_t<NumItemsT>;
    using default_policy_selector = detail::adjacent_difference::policy_selector_from_types<InputIteratorT, false>;

    return detail::dispatch_with_env_and_tuning<default_policy_selector>(
      env, [&](auto policy_selector, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
        return detail::adjacent_difference::dispatch<MayAlias::No, ReadOption::Left>(
          d_temp_storage,
          temp_storage_bytes,
          d_input,
          d_output,
          static_cast<OffsetT>(num_items),
          difference_op,
          stream,
          policy_selector);
      });
  }

  //! @rst
  //! Subtracts the left element of each adjacent pair of elements in-place.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! This is an environment-based API that allows customization of:
  //!
  //! - Stream: Query via ``cuda::get_stream``
  //!
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Calculates the differences of adjacent elements in ``d_input``. That is, for
  //! each iterator ``i`` in the range ``[d_input + 1, d_input + num_items)``, the
  //! result of ``difference_op(*i, *(i - 1))`` is assigned to
  //! ``*(d_input + (i - d_input))``.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates how to use ``SubtractLeft`` with a custom stream
  //! via an environment.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_adjacent_difference_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin subtract-left-env-stream
  //!     :end-before: example-end subtract-left-env-stream
  //!
  //! @endrst
  //!
  //! @tparam RandomAccessIteratorT
  //!   **[inferred]** Random-access iterator type for reading and writing elements @iterator
  //!
  //! @tparam DifferenceOpT
  //!   **[inferred]** Binary function object type used to compute differences
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!   Supports customization of stream via ``cuda::get_stream``.
  //!
  //! @param[in,out] d_input
  //!   Beginning of the input sequence and the result
  //!
  //! @param[in] num_items
  //!   Number of items in the input sequence
  //!
  //! @param[in] difference_op
  //!   The binary function used to compute differences
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename RandomAccessIteratorT,
            typename DifferenceOpT,
            typename NumItemsT,
            typename EnvT = ::cuda::std::execution::env<>,
            ::cuda::std::enable_if_t<!::cuda::std::is_same_v<RandomAccessIteratorT, void*>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  SubtractLeft(RandomAccessIteratorT d_input, NumItemsT num_items, DifferenceOpT difference_op, EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceAdjacentDifference::SubtractLeft");

    using OffsetT = detail::choose_offset_t<NumItemsT>;
    using default_policy_selector =
      detail::adjacent_difference::policy_selector_from_types<RandomAccessIteratorT, true>;

    return detail::dispatch_with_env_and_tuning<default_policy_selector>(
      env, [&](auto policy_selector, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
        return detail::adjacent_difference::dispatch<MayAlias::Yes, ReadOption::Left>(
          d_temp_storage,
          temp_storage_bytes,
          d_input,
          d_input,
          static_cast<OffsetT>(num_items),
          difference_op,
          stream,
          policy_selector);
      });
  }

  //! @rst
  //! Subtracts the right element of each adjacent pair of elements residing within device-accessible memory.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! This is an environment-based API that allows customization of:
  //!
  //! - Stream: Query via ``cuda::get_stream``
  //!
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! - Calculates the right differences of adjacent elements in ``d_input``.
  //!   That is, ``*(d_input + num_items - 1)`` is assigned to
  //!   ``*(d_output + num_items - 1)``, and, for each iterator ``i`` in the range
  //!   ``[d_input, d_input + num_items - 1)``, the result of
  //!   ``difference_op(*i, *(i + 1))`` is assigned to
  //!   ``*(d_output + (i - d_input))``.
  //! - Note that the behavior is undefined if the input and output ranges
  //!   overlap in any way.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates how to use ``SubtractRightCopy`` with a custom stream
  //! via an environment.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_adjacent_difference_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin subtract-right-copy-env-stream
  //!     :end-before: example-end subtract-right-copy-env-stream
  //!
  //! @endrst
  //!
  //! @tparam InputIteratorT
  //!   **[inferred]** Random-access input iterator type for reading input elements @iterator
  //!
  //! @tparam OutputIteratorT
  //!   **[inferred]** Random-access output iterator type for writing output elements @iterator
  //!
  //! @tparam DifferenceOpT
  //!   **[inferred]** Binary function object type used to compute differences
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!   Supports customization of stream via ``cuda::get_stream``.
  //!
  //! @param[in] d_input
  //!   Beginning of the input sequence
  //!
  //! @param[out] d_output
  //!   Beginning of the output sequence
  //!
  //! @param[in] num_items
  //!   Number of items in the input sequence
  //!
  //! @param[in] difference_op
  //!   The binary function used to compute differences
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename InputIteratorT,
            typename OutputIteratorT,
            typename DifferenceOpT,
            typename NumItemsT,
            typename EnvT = ::cuda::std::execution::env<>,
            ::cuda::std::enable_if_t<!::cuda::std::is_same_v<InputIteratorT, void*>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t SubtractRightCopy(
    InputIteratorT d_input, OutputIteratorT d_output, NumItemsT num_items, DifferenceOpT difference_op, EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceAdjacentDifference::SubtractRightCopy");

    using OffsetT                 = detail::choose_offset_t<NumItemsT>;
    using default_policy_selector = detail::adjacent_difference::policy_selector_from_types<InputIteratorT, false>;

    return detail::dispatch_with_env_and_tuning<default_policy_selector>(
      env, [&](auto policy_selector, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
        return detail::adjacent_difference::dispatch<MayAlias::No, ReadOption::Right>(
          d_temp_storage,
          temp_storage_bytes,
          d_input,
          d_output,
          static_cast<OffsetT>(num_items),
          difference_op,
          stream,
          policy_selector);
      });
  }

  //! @rst
  //! Subtracts the right element of each adjacent pair of elements in-place.
  //!
  //! .. versionadded:: 3.4.0
  //!    First appears in CUDA Toolkit 13.4.
  //!
  //! This is an environment-based API that allows customization of:
  //!
  //! - Stream: Query via ``cuda::get_stream``
  //!
  //! Overview
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! Calculates the right differences of adjacent elements in ``d_input``.
  //! That is, for each iterator ``i`` in the range
  //! ``[d_input, d_input + num_items - 1)``, the result of
  //! ``difference_op(*i, *(i + 1))`` is assigned to ``*(d_input + (i - d_input))``.
  //!
  //! Snippet
  //! +++++++++++++++++++++++++++++++++++++++++++++
  //!
  //! The code snippet below illustrates how to use ``SubtractRight`` with a custom stream
  //! via an environment.
  //!
  //! .. literalinclude:: ../../../cub/test/catch2_test_device_adjacent_difference_env_api.cu
  //!     :language: c++
  //!     :dedent:
  //!     :start-after: example-begin subtract-right-env-stream
  //!     :end-before: example-end subtract-right-env-stream
  //!
  //! @endrst
  //!
  //! @tparam RandomAccessIteratorT
  //!   **[inferred]** Random-access iterator type for reading and writing elements @iterator
  //!
  //! @tparam DifferenceOpT
  //!   **[inferred]** Binary function object type used to compute differences
  //!
  //! @tparam NumItemsT
  //!   **[inferred]** Type of num_items
  //!
  //! @tparam EnvT
  //!   **[inferred]** Execution environment type. Default is ``cuda::std::execution::env<>``.
  //!   Supports customization of stream via ``cuda::get_stream``.
  //!
  //! @param[in,out] d_input
  //!   Beginning of the input sequence
  //!
  //! @param[in] num_items
  //!   Number of items in the input sequence
  //!
  //! @param[in] difference_op
  //!   The binary function used to compute differences
  //!
  //! @param[in] env
  //!   @rst
  //!   **[optional]** Execution environment. Default is ``cuda::std::execution::env{}``.
  //!   @endrst
  template <typename RandomAccessIteratorT,
            typename DifferenceOpT,
            typename NumItemsT,
            typename EnvT = ::cuda::std::execution::env<>,
            ::cuda::std::enable_if_t<!::cuda::std::is_same_v<RandomAccessIteratorT, void*>, int> = 0>
  [[nodiscard]] CUB_RUNTIME_FUNCTION static cudaError_t
  SubtractRight(RandomAccessIteratorT d_input, NumItemsT num_items, DifferenceOpT difference_op, EnvT env = {})
  {
    _CCCL_NVTX_RANGE_SCOPE("cub::DeviceAdjacentDifference::SubtractRight");

    using OffsetT = detail::choose_offset_t<NumItemsT>;
    using default_policy_selector =
      detail::adjacent_difference::policy_selector_from_types<RandomAccessIteratorT, true>;

    return detail::dispatch_with_env_and_tuning<default_policy_selector>(
      env, [&](auto policy_selector, void* d_temp_storage, size_t& temp_storage_bytes, cudaStream_t stream) {
        return detail::adjacent_difference::dispatch<MayAlias::Yes, ReadOption::Right>(
          d_temp_storage,
          temp_storage_bytes,
          d_input,
          d_input,
          static_cast<OffsetT>(num_items),
          difference_op,
          stream,
          policy_selector);
      });
  }
};

CUB_NAMESPACE_END
