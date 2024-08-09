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

#include <cub/detail/nvtx.cuh>
#include <cub/device/dispatch/dispatch_transform.cuh>
#include <cub/util_namespace.cuh>

#include <cuda/std/tuple>

CUB_NAMESPACE_BEGIN

// TODO(bgruber): Add API usage examples

//! DeviceTransform provides device-wide, parallel operations for transforming elements tuple-wise from multiple input
//! streams into an output stream.
struct DeviceTransform
{
  // Many input streams, one output stream

  /// Transforms many input streams into one output stream, by applying a transformation operation on corresponding
  /// input elements and writing the result to the corresponding output element. No guarantee is given on the identity
  /// (i.e. address) of the objects passed to the call operator of the transformation operation.
  ///
  /// @param count The number of elements in each input stream.
  /// @param inputs A tuple of iterators to the input streams where count elements are read from each. The iterators'
  /// value types must be trivially relocatable.
  /// @param output An iterator to the output stream where count results are written to.
  /// @param transform_op An n-ary function object, where n is the number of input streams. The input iterators' value
  /// types must be convertible to the parameters of the function object's call operator. The return type of the call
  /// operator must be assignable to the dereferenced output iterator.
  template <typename... RandomAccessIteratorsIn, typename RandomAccessIteratorOut, typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t Transform(
    int count,
    ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
    RandomAccessIteratorOut output,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE("cub::DeviceTransform::Transform");
    return detail::transform::
      dispatch_t<false, int, ::cuda::std::tuple<RandomAccessIteratorsIn...>, RandomAccessIteratorOut, TransformOp>::
        dispatch(count, ::cuda::std::move(inputs), ::cuda::std::move(output), ::cuda::std::move(transform_op), stream);
  }

  // temp storage overload
  template <typename... RandomAccessIteratorsIn, typename RandomAccessIteratorOut, typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t Transform(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    int count,
    ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
    RandomAccessIteratorOut output,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    return Transform(
      count, ::cuda::std::move(inputs), ::cuda::std::move(output), ::cuda::std::move(transform_op), stream);
  }

  // One input stream, one output stream

  template <typename RandomAccessIteratorIn, typename RandomAccessIteratorOut, typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t Transform(
    int count,
    RandomAccessIteratorIn input,
    RandomAccessIteratorOut output,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    return Transform(
      count,
      ::cuda::std::make_tuple(::cuda::std::move(input)),
      ::cuda::std::move(output),
      ::cuda::std::move(transform_op),
      stream);
  }

  // temp storage overload
  template <typename RandomAccessIteratorIn, typename RandomAccessIteratorOut, typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t Transform(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    int count,
    RandomAccessIteratorIn input,
    RandomAccessIteratorOut output,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    return Transform(
      count,
      ::cuda::std::make_tuple(::cuda::std::move(input)),
      ::cuda::std::move(output),
      ::cuda::std::move(transform_op),
      stream);
  }

  // Many input streams, one output stream, address stable

  /// Like \ref Transform, but the objects passed to the call operator of the transformation operation are guaranteed to
  /// reside in the input streams and are never copied.
  template <typename... RandomAccessIteratorsIn, typename RandomAccessIteratorOut, typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformStableArgumentAddresses(
    int count,
    ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
    RandomAccessIteratorOut output,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    CUB_DETAIL_NVTX_RANGE_SCOPE("cub::DeviceTransform::Transform");
    return detail::transform::
      dispatch_t<true, int, ::cuda::std::tuple<RandomAccessIteratorsIn...>, RandomAccessIteratorOut, TransformOp>::
        dispatch(count, ::cuda::std::move(inputs), ::cuda::std::move(output), ::cuda::std::move(transform_op), stream);
  }

  // temp storage overload
  template <typename... RandomAccessIteratorsIn, typename RandomAccessIteratorOut, typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformStableArgumentAddresses(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    int count,
    ::cuda::std::tuple<RandomAccessIteratorsIn...> inputs,
    RandomAccessIteratorOut output,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    return TransformStableArgumentAddresses(
      count, ::cuda::std::move(inputs), ::cuda::std::move(output), ::cuda::std::move(transform_op), stream);
  }

  // One input stream, one output stream

  template <typename RandomAccessIteratorIn, typename RandomAccessIteratorOut, typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformStableArgumentAddresses(
    int count,
    RandomAccessIteratorIn inputs,
    RandomAccessIteratorOut output,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    return TransformStableArgumentAddresses(
      count,
      ::cuda::std::make_tuple(::cuda::std::move(inputs)),
      ::cuda::std::move(output),
      ::cuda::std::move(transform_op),
      stream);
  }

  // temp storage overload
  template <typename RandomAccessIteratorIn, typename RandomAccessIteratorOut, typename TransformOp>
  CUB_RUNTIME_FUNCTION static cudaError_t TransformStableArgumentAddresses(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    int count,
    RandomAccessIteratorIn inputs,
    RandomAccessIteratorOut output,
    TransformOp transform_op,
    cudaStream_t stream = nullptr)
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    return TransformStableArgumentAddresses(
      count,
      ::cuda::std::make_tuple(::cuda::std::move(inputs)),
      ::cuda::std::move(output),
      ::cuda::std::move(transform_op),
      stream);
  }
};

CUB_NAMESPACE_END
