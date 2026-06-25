// SPDX-FileCopyrightText: Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDA_COMPILATION()

#  include <thrust/system/cuda/config.h>

#  include <cub/device/device_reduce.cuh>

#  include <thrust/detail/alignment.h>
#  include <thrust/detail/temporary_array.h>
#  include <thrust/system/cuda/detail/cdp_dispatch.h>
#  include <thrust/system/cuda/detail/execution_policy.h>
#  include <thrust/system/cuda/detail/get_value.h>
#  include <thrust/system/cuda/detail/util.h>

#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN

// Forward declare generic reduce circumvent circular dependency.
template <typename DerivedPolicy, typename InputIterator, typename T, typename BinaryFunction>
T _CCCL_HOST_DEVICE
reduce(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
       InputIterator first,
       InputIterator last,
       T init,
       BinaryFunction binary_op);

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename T, typename BinaryFunction>
void _CCCL_HOST_DEVICE reduce_into(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator output,
  T init,
  BinaryFunction binary_op);

namespace cuda_cub
{
namespace detail
{
template <typename Derived, typename InputIt, typename Size, typename T, typename BinaryOp>
THRUST_RUNTIME_FUNCTION size_t get_reduce_n_temporary_storage_size(
  execution_policy<Derived>& policy, InputIt first, Size num_items, T init, BinaryOp binary_op)
{
  cudaStream_t stream = cuda_cub::stream(policy);
  cudaError_t status;

  size_t tmp_size = 0;

  THRUST_INDEX_TYPE_DISPATCH(
    status,
    cub::DeviceReduce::Reduce,
    num_items,
    (nullptr, tmp_size, first, static_cast<T*>(nullptr), num_items_fixed, binary_op, init, stream));
  cuda_cub::throw_on_error(status, "after determining reduce temporary storage size");

  return tmp_size;
}

template <typename Derived, typename InputIt, typename Size, typename T, typename BinaryOp>
THRUST_RUNTIME_FUNCTION T
reduce_n_impl(execution_policy<Derived>& policy, InputIt first, Size num_items, T init, BinaryOp binary_op)
{
  cudaStream_t stream = cuda_cub::stream(policy);
  cudaError_t status;

  // Determine temporary device storage requirements.

  size_t tmp_size = get_reduce_n_temporary_storage_size(policy, first, num_items, init, binary_op);

  // Allocate temporary storage.

  // We allocate both the temporary storage needed for the algorithm, and a `T`
  // to store the result.
  //
  // The array was dynamically allocated, so we assume that it's suitably
  // aligned for any type of data. `malloc`/`cudaMalloc`/`new`/`std::allocator`
  // make this guarantee.

  thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, tmp_size + sizeof(T));

  // Run reduction.

  T* ret_ptr    = thrust::detail::aligned_reinterpret_cast<T*>(tmp.data().get());
  void* tmp_ptr = static_cast<void*>((tmp.data() + sizeof(T)).get());
  THRUST_INDEX_TYPE_DISPATCH(
    status,
    cub::DeviceReduce::Reduce,
    num_items,
    (tmp_ptr, tmp_size, first, ret_ptr, num_items_fixed, binary_op, init, stream));
  cuda_cub::throw_on_error(status, "after reduce invocation");

  // Synchronize the stream and get the value.

  status = cuda_cub::synchronize(policy);
  cuda_cub::throw_on_error(status, "reduce failed to synchronize");
  return thrust::cuda_cub::get_value(policy, thrust::detail::aligned_reinterpret_cast<T*>(tmp.data().get()));
}

template <typename Derived, typename InputIt, typename Size, typename OutputIt, typename T, typename BinaryOp>
THRUST_RUNTIME_FUNCTION void reduce_n_into_impl(
  execution_policy<Derived>& policy, InputIt first, Size num_items, OutputIt output, T init, BinaryOp binary_op)
{
  cudaStream_t stream = cuda_cub::stream(policy);
  cudaError_t status;

  // Determine temporary device storage requirements.

  size_t tmp_size = get_reduce_n_temporary_storage_size(policy, first, num_items, init, binary_op);

  // Allocate temporary storage.

  thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, tmp_size);

  // Run reduction.

  void* tmp_ptr = thrust::raw_pointer_cast(tmp.data());
  THRUST_INDEX_TYPE_DISPATCH(
    status,
    cub::DeviceReduce::Reduce,
    num_items,
    (tmp_ptr, tmp_size, first, output, num_items_fixed, binary_op, init, stream));
  cuda_cub::throw_on_error(status, "after reduce invocation");

  // Synchronize the stream and get the value.

  status = cuda_cub::synchronize_optional(policy);
  cuda_cub::throw_on_error(status, "reduce failed to synchronize");
}
} // namespace detail

//-------------------------
// Thrust API entry points
//-------------------------

_CCCL_EXEC_CHECK_DISABLE
template <typename Derived, typename InputIt, typename Size, typename T, typename BinaryOp>
_CCCL_HOST_DEVICE T
reduce_n(execution_policy<Derived>& policy, InputIt first, Size num_items, T init, BinaryOp binary_op)
{
  THRUST_CDP_DISPATCH(
    (init = thrust::cuda_cub::detail::reduce_n_impl(policy, first, num_items, init, binary_op);),
    (init = thrust::reduce(cvt_to_seq(derived_cast(policy)), first, first + num_items, init, binary_op);));
  return init;
}

_CCCL_EXEC_CHECK_DISABLE
template <typename Derived, typename InputIt, typename Size, typename OutputIt, typename T, typename BinaryOp>
_CCCL_HOST_DEVICE void reduce_n_into(
  execution_policy<Derived>& policy, InputIt first, Size num_items, OutputIt output, T init, BinaryOp binary_op)
{
  THRUST_CDP_DISPATCH(
    (thrust::cuda_cub::detail::reduce_n_into_impl(policy, first, num_items, output, init, binary_op);),
    (thrust::reduce_into(cvt_to_seq(derived_cast(policy)), first, first + num_items, output, init, binary_op);));
}

template <class Derived, class InputIt, class T, class BinaryOp>
_CCCL_HOST_DEVICE T reduce(execution_policy<Derived>& policy, InputIt first, InputIt last, T init, BinaryOp binary_op)
{
  using size_type = thrust::detail::it_difference_t<InputIt>;
  // FIXME: Check for RA iterator.
  size_type num_items = static_cast<size_type>(::cuda::std::distance(first, last));
  return cuda_cub::reduce_n(policy, first, num_items, init, binary_op);
}

template <class Derived, class InputIt, class T>
_CCCL_HOST_DEVICE T reduce(execution_policy<Derived>& policy, InputIt first, InputIt last, T init)
{
  return cuda_cub::reduce(policy, first, last, init, ::cuda::std::plus<T>());
}

template <class Derived, class InputIt>
_CCCL_HOST_DEVICE thrust::detail::it_value_t<InputIt>
reduce(execution_policy<Derived>& policy, InputIt first, InputIt last)
{
  using value_type = thrust::detail::it_value_t<InputIt>;
  return cuda_cub::reduce(policy, first, last, value_type(0));
}

template <class Derived, class InputIt, class OutputIt, class T, class BinaryOp>
_CCCL_HOST_DEVICE void
reduce_into(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt output, T init, BinaryOp binary_op)
{
  using size_type = thrust::detail::it_difference_t<InputIt>;
  // FIXME: Check for RA iterator.
  size_type num_items = static_cast<size_type>(::cuda::std::distance(first, last));
  cuda_cub::reduce_n_into(policy, first, num_items, output, init, binary_op);
}

template <class Derived, class InputIt, class OutputIt, class T>
_CCCL_HOST_DEVICE void
reduce_into(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt output, T init)
{
  cuda_cub::reduce_into(policy, first, last, output, init, ::cuda::std::plus<T>());
}

template <class Derived, class InputIt, class OutputIt>
_CCCL_HOST_DEVICE void reduce_into(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt output)
{
  using value_type = thrust::detail::it_value_t<InputIt>;
  return cuda_cub::reduce_into(policy, first, last, output, value_type(0));
}
} // namespace cuda_cub

THRUST_NAMESPACE_END

#  include <thrust/memory.h>
#  include <thrust/reduce.h>

#endif // _CCCL_CUDA_COMPILATION()
