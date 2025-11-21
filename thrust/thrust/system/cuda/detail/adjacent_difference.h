/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
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

#  include <cub/device/device_adjacent_difference.cuh>
#  include <cub/util_math.cuh>

#  include <thrust/detail/temporary_array.h>
#  include <thrust/detail/type_traits.h>
#  include <thrust/functional.h>
#  include <thrust/system/cuda/detail/cdp_dispatch.h>
#  include <thrust/system/cuda/detail/dispatch.h>
#  include <thrust/system/cuda/detail/execution_policy.h>
#  include <thrust/system/cuda/detail/util.h>
#  include <thrust/type_traits/is_contiguous_iterator.h>
#  include <thrust/type_traits/unwrap_contiguous_iterator.h>

#  include <cuda/std/__functional/operations.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__type_traits/is_pointer.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename BinaryFunction>
_CCCL_HOST_DEVICE OutputIterator adjacent_difference(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator result,
  BinaryFunction binary_op);

namespace cuda_cub
{
namespace __adjacent_difference
{
template <cub::MayAlias AliasOpt, class InputIt, class OutputIt, class BinaryOp>
cudaError_t THRUST_RUNTIME_FUNCTION doit_step(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIt first,
  OutputIt result,
  BinaryOp binary_op,
  std::size_t num_items,
  cudaStream_t stream)
{
  if (num_items == 0)
  {
    return cudaSuccess;
  }

  constexpr cub::ReadOption read_left = cub::ReadOption::Left;

  using Dispatch32 = cub::DispatchAdjacentDifference<InputIt, OutputIt, BinaryOp, std::int32_t, AliasOpt, read_left>;
  using Dispatch64 = cub::DispatchAdjacentDifference<InputIt, OutputIt, BinaryOp, std::int64_t, AliasOpt, read_left>;

  cudaError_t status;
  THRUST_INDEX_TYPE_DISPATCH2(
    status,
    Dispatch32::Dispatch,
    Dispatch64::Dispatch,
    num_items,
    (d_temp_storage, temp_storage_bytes, first, result, num_items_fixed, binary_op, stream));
  return status;
}

template <class InputIt, class OutputIt, class BinaryOp>
cudaError_t THRUST_RUNTIME_FUNCTION doit_step(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIt first,
  OutputIt result,
  BinaryOp binary_op,
  std::size_t num_items,
  cudaStream_t stream,
  thrust::detail::integral_constant<bool, false> /* comparable */)
{
  return doit_step<cub::MayAlias::Yes>(d_temp_storage, temp_storage_bytes, first, result, binary_op, num_items, stream);
}

template <class InputIt, class OutputIt, class BinaryOp>
cudaError_t THRUST_RUNTIME_FUNCTION doit_step(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIt first,
  OutputIt result,
  BinaryOp binary_op,
  std::size_t num_items,
  cudaStream_t stream,
  thrust::detail::integral_constant<bool, true> /* comparable */)
{
  // The documentation states that pointers might be equal but can't alias in
  // any other way. That is, the distance should be equal to zero or exceed
  // `num_items`. In the latter case, we use an optimized version.
  if (first != result)
  {
    return doit_step<cub::MayAlias::No>(d_temp_storage, temp_storage_bytes, first, result, binary_op, num_items, stream);
  }

  return doit_step<cub::MayAlias::Yes>(d_temp_storage, temp_storage_bytes, first, result, binary_op, num_items, stream);
}

template <typename Derived, typename InputIt, typename OutputIt, typename BinaryOp>
OutputIt THRUST_RUNTIME_FUNCTION
adjacent_difference(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result, BinaryOp binary_op)
{
  const auto num_items     = static_cast<std::size_t>(::cuda::std::distance(first, last));
  std::size_t storage_size = 0;
  cudaStream_t stream      = cuda_cub::stream(policy);

  using UnwrapInputIt  = thrust::try_unwrap_contiguous_iterator_t<InputIt>;
  using UnwrapOutputIt = thrust::try_unwrap_contiguous_iterator_t<OutputIt>;

  using InputValueT  = thrust::detail::it_value_t<UnwrapInputIt>;
  using OutputValueT = thrust::detail::it_value_t<UnwrapOutputIt>;

  constexpr bool can_compare_iterators =
    ::cuda::std::is_pointer_v<UnwrapInputIt> && ::cuda::std::is_pointer_v<UnwrapOutputIt>
    && std::is_same_v<InputValueT, OutputValueT>;

  auto first_unwrap  = thrust::try_unwrap_contiguous_iterator(first);
  auto result_unwrap = thrust::try_unwrap_contiguous_iterator(result);

  thrust::detail::integral_constant<bool, can_compare_iterators> comparable;

  cudaError_t status =
    doit_step(nullptr, storage_size, first_unwrap, result_unwrap, binary_op, num_items, stream, comparable);
  cuda_cub::throw_on_error(status, "adjacent_difference failed on 1st step");

  // Allocate temporary storage.
  thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, storage_size);

  status = doit_step(
    static_cast<void*>(tmp.data().get()),
    storage_size,
    first_unwrap,
    result_unwrap,
    binary_op,
    num_items,
    stream,
    comparable);
  cuda_cub::throw_on_error(status, "adjacent_difference failed on 2nd step");

  status = cuda_cub::synchronize_optional(policy);
  cuda_cub::throw_on_error(status, "adjacent_difference failed to synchronize");

  return result + num_items;
}
} // namespace __adjacent_difference

//-------------------------
// Thrust API entry points
//-------------------------

_CCCL_EXEC_CHECK_DISABLE
template <class Derived, class InputIt, class OutputIt, class BinaryOp>
OutputIt _CCCL_HOST_DEVICE
adjacent_difference(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result, BinaryOp binary_op)
{
  THRUST_CDP_DISPATCH(
    (result = __adjacent_difference::adjacent_difference(policy, first, last, result, binary_op);),
    (result = thrust::adjacent_difference(cvt_to_seq(derived_cast(policy)), first, last, result, binary_op);));
  return result;
}

template <class Derived, class InputIt, class OutputIt>
OutputIt _CCCL_HOST_DEVICE
adjacent_difference(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result)
{
  using input_type = thrust::detail::it_value_t<InputIt>;
  return cuda_cub::adjacent_difference(policy, first, last, result, ::cuda::std::minus<input_type>());
}
} // namespace cuda_cub
THRUST_NAMESPACE_END

//
#  include <thrust/adjacent_difference.h>
#  include <thrust/memory.h>
#endif // _CCCL_CUDA_COMPILATION()
