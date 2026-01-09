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
#  include <thrust/detail/internal_functional.h>
#  include <thrust/system/cuda/detail/transform.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
template <class Derived, class Iterator, class T>
void _CCCL_HOST_DEVICE
replace(execution_policy<Derived>& policy, Iterator first, Iterator last, T const& old_value, T const& new_value)
{
  cuda_cub::transform_if(
    policy,
    first,
    last,
    first,
    CUB_NS_QUALIFIER::detail::__return_constant<T>{new_value},
    thrust::detail::equal_to_value<T>{old_value});
}

template <class Derived, class Iterator, class Predicate, class T>
void _CCCL_HOST_DEVICE
replace_if(execution_policy<Derived>& policy, Iterator first, Iterator last, Predicate pred, T const& new_value)
{
  cuda_cub::transform_if(policy, first, last, first, CUB_NS_QUALIFIER::detail::__return_constant<T>{new_value}, pred);
}

template <class Derived, class Iterator, class StencilIt, class Predicate, class T>
void _CCCL_HOST_DEVICE replace_if(
  execution_policy<Derived>& policy,
  Iterator first,
  Iterator last,
  StencilIt stencil,
  Predicate pred,
  T const& new_value)
{
  cuda_cub::transform_if(
    policy, first, last, stencil, first, CUB_NS_QUALIFIER::detail::__return_constant<T>{new_value}, pred);
}

template <class Derived, class InputIt, class OutputIt, class Predicate, class T>
OutputIt _CCCL_HOST_DEVICE replace_copy_if(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  OutputIt result,
  Predicate predicate,
  T const& new_value)
{
  using output_type    = thrust::detail::it_value_t<OutputIt>;
  using new_value_if_t = thrust::detail::new_value_if_f<Predicate, T, output_type>;
  return cuda_cub::transform(policy, first, last, result, new_value_if_t{predicate, new_value});
}

template <class Derived, class InputIt, class StencilIt, class OutputIt, class Predicate, class T>
OutputIt _CCCL_HOST_DEVICE replace_copy_if(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  StencilIt stencil,
  OutputIt result,
  Predicate predicate,
  T const& new_value)
{
  using output_type    = thrust::detail::it_value_t<OutputIt>;
  using new_value_if_t = thrust::detail::new_value_if_f<Predicate, T, output_type>;
  return cuda_cub::transform(policy, first, last, stencil, result, new_value_if_t{predicate, new_value});
}

template <class Derived, class InputIt, class OutputIt, class T>
OutputIt _CCCL_HOST_DEVICE replace_copy(
  execution_policy<Derived>& policy,
  InputIt first,
  InputIt last,
  OutputIt result,
  T const& old_value,
  T const& new_value)
{
  return cuda_cub::replace_copy_if(policy, first, last, result, thrust::detail::equal_to_value<T>{old_value}, new_value);
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
