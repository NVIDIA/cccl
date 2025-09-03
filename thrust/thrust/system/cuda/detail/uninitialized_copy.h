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

#if _CCCL_HAS_CUDA_COMPILER()
#  include <thrust/system/cuda/detail/execution_policy.h>
#  include <thrust/system/cuda/detail/parallel_for.h>
#  include <thrust/system/cuda/detail/transform.h>
#  include <thrust/system/cuda/detail/util.h>
#  include <thrust/type_traits/is_trivially_relocatable.h>

#  include <cuda/std/iterator>

THRUST_NAMESPACE_BEGIN

namespace cuda_cub
{
// forward declare and not include thrust/system/cuda/detail/copy.h to avoid a circular dependency
template <class System, class InputIterator, class Size, class OutputIterator>
OutputIterator _CCCL_HOST_DEVICE
copy_n(execution_policy<System>& system, InputIterator first, Size n, OutputIterator result);

namespace __uninitialized_copy
{
template <class InputIt, class OutputIt>
struct functor
{
  InputIt input;
  OutputIt output;

  using InputType  = thrust::detail::it_value_t<InputIt>;
  using OutputType = thrust::detail::it_value_t<OutputIt>;

  template <class Size>
  void THRUST_DEVICE_FUNCTION operator()(Size idx)
  {
    InputType const& in = raw_reference_cast(input[idx]);
    OutputType& out     = raw_reference_cast(output[idx]);

#  if _CCCL_CUDA_COMPILER(CLANG)
    // XXX unsafe, but clang is seemngly unable to call in-place new
    out = in;
#  else
    ::new (static_cast<void*>(&out)) OutputType(in);
#  endif
  }
};
} // namespace __uninitialized_copy

template <class Derived, class InputIt, class Size, class OutputIt>
OutputIt _CCCL_HOST_DEVICE
uninitialized_copy_n(execution_policy<Derived>& policy, InputIt first, Size count, OutputIt result)
{
  // if the output type is trivially constructible from the input, it has no side effect, and we can skip placement new
  // and calling a constructor. So we can delegate to copy_n.
  using ctor_arg_t = thrust::detail::raw_reference_t<::cuda::std::iter_reference_t<InputIt>>;
  using output_t   = thrust::detail::it_value_t<OutputIt>;
  if constexpr (::cuda::std::is_trivially_constructible_v<output_t, ctor_arg_t>)
  {
    cuda_cub::copy_n(policy, first, count, result);
  }
  else
  {
    cuda_cub::parallel_for(policy, __uninitialized_copy::functor<InputIt, OutputIt>{first, result}, count);
  }

  return result + count;
}

template <class Derived, class InputIt, class OutputIt>
OutputIt _CCCL_HOST_DEVICE
uninitialized_copy(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt result)
{
  return cuda_cub::uninitialized_copy_n(policy, first, ::cuda::std::distance(first, last), result);
}

} // namespace cuda_cub

THRUST_NAMESPACE_END
#endif
