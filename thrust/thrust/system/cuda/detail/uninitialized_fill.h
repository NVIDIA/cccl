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
#  include <thrust/system/cuda/detail/execution_policy.h>
#  include <thrust/system/cuda/detail/fill.h>
#  include <thrust/system/cuda/detail/parallel_for.h>
#  include <thrust/system/cuda/detail/util.h>

#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__new/device_new.h>
#  include <cuda/std/__type_traits/is_trivially_assignable.h>
#  include <cuda/std/__type_traits/is_trivially_constructible.h>

THRUST_NAMESPACE_BEGIN

namespace cuda_cub
{
namespace __uninitialized_fill
{
template <class Iterator, class T>
struct functor
{
  Iterator items;
  T value;

  using value_type = thrust::detail::it_value_t<Iterator>;

  template <class Size>
  void _CCCL_DEVICE_API _CCCL_FORCEINLINE operator()(Size idx)
  {
    value_type& out = raw_reference_cast(items[idx]);
    ::new (static_cast<void*>(&out)) value_type(value);
  }
};
} // namespace __uninitialized_fill

template <class Derived, class Iterator, class Size, class T>
Iterator _CCCL_HOST_DEVICE
uninitialized_fill_n(execution_policy<Derived>& policy, Iterator first, Size count, T const& x)
{
  // if the output type is trivially constructible from the input, it has no side effect, and we can skip placement new
  // and calling a constructor. Furthermore, if assigning the input value to an output element is also trivial, there is
  // no copy constructor which could have a side effect and we can delegate to fill_n (which uses
  // cub::DeviceTransform::Fill).
  using value_t = thrust::detail::it_value_t<Iterator>;
  if constexpr (::cuda::std::is_trivially_constructible_v<value_t, T const&>
                && ::cuda::std::is_trivially_assignable_v<value_t, T const&>)
  {
    cuda_cub::fill_n(policy, first, count, x);
  }
  else
  {
    cuda_cub::parallel_for(policy, __uninitialized_fill::functor<Iterator, T>{first, x}, count);
  }
  return first + count;
}

template <class Derived, class Iterator, class T>
void _CCCL_HOST_DEVICE uninitialized_fill(execution_policy<Derived>& policy, Iterator first, Iterator last, T const& x)
{
  cuda_cub::uninitialized_fill_n(policy, first, ::cuda::std::distance(first, last), x);
}
} // namespace cuda_cub

THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
