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

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
template <class Derived, class ItemsIt, class ResultIt>
ResultIt _CCCL_HOST_DEVICE reverse_copy(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, ResultIt result);

template <class Derived, class ItemsIt>
void _CCCL_HOST_DEVICE reverse(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last);
} // namespace cuda_cub
THRUST_NAMESPACE_END

#  include <thrust/system/cuda/detail/copy.h>
#  include <thrust/system/cuda/detail/swap_ranges.h>

#  include <cuda/std/__iterator/advance.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/reverse_iterator.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
template <class Derived, class ItemsIt, class ResultIt>
ResultIt _CCCL_HOST_DEVICE reverse_copy(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last, ResultIt result)
{
  return cuda_cub::copy(policy, ::cuda::std::reverse_iterator{last}, ::cuda::std::reverse_iterator{first}, result);
}

template <class Derived, class ItemsIt>
void _CCCL_HOST_DEVICE reverse(execution_policy<Derived>& policy, ItemsIt first, ItemsIt last)
{
  using difference_type = thrust::detail::it_difference_t<ItemsIt>;

  // find the midpoint of [first,last)
  difference_type N = ::cuda::std::distance(first, last);
  ItemsIt mid(first);
  ::cuda::std::advance(mid, N / 2);

  cuda_cub::swap_ranges(policy, first, mid, ::cuda::std::make_reverse_iterator(last));
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
