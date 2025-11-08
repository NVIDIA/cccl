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

#  include <thrust/iterator/zip_iterator.h>
#  include <thrust/system/cuda/detail/parallel_for.h>
#  include <thrust/system/cuda/detail/transform.h>
#  include <thrust/type_traits/is_trivially_relocatable.h>

#  include <cuda/__functional/address_stability.h>
#  include <cuda/std/__iterator/distance.h>
#  include <cuda/std/__iterator/iter_swap.h>
#  include <cuda/std/__utility/swap.h>

THRUST_NAMESPACE_BEGIN
namespace cuda_cub
{
struct __swap_f
{
  template <typename T, typename U>
  _CCCL_HOST_DEVICE auto operator()(T t, U u) const -> tuple<T, U>
  {
    using ::cuda::std::swap;
    swap(t, u);
    return tuple{t, u};
  }
};

template <class ItemsIt1, class ItemsIt2>
struct __swap_fallback_f
{
  ItemsIt1 items1;
  ItemsIt2 items2;

  template <class Size>
  _CCCL_HOST_DEVICE void operator()(Size idx) const
  {
    ::cuda::std::iter_swap(items1 + idx, items2 + idx);
  }
};

template <class Derived, class ItemsIt1, class ItemsIt2>
_CCCL_HOST_DEVICE ItemsIt2
swap_ranges(execution_policy<Derived>& policy, ItemsIt1 first1, ItemsIt1 last1, ItemsIt2 first2)
{
  if constexpr (is_indirectly_trivially_relocate_to_v<ItemsIt1, ItemsIt2>
                && is_indirectly_trivially_relocate_to_v<ItemsIt2, ItemsIt1>)
  {
    return get<1>(
      cuda_cub::transform(
        policy, first1, last1, first2, zip_iterator{first1, first2}, ::cuda::proclaim_copyable_arguments(__swap_f{}))
        .get_iterator_tuple());
  }
  else
  {
    const auto num_items = ::cuda::std::distance(first1, last1);
    cuda_cub::parallel_for(policy, __swap_fallback_f<ItemsIt1, ItemsIt2>{first1, first2}, num_items);
    return first2 + num_items;
  }
}
} // namespace cuda_cub
THRUST_NAMESPACE_END
#endif // _CCCL_CUDA_COMPILATION()
