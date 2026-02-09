//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_CUTE_COMPLEMENT_H
#define __CUDAX_COPY_CUTE_COMPLEMENT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__algorithm/fill.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/array>
#  include <cuda/std/cstdint>

#  include <cuda/experimental/__copy/cute/coalesce_and_filter.cuh>
#  include <cuda/experimental/__copy/cute/utils.cuh>

#  include <cute/layout.hpp>
//
#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
/**
 * @brief Runtime equivalent of CuTe's `complement` for layouts with dynamic strides.
 *
 *  Produces a layout that covers all elements in the codomain up to the given size.
 *  Example: codomain(4:2)   -> {0, 2, 4, 6}
 *           complement(4:2) -> (2:1) -> {0, 1}
 *           logical_product(input (4:2), complement (2:1)) -> (4:2,2:1) -> {0, 1, 2, 3, 4, 5, 6, 7}
 *
 * @par Algorithm
 * 1. Filter the layout: flatten and remove stride-0 / size-1 modes.
 * 2. Sort dimensions by stride (ascending).
 * 3. Iterate over dimensions in stride-sorted order and fill the gap dimensions.
 *
 * @param __layout  The layout to compute the complement of.
 * @param __codomain_size The codomain size up to which the complement is computed.
 */
template <class _Shape, class _Stride>
_CCCL_HOST_API auto
__complement_dynamic(const ::cute::Layout<_Shape, _Stride>& __layout, ::cuda::std::int64_t __codomain_size) noexcept
{
  using ::cuda::std::int64_t;
  const auto __flat         = ::cuda::experimental::__filter_dynamic(__layout);
  const auto __flat_shape   = __flat.shape();
  const auto __flat_stride  = __flat.stride();
  constexpr auto __rank     = __rank_v<_Shape>;
  constexpr auto __out_rank = __rank + 1; // +1 for the cotarget remainder
  ::cuda::std::array<int64_t, __out_rank> __result_shapes;
  ::cuda::std::array<int64_t, __out_rank> __result_strides{};
  ::cuda::std::fill(__result_shapes.begin(), __result_shapes.end(), int64_t{1});
  if constexpr (__rank > 0)
  {
    constexpr ::cuda::std::make_index_sequence<__rank> __rank_seq{};
    ::cuda::std::array<int64_t, __rank> __shapes{};
    ::cuda::std::array<int64_t, __rank> __strides{};
    ::cuda::std::array<int64_t, __rank> __orders{};
    ::cuda::experimental::__init_and_sort_layout(__flat_shape, __flat_stride, __shapes, __strides, __orders, __rank_seq);
    int64_t __accumulated = 1;
    for (::cuda::std::size_t __i = 0; __i < __rank; ++__i)
    {
      const auto __idx      = __orders[__i];
      __result_shapes[__i]  = __strides[__idx] / __accumulated;
      __result_strides[__i] = __accumulated;
      __accumulated         = __strides[__idx] * __shapes[__idx];
    }
    __result_shapes[__rank]  = __codomain_size / __accumulated;
    __result_strides[__rank] = __accumulated;
  }
  else
  {
    __result_shapes[0]  = __codomain_size;
    __result_strides[0] = int64_t{1};
  }
  constexpr ::cuda::std::make_index_sequence<__out_rank> __out_seq{};
  const auto __s = ::cuda::experimental::__to_cute_tuple(__result_shapes, __out_seq);
  const auto __d = ::cuda::experimental::__to_cute_tuple(__result_strides, __out_seq);
  return ::cuda::experimental::__coalesce_dynamic(::cute::make_layout(__s, __d));
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_CUTE_COMPLEMENT_H
