//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_CUTE_MAX_COMMON_LAYOUT_H
#define __CUDAX_COPY_CUTE_MAX_COMMON_LAYOUT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/array>
#  include <cuda/std/cstdint>

#  include <cuda/experimental/__copy/cute/utils.cuh>

#  include <cute/layout.hpp>
#  include <cute/tensor_impl.hpp>
//
#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
/**
 * @brief Runtime version of the CuTe expression
 *        `common_layout =coalesce(composition(layoutA, right_inverse(layoutB)))`.
 *        `return size(layout<0>(common_layout))`.
 *
 * @par Algorithm
 * 1. Sort both layouts by stride.
 * 2. Iterate over both layouts in lockstep.
 * 3. Accumulate the product of shapes when the contiguous dimensions match, exit otherwise.
 *
 * @return The number of contiguous elements of the maximal common layout.
 */
template <class _ShapeA, class _StrideA, class _ShapeB, class _StrideB>
[[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::int64_t __max_common_contiguos_size(
  const ::cute::Layout<_ShapeA, _StrideA>& __layout_a, const ::cute::Layout<_ShapeB, _StrideB>& __layout_b) noexcept
{
  using ::cuda::std::int64_t;
  using ::cuda::std::size_t;
  constexpr size_t __rank = __rank_v<_ShapeA>;
  static_assert(__rank == __rank_v<_ShapeB>);
  if constexpr (__rank == 0)
  {
    return int64_t{1};
  }
  else
  {
    constexpr ::cuda::std::make_index_sequence<__rank> __rank_seq{};
    ::cuda::std::array<int64_t, __rank> __shapes_a;
    ::cuda::std::array<int64_t, __rank> __strides_a;
    ::cuda::std::array<int64_t, __rank> __order_a;
    ::cuda::std::array<int64_t, __rank> __shapes_b;
    ::cuda::std::array<int64_t, __rank> __strides_b;
    ::cuda::std::array<int64_t, __rank> __order_b;
    ::cuda::experimental::__init_and_sort_layout(
      __layout_a.shape(), __layout_a.stride(), __shapes_a, __strides_a, __order_a, __rank_seq);
    ::cuda::experimental::__init_and_sort_layout(
      __layout_b.shape(), __layout_b.stride(), __shapes_b, __strides_b, __order_b, __rank_seq);
    int64_t __curr_a = 1;
    int64_t __curr_b = 1;
    int64_t __common = 1;
    size_t __i       = 0;
    size_t __j       = 0;
    while (__i < __rank && __j < __rank)
    {
      while (__i < __rank && __shapes_a[__order_a[__i]] == 1) // skip size-1 modes in A
      {
        ++__i;
      }
      while (__j < __rank && __shapes_b[__order_b[__j]] == 1) // skip size-1 modes in B
      {
        ++__j;
      }
      if (__i >= __rank || __j >= __rank)
      {
        break;
      }
      const auto __dim_a = __order_a[__i];
      const auto __dim_b = __order_b[__j];
      // Both must refer to the same original dimension and be contiguous
      if (__dim_a != __dim_b || __strides_a[__dim_a] != __curr_a || __strides_b[__dim_b] != __curr_b)
      {
        break;
      }
      __common *= __shapes_a[__dim_a];
      __curr_a *= __shapes_a[__dim_a];
      __curr_b *= __shapes_b[__dim_b];
      ++__i;
      ++__j;
    }
    return __common;
  }
}

/**
 * @brief Runtime version of the CuTe `max_common_layout`
 *
 * Computes the maximal common layout between two layouts.
 *
 */
template <class _ShapeA, class _StrideA, class _ShapeB, class _StrideB>
[[nodiscard]] _CCCL_HOST_API constexpr auto __max_common_layout(
  const ::cute::Layout<_ShapeA, _StrideA>& __layout_a, const ::cute::Layout<_ShapeB, _StrideB>& __layout_b) noexcept
{
  if constexpr (::cute::is_static<_StrideA>::value && ::cute::is_static<_StrideB>::value)
  {
    return ::cute::max_common_layout(__layout_a, __layout_b);
  }
  else
  {
    return ::cute::make_layout(::cuda::experimental::__max_common_contiguos_size(__layout_a, __layout_b));
  }
}
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_CUTE_MAX_COMMON_LAYOUT_H
