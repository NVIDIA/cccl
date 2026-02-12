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
#  include <cuda/std/__numeric/gcd_lcm.h>
#  include <cuda/std/__utility/integer_sequence.h>
#  include <cuda/std/cstdint>

#  include <cuda/experimental/__copy/cute/utils.cuh>

#  include <cute/layout.hpp>
//
#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
enum class __tile_order
{
  __stride,
  __logical,
};

template <::cuda::std::size_t _Rank>
[[nodiscard]] _CCCL_HOST_API constexpr bool __same_stride_order(
  ::cuda::std::array<::cuda::std::size_t, _Rank> __shapes_a,
  ::cuda::std::array<::cuda::std::int64_t, _Rank> __strides_a,
  ::cuda::std::array<::cuda::std::size_t, _Rank> __shapes_b,
  ::cuda::std::array<::cuda::std::int64_t, _Rank> __strides_b) noexcept
{
  const auto __orders_a = ::cuda::experimental::__sort_by_stride_layout<_Rank>(__shapes_a, __strides_a);
  const auto __orders_b = ::cuda::experimental::__sort_by_stride_layout<_Rank>(__shapes_b, __strides_b);

  ::cuda::std::size_t __i = 0;
  ::cuda::std::size_t __j = 0;
  while (__i < _Rank && __j < _Rank)
  {
    while (__i < _Rank && __shapes_a[__i] == 1)
    {
      ++__i;
    }
    while (__j < _Rank && __shapes_b[__j] == 1)
    {
      ++__j;
    }
    if (__i == _Rank || __j == _Rank)
    {
      break;
    }
    if (__orders_a[__i] != __orders_b[__j])
    {
      return false;
    }
    ++__i;
    ++__j;
  }

  while (__i < _Rank && __shapes_a[__i] == 1)
  {
    ++__i;
  }
  while (__j < _Rank && __shapes_b[__j] == 1)
  {
    ++__j;
  }
  return __i == _Rank && __j == _Rank;
}

template <class _ShapeA, class _StrideA, class _ShapeB, class _StrideB>
[[nodiscard]] _CCCL_HOST_API constexpr bool
__has_same_stride_order(const ::cute::Layout<_ShapeA, _StrideA>& __layout_a,
                        const ::cute::Layout<_ShapeB, _StrideB>& __layout_b) noexcept
{
  using ::cuda::std::int64_t;
  using ::cuda::std::size_t;

  constexpr auto __rank = __rank_v<_ShapeA>;
  static_assert(__rank == __rank_v<_ShapeB>, "The ranks of the layouts must be the same");
  constexpr ::cuda::std::make_index_sequence<__rank> __rank_seq{};

  ::cuda::std::array<size_t, __rank> __shapes_a;
  ::cuda::std::array<int64_t, __rank> __strides_a;
  ::cuda::std::array<size_t, __rank> __shapes_b;
  ::cuda::std::array<int64_t, __rank> __strides_b;
  ::cuda::experimental::__init_layout(__layout_a.shape(), __layout_a.stride(), __shapes_a, __strides_a, __rank_seq);
  ::cuda::experimental::__init_layout(__layout_b.shape(), __layout_b.stride(), __shapes_b, __strides_b, __rank_seq);

  return ::cuda::experimental::__same_stride_order<__rank>(__shapes_a, __strides_a, __shapes_b, __strides_b);
}

/**
 * @brief Computes the maximal common contiguous size between two layouts.
 *
 * @param[in] __layout_a The first layout.
 * @param[in] __layout_b The second layout.
 * @return The maximal common contiguous size.
 */
template <class _ShapeA, class _StrideA, class _ShapeB, class _StrideB>
[[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::size_t __max_common_contiguous_size(
  const ::cute::Layout<_ShapeA, _StrideA>& __layout_a,
  const ::cute::Layout<_ShapeB, _StrideB>& __layout_b,
  __tile_order __order = __tile_order::__stride) noexcept
{
  using ::cuda::std::int64_t;
  using ::cuda::std::size_t;
  constexpr auto __rank = __rank_v<_ShapeA>;
  static_assert(__rank == __rank_v<_ShapeB>, "The ranks of the layouts must be the same");
  if constexpr (__rank == 0)
  {
    return int64_t{1};
  }
  else
  {
    constexpr ::cuda::std::make_index_sequence<__rank> __rank_seq{};
    ::cuda::std::array<size_t, __rank> __shapes_a;
    ::cuda::std::array<int64_t, __rank> __strides_a;
    ::cuda::std::array<size_t, __rank> __shapes_b;
    ::cuda::std::array<int64_t, __rank> __strides_b;
    ::cuda::experimental::__init_layout(__layout_a.shape(), __layout_a.stride(), __shapes_a, __strides_a, __rank_seq);
    ::cuda::experimental::__init_layout(__layout_b.shape(), __layout_b.stride(), __shapes_b, __strides_b, __rank_seq);
    if (__order == __tile_order::__stride)
    {
      const auto __orders_a = ::cuda::experimental::__sort_by_stride_layout<__rank>(__shapes_a, __strides_a);
      const auto __orders_b = ::cuda::experimental::__sort_by_stride_layout<__rank>(__shapes_b, __strides_b);
      size_t __curr_a       = 1;
      size_t __curr_b       = 1;
      size_t __i            = 0;
      size_t __j            = 0;
      while (__i < __rank && __j < __rank)
      {
        const auto __dim_a = __orders_a[__i];
        const auto __dim_b = __orders_b[__j];
        // Both layouts must visit the same logical dimension and remain contiguous.
        if (__dim_a != __dim_b //
            || (__strides_a[__i] != __curr_a && __shapes_a[__i] != 1)
            || (__strides_b[__j] != __curr_b && __shapes_b[__j] != 1))
        {
          break;
        }
        __curr_a *= __shapes_a[__i];
        __curr_b *= __shapes_b[__j];
        ++__i;
        ++__j;
      }
      return ::cuda::std::gcd(__curr_a, __curr_b);
    }

    size_t __curr_a = 1;
    size_t __curr_b = 1;
    for (int __i = __rank - 1; __i >= 0; --__i)
    {
      if ((__strides_a[__i] != static_cast<int64_t>(__curr_a) && __shapes_a[__i] != 1)
          || (__strides_b[__i] != static_cast<int64_t>(__curr_b) && __shapes_b[__i] != 1))
      {
        break;
      }
      __curr_a *= __shapes_a[__i];
      __curr_b *= __shapes_b[__i];
    }
    return ::cuda::std::gcd(__curr_a, __curr_b);
  }
}

/**
 * @brief  Computes the maximal common layout between two layouts. The layouts may have different ranks.
 *
 * Runtime version of the CuTe `max_common_layout`
 *
 * @return A stride-1 layout whose size is the minimum of the contiguous sizes of both layouts.
 */
// template <class _ShapeA, class _StrideA, class _ShapeB, class _StrideB>
//[[nodiscard]] _CCCL_HOST_API constexpr auto __max_common_layout(
//   const ::cute::Layout<_ShapeA, _StrideA>& __layout_a, const ::cute::Layout<_ShapeB, _StrideB>& __layout_b)
//{
//   //  if constexpr (::cute::is_static<_StrideA>::value && ::cute::is_static<_StrideB>::value)
//   //  {
//   //    return ::cute::max_common_layout(__layout_a, __layout_b);
//   //  }
//   //  else
//   //  {
//   return ::cute::make_layout(::cuda::experimental::__max_common_contiguous_size(__layout_a, __layout_b));
//   //  }
// }
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_CUTE_MAX_COMMON_LAYOUT_H
