//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_COPY_CUTE_UTILS_H
#define __CUDAX_COPY_CUTE_UTILS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__algorithm/stable_sort.h>

#  include <cute/layout.hpp>
//
#  include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
/**
 * @brief Converts a value from CuTe shape or stride type to its runtime equivalent.
 *
 * If @p __value is a compile-time static type (e.g., @c cute::C<N>), returns the underlying integral constant.
 */
template <class _Tp>
[[nodiscard]] _CCCL_HOST_API constexpr auto __to_runtime_value(_Tp __value) noexcept
{
  if constexpr (::cute::is_static<_Tp>::value)
  {
    return _Tp::value;
  }
  else
  {
    return __value;
  }
}

/**
 * @brief Converts a @c cuda::std::array to a @c cute::tuple
 */
template <class _Tp, ::cuda::std::size_t _N, ::cuda::std::size_t... _Is>
[[nodiscard]] _CCCL_HOST_API constexpr auto
__to_cute_tuple(const ::cuda::std::array<_Tp, _N>& __values, ::cuda::std::index_sequence<_Is...>) noexcept
{
  return ::cute::make_tuple(__values[_Is]...);
}

template <class _Shape, class _Stride, ::cuda::std::size_t _Rank, ::cuda::std::size_t... _Is>
_CCCL_HOST_API constexpr void __init_layout(
  const _Shape& __shapes_tuple,
  const _Stride& __strides_tuple,
  ::cuda::std::array<::cuda::std::int64_t, _Rank>& __shapes,
  ::cuda::std::array<::cuda::std::int64_t, _Rank>& __strides,
  ::cuda::std::index_sequence<_Is...>) noexcept
{
  ((__shapes[_Is] = ::cuda::experimental::__to_runtime_value(::cute::get<_Is>(__shapes_tuple))), ...);
  ((__strides[_Is] = ::cuda::experimental::__to_runtime_value(::cute::get<_Is>(__strides_tuple))), ...);
}

/**
 * @brief Extracts shape, stride, and order information from CuTe tuples into plain arrays.
 */
template <class _Shape, class _Stride, ::cuda::std::size_t _Rank, ::cuda::std::size_t... _Is>
_CCCL_HOST_API constexpr void __init_and_sort_layout(
  const _Shape& __shapes_tuple,
  const _Stride& __strides_tuple,
  ::cuda::std::array<::cuda::std::int64_t, _Rank>& __shapes,
  ::cuda::std::array<::cuda::std::int64_t, _Rank>& __strides,
  ::cuda::std::array<::cuda::std::int64_t, _Rank>& __orders,
  ::cuda::std::index_sequence<_Is...> __rank_seq) noexcept
{
  __init_layout(__shapes_tuple, __strides_tuple, __shapes, __strides, __rank_seq);
  ((__orders[_Is] = _Is), ...);
  // Sort by strides
  ::cuda::std::stable_sort(__orders.begin(), __orders.end(), [&](auto __a, auto __b) {
    return __strides[__a] < __strides[__b];
  });
}

template <class _Shape>
_CCCL_API constexpr auto __rank_error()
{
  static_assert(::cuda::std::__always_false_v<_Shape>, "rank assertion failed");
  return 0;
}

template <class _Shape>
constexpr auto __rank_v = __rank_error<_Shape>();

template <class... _Values>
constexpr auto __rank_v<::cute::tuple<_Values...>> = sizeof...(_Values);
} // namespace cuda::experimental

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC)
#endif // __CUDAX_COPY_CUTE_UTILS_H
