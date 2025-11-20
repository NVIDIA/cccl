// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD_EXPERIMENTAL___SIMD_UTILITY_H
#define _CUDA_STD_EXPERIMENTAL___SIMD_UTILITY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/experimental/__simd/config.h>
#include <cuda/std/limits>

#if _LIBCUDACXX_EXPERIMENTAL_SIMD_ENABLED

#  include <cuda/std/__cstddef/size_t.h>
#  include <cuda/std/__type_traits/is_arithmetic.h>
#  include <cuda/std/__type_traits/is_const.h>
#  include <cuda/std/__type_traits/is_constant_evaluated.h>
#  include <cuda/std/__type_traits/is_convertible.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__type_traits/is_unsigned.h>
#  include <cuda/std/__type_traits/is_volatile.h>
#  include <cuda/std/__type_traits/void_t.h>
#  include <cuda/std/__utility/declval.h>
#  include <cuda/std/__utility/integer_sequence.h>
#  include <cuda/std/cstdint>
#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace experimental
{
inline namespace parallelism_v2
{

template <class _Tp>
inline constexpr bool __is_vectorizable_v =
  is_arithmetic_v<_Tp> && !is_const_v<_Tp> && !is_volatile_v<_Tp> && !is_same_v<_Tp, bool>;

template <class _From, class _To, class = void>
inline constexpr bool __is_non_narrowing_convertible_v = false;

template <class _From, class _To>
inline constexpr bool
  __is_non_narrowing_convertible_v<_From, _To, ::cuda::std::void_t<decltype(_To{::cuda::std::declval<_From>()})>> = true;

template <class _Tp, class _Up>
inline constexpr bool __can_broadcast_v =
  (__is_vectorizable_v<_Up> && __is_non_narrowing_convertible_v<_Up, _Tp>) ||
  (!__is_vectorizable_v<_Up> && is_convertible_v<_Up, _Tp>) || is_same_v<_Up, int> ||
  (is_same_v<_Up, unsigned int> && is_unsigned_v<_Tp>);

template <class _Tp, class _Generator, size_t _Idx, class = void>
inline constexpr bool __is_well_formed = false;

template <class _Tp, class _Generator, size_t _Idx>
inline constexpr bool
  __is_well_formed<_Tp,
                   _Generator,
                   _Idx,
                   ::cuda::std::void_t<decltype(::cuda::std::declval<_Generator>()(
                     ::cuda::std::integral_constant<size_t, _Idx>())))> =
    __can_broadcast_v<_Tp,
                      decltype(::cuda::std::declval<_Generator>()(::cuda::std::integral_constant<size_t, _Idx>()))>;

template <class _Tp, class _Generator, size_t... _Idxes>
_CCCL_HIDE_FROM_ABI constexpr bool __can_generate(::cuda::std::index_sequence<_Idxes...>)
{
  return (true && ... && __is_well_formed<_Tp, _Generator, _Idxes>);
}

template <class _Tp, class _Generator, size_t _Size>
inline constexpr bool __can_generate_v =
  experimental::__can_generate<_Tp, _Generator>(::cuda::std::make_index_sequence<_Size>());

} // namespace parallelism_v2
} // namespace experimental

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _LIBCUDACXX_EXPERIMENTAL_SIMD_ENABLED

#endif // _CUDA_STD_EXPERIMENTAL___SIMD_UTILITY_H

