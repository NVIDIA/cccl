//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FWD_ITERATOR_H
#define _CUDA___FWD_ITERATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/random.h>
#include <cuda/std/__concepts/arithmetic.h>
#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/incrementable_traits.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Tp, class _Index = ::cuda::std::ptrdiff_t>
class constant_iterator;

template <class _Tp>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto __get_wider_signed() noexcept
{
  if constexpr (sizeof(_Tp) < sizeof(int))
  {
    return ::cuda::std::type_identity<int>{};
  }
  else if constexpr (sizeof(_Tp) < sizeof(long))
  {
    return ::cuda::std::type_identity<long>{};
  }
#if _CCCL_HAS_INT128()
  else if constexpr (sizeof(_Tp) < sizeof(long long))
  {
    return ::cuda::std::type_identity<long long>{};
  }
  else // if constexpr (sizeof(_Start) < sizeof(__int128_t))
  {
    return ::cuda::std::type_identity<__int128_t>{};
  }
#else // ^^^ _CCCL_HAS_INT128() ^^^ / vvv !_CCCL_HAS_INT128() vvv
  else // if constexpr (sizeof(_Start) < sizeof(long long))
  {
    return ::cuda::std::type_identity<long long>{};
  }
#endif // _CCCL_HAS_INT128()
}

template <class _Start>
using _IotaDiffT = typename ::cuda::std::conditional_t<
  (!::cuda::std::integral<_Start> || sizeof(::cuda::std::iter_difference_t<_Start>) > sizeof(_Start)),
  ::cuda::std::type_identity<::cuda::std::iter_difference_t<_Start>>,
  decltype(::cuda::__get_wider_signed<_Start>())>::type;

#if _CCCL_HAS_CONCEPTS()
template <::cuda::std::weakly_incrementable _Start, ::cuda::std::signed_integral _DiffT = _IotaDiffT<_Start>>
  requires ::cuda::std::copyable<_Start>
#else // ^^^ _CCCL_HAS_CONCEPTS() ^^^ / vvv !_CCCL_HAS_CONCEPTS() vvv
template <class _Start,
          class _DiffT                                                             = _IotaDiffT<_Start>,
          ::cuda::std::enable_if_t<::cuda::std::weakly_incrementable<_Start>, int> = 0,
          ::cuda::std::enable_if_t<::cuda::std::copyable<_Start>, int>             = 0,
          ::cuda::std::enable_if_t<::cuda::std::signed_integral<_DiffT>, int>      = 0>
#endif // ^^^ !_CCCL_HAS_CONCEPTS() ^^^
class counting_iterator;

class discard_iterator;

template <class _Iter, class _Index = _Iter>
class permutation_iterator;

template <class _IndexType = ::cuda::std::size_t, class _Bijection = random_bijection<_IndexType>>
class shuffle_iterator;

template <class _Iter, class _Stride = ::cuda::std::iter_difference_t<_Iter>>
class strided_iterator;

template <class _Fn, class _Index = ::cuda::std::ptrdiff_t>
class tabulate_output_iterator;

template <class _InputFn, class _OutputFn, class _Iter>
class transform_input_output_iterator;

template <class _Fn, class _Iter>
class transform_iterator;

template <class _Fn, class _Iter>
class transform_output_iterator;

template <class... _Iterators>
class zip_iterator;

template <class>
inline constexpr bool __is_zip_iterator = false;

template <class... _Iterators>
inline constexpr bool __is_zip_iterator<zip_iterator<_Iterators...>> = true;

template <class _Fn>
class zip_function;

template <class>
inline constexpr bool __is_zip_function = false;

template <class _Fn>
inline constexpr bool __is_zip_function<zip_function<_Fn>> = true;

template <class _Fn, class... _Iterators>
class zip_transform_iterator;

template <class>
inline constexpr bool __is_zip_transform_iterator = false;

template <class _Fn, class... _Iterators>
inline constexpr bool __is_zip_transform_iterator<zip_transform_iterator<_Fn, _Iterators...>> = true;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___FWD_ITERATOR_H
