// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
#ifndef _CUDA___ITERATOR_ZIP_COMMON_H
#define _CUDA___ITERATOR_ZIP_COMMON_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__fwd/zip_iterator.h>
#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/iter_move.h>
#include <cuda/std/__iterator/iter_swap.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__utility/declval.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class... _Iterators>
struct __zip_iter_constraints
{
  static constexpr bool __all_forward       = (::cuda::std::__has_forward_traversal<_Iterators> && ...);
  static constexpr bool __all_bidirectional = (::cuda::std::__has_bidirectional_traversal<_Iterators> && ...);
  static constexpr bool __all_random_access = (::cuda::std::__has_random_access_traversal<_Iterators> && ...);

  static constexpr bool __all_equality_comparable = (::cuda::std::equality_comparable<_Iterators> && ...);

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
  static constexpr bool __all_three_way_comparable = (::cuda::std::three_way_comparable<_Iterators> && ...);
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

  // Our C++17 iterators sometimes do not satisfy `sized_sentinel_for` but they should all be random_access
  static constexpr bool __all_sized_sentinel =
    (::cuda::std::sized_sentinel_for<_Iterators, _Iterators> && ...) || __all_random_access;

  static constexpr bool __all_nothrow_iter_movable =
    (noexcept(::cuda::std::ranges::iter_move(::cuda::std::declval<const _Iterators&>())) && ...)
    && (::cuda::std::is_nothrow_move_constructible_v<::cuda::std::iter_rvalue_reference_t<_Iterators>> && ...);

  static constexpr bool __all_indirectly_swappable = (::cuda::std::indirectly_swappable<_Iterators> && ...);

  static constexpr bool __all_noexcept_swappable = (::cuda::std::__noexcept_swappable<_Iterators> && ...);

  static constexpr bool __all_nothrow_move_constructible =
    (::cuda::std::is_nothrow_move_constructible_v<_Iterators> && ...);

  static constexpr bool __all_default_initializable = (::cuda::std::default_initializable<_Iterators> && ...);

  static constexpr bool __all_nothrow_default_constructible =
    (::cuda::std::is_nothrow_default_constructible_v<_Iterators> && ...);
};

template <class... _Iterators>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto __get_zip_iterator_concept()
{
  using _Constraints = __zip_iter_constraints<_Iterators...>;
  if constexpr (_Constraints::__all_random_access)
  {
    return ::cuda::std::random_access_iterator_tag();
  }
  else if constexpr (_Constraints::__all_bidirectional)
  {
    return ::cuda::std::bidirectional_iterator_tag();
  }
  else if constexpr (_Constraints::__all_forward)
  {
    return ::cuda::std::forward_iterator_tag();
  }
  else
  {
    return ::cuda::std::input_iterator_tag();
  }
}

//! @note Not static functions because nvc++ sometimes has issues with class static functions in device code
struct __zip_op_star
{
  template <class... _Iterators>
  using reference = ::cuda::std::tuple<::cuda::std::iter_reference_t<_Iterators>...>;

  _CCCL_EXEC_CHECK_DISABLE
  template <class... _Iterators>
  [[nodiscard]] _CCCL_API constexpr reference<_Iterators...> operator()(const _Iterators&... __iters) const
    noexcept(noexcept(reference<_Iterators...>{*__iters...}))
  {
    return reference<_Iterators...>{*__iters...};
  }
};

struct __zip_op_increment
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class... _Iterators>
  _CCCL_API constexpr void operator()(_Iterators&... __iters) const noexcept(noexcept(((void) ++__iters, ...)))
  {
    ((void) ++__iters, ...);
  }
};

struct __zip_op_decrement
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class... _Iterators>
  _CCCL_API constexpr void operator()(_Iterators&... __iters) const noexcept(noexcept(((void) --__iters, ...)))
  {
    ((void) --__iters, ...);
  }
};

struct __zip_iter_move
{
  template <class... _Iterators>
  using __iter_move_ret = ::cuda::std::tuple<::cuda::std::iter_rvalue_reference_t<_Iterators>...>;

  _CCCL_EXEC_CHECK_DISABLE
  template <class... _Iterators>
  [[nodiscard]] _CCCL_API constexpr __iter_move_ret<_Iterators...> operator()(const _Iterators&... __iters) const
    noexcept(noexcept(__iter_move_ret<_Iterators...>{::cuda::std::ranges::iter_move(__iters)...}))
  {
    return __iter_move_ret<_Iterators...>{::cuda::std::ranges::iter_move(__iters)...};
  }
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_ZIP_COMMON_H
