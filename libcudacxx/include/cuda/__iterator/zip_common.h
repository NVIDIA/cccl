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

#include <cuda/__fwd/iterator.h>
#include <cuda/std/__algorithm/ranges_min_element.h>
#include <cuda/std/__concepts/constructible.h>
#include <cuda/std/__fwd/pair.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__iterator/concepts.h>
#include <cuda/std/__iterator/iter_move.h>
#include <cuda/std/__iterator/iter_swap.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__iterator/readable_traits.h>
#include <cuda/std/__tuple_dir/get.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_move_constructible.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/integer_sequence.h>

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
    (noexcept(::cuda::std::ranges::__iter_move_cpo{}(::cuda::std::declval<const _Iterators&>())) && ...)
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
  [[nodiscard]] _CCCL_API constexpr auto operator()(const _Iterators&... __iters) const
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
  [[nodiscard]] _CCCL_API constexpr auto operator()(const _Iterators&... __iters) const
    noexcept(noexcept(__iter_move_ret<_Iterators...>{::cuda::std::ranges::__iter_move_cpo{}(__iters)...}))
  {
    return __iter_move_ret<_Iterators...>{::cuda::std::ranges::__iter_move_cpo{}(__iters)...};
  }
};

struct __zip_op_eq
{
  // Extra level of indirection needed because GCC7 and older clang don't allow you to use
  // member functions in noexcept() clauses. We also can't use
  // __is_cpp17_nothrow_equality_comparable_v because the tuple-like type passed to these
  // functions might not implement operator==().
  template <class _Tuple1, class _Tuple2, ::cuda::std::size_t... _Indices>
  [[nodiscard]] _CCCL_API static constexpr bool
  __do_it(const _Tuple1& __tuple1, const _Tuple2& __tuple2, ::cuda::std::index_sequence<_Indices...>) noexcept(
    noexcept(((::cuda::std::get<_Indices>(__tuple1) == ::cuda::std::get<_Indices>(__tuple2)) || ...)))
  {
    return ((::cuda::std::get<_Indices>(__tuple1) == ::cuda::std::get<_Indices>(__tuple2)) || ...);
  }

  template <class _Tuple1, class _Tuple2, ::cuda::std::size_t... _Indices>
  [[nodiscard]] _CCCL_API constexpr bool
  operator()(const _Tuple1& __tuple1, const _Tuple2& __tuple2, ::cuda::std::index_sequence<_Indices...> __seq) const
    noexcept(noexcept(::cuda::__zip_op_eq::__do_it(__tuple1, __tuple2, __seq)))
  {
    return ::cuda::__zip_op_eq::__do_it(__tuple1, __tuple2, __seq);
  }

  template <class _Tuple1, class _Tuple2>
  [[nodiscard]] _CCCL_API constexpr bool operator()(const _Tuple1& __tuple1, const _Tuple2& __tuple2) const
    noexcept(noexcept(::cuda::__zip_op_eq::__do_it(
      __tuple1,
      __tuple2,
      ::cuda::std::make_index_sequence<::cuda::std::tuple_size_v<::cuda::std::remove_cvref_t<_Tuple1>>>{})))
  {
    return ::cuda::__zip_op_eq::__do_it(
      __tuple1,
      __tuple2,
      ::cuda::std::make_index_sequence<::cuda::std::tuple_size_v<::cuda::std::remove_cvref_t<_Tuple1>>>{});
  }
};

template <class _Tp, class _Up>
inline constexpr bool __nothrow_distance =
  noexcept(::cuda::std::declval<const _Tp&>() - ::cuda::std::declval<const _Up&>());

template <class _Diff>
struct __zip_op_minus
{
  struct __op_comp_abs
  {
    // abs in cstdlib is not constexpr
    _CCCL_EXEC_CHECK_DISABLE
    [[nodiscard]] _CCCL_API static constexpr _Diff __abs(_Diff __t) noexcept(noexcept(__t < 0 ? -__t : __t))
    {
      return __t < 0 ? -__t : __t;
    }

    _CCCL_EXEC_CHECK_DISABLE
    [[nodiscard]] _CCCL_API constexpr bool operator()(const _Diff& __x, const _Diff& __y) const
      noexcept(noexcept(__op_comp_abs::__abs(__x) < __op_comp_abs::__abs(__y)))
    {
      return __op_comp_abs::__abs(__x) < __op_comp_abs::__abs(__y);
    }
  };

  // Extra level of indirection needed because GCC7 and older clang don't allow you to use
  // member functions in noexcept() clauses.
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tuple1, class _Tuple2, ::cuda::std::size_t _Zero, ::cuda::std::size_t... _Indices>
  [[nodiscard]] _CCCL_API static constexpr _Diff
  __do_it(const _Tuple1& __tuple1, const _Tuple2& __tuple2, ::cuda::std::index_sequence<_Zero, _Indices...>) noexcept(
    __nothrow_distance<::cuda::std::tuple_element_t<_Zero, _Tuple1>, ::cuda::std::tuple_element_t<_Zero, _Tuple2>>
    && (__nothrow_distance<::cuda::std::tuple_element_t<_Indices, _Tuple1>,
                           ::cuda::std::tuple_element_t<_Indices, _Tuple2>>
        && ...))
  {
    const _Diff __first = ::cuda::std::get<0>(__tuple1) - ::cuda::std::get<0>(__tuple2);
    if (__first == 0)
    {
      return __first;
    }

    const _Diff __temp[] = {__first, ::cuda::std::get<_Indices>(__tuple1) - ::cuda::std::get<_Indices>(__tuple2)...};
    return *::cuda::std::ranges::__min_element_cpo{}(__temp, __op_comp_abs{});
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tuple1, class _Tuple2, ::cuda::std::size_t... _Indices>
  [[nodiscard]] _CCCL_API constexpr _Diff
  operator()(const _Tuple1& __tuple1, const _Tuple2& __tuple2, ::cuda::std::index_sequence<_Indices...> __seq) const
    noexcept(noexcept(__zip_op_minus::__do_it(__tuple1, __tuple2, __seq)))
  {
    return __zip_op_minus::__do_it(__tuple1, __tuple2, __seq);
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tuple1, class _Tuple2>
  [[nodiscard]] _CCCL_API constexpr _Diff operator()(const _Tuple1& __tuple1, const _Tuple2& __tuple2) const
    noexcept(noexcept(__zip_op_minus::__do_it(
      __tuple1,
      __tuple2,
      ::cuda::std::make_index_sequence<::cuda::std::tuple_size_v<::cuda::std::remove_cvref_t<_Tuple1>>>{})))
  {
    return __zip_op_minus::__do_it(
      __tuple1,
      __tuple2,
      ::cuda::std::make_index_sequence<::cuda::std::tuple_size_v<::cuda::std::remove_cvref_t<_Tuple1>>>{});
  }
};

// We need this to make proxy iterators work because those might not have a working `iter_value_t`
template <class _Iter, class = void>
struct __zip_maybe_proxy_helper
{
  using reference  = decltype(*::cuda::std::declval<_Iter>());
  using value_type = ::cuda::std::remove_reference_t<reference>;
};

template <class _Iter>
struct __zip_maybe_proxy_helper<_Iter, ::cuda::std::void_t<::cuda::std::iter_value_t<_Iter>>>
{
  using reference  = ::cuda::std::iter_reference_t<_Iter>;
  using value_type = ::cuda::std::iter_value_t<_Iter>;
};

template <class _Iter>
using __zip_maybe_proxy_reference_t = typename __zip_maybe_proxy_helper<_Iter>::reference;
template <class _Iter>
using __zip_maybe_proxy_value_type_t = typename __zip_maybe_proxy_helper<_Iter>::value_type;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ITERATOR_ZIP_COMMON_H
