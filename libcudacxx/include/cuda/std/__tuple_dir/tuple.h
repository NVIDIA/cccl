//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TUPLE_TUPLE_H
#define _CUDA_STD___TUPLE_TUPLE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/get.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__memory/allocator_arg_t.h>
#include <cuda/std/__tuple_dir/tuple_constraints.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_indices.h>
#include <cuda/std/__tuple_dir/tuple_leaf.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__tuple_dir/tuple_types.h>
#include <cuda/std/__type_traits/is_nothrow_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/swap.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class... _Tp>
class _CCCL_TYPE_VISIBILITY_DEFAULT tuple
{
private:
  using _BaseT = __tuple_impl<__make_tuple_indices_t<sizeof...(_Tp)>, _Tp...>;

  _BaseT __base_;

public:
  template <size_t _Ip>
  _CCCL_API constexpr tuple_element_t<_Ip, tuple>& __get_impl() & noexcept
  {
    using type _CCCL_NODEBUG_ALIAS = tuple_element_t<_Ip, tuple>;
    return static_cast<__tuple_leaf<_Ip, type>&>(__base_).get();
  }

  template <size_t _Ip>
  _CCCL_API constexpr const tuple_element_t<_Ip, tuple>& __get_impl() const& noexcept
  {
    using type _CCCL_NODEBUG_ALIAS = tuple_element_t<_Ip, tuple>;
    return static_cast<const __tuple_leaf<_Ip, type>&>(__base_).get();
  }

  template <size_t _Ip>
  _CCCL_API constexpr tuple_element_t<_Ip, tuple>&& __get_impl() && noexcept
  {
    using type _CCCL_NODEBUG_ALIAS = tuple_element_t<_Ip, tuple>;
    return static_cast<type&&>(static_cast<__tuple_leaf<_Ip, type>&&>(__base_).get());
  }

  template <size_t _Ip>
  _CCCL_API constexpr const tuple_element_t<_Ip, tuple>&& __get_impl() const&& noexcept
  {
    using type _CCCL_NODEBUG_ALIAS = tuple_element_t<_Ip, tuple>;
    return static_cast<const type&&>(static_cast<const __tuple_leaf<_Ip, type>&&>(__base_).get());
  }

  template <class _Constraints                                               = __tuple_constraints<_Tp...>,
            enable_if_t<_Constraints::__implicit_default_constructible, int> = 0>
  _CCCL_API constexpr tuple() noexcept(_Constraints::__nothrow_default_constructible)
  {}

  template <class _Constraints                                               = __tuple_constraints<_Tp...>,
            enable_if_t<_Constraints::__explicit_default_constructible, int> = 0>
  _CCCL_API explicit constexpr tuple() noexcept(_Constraints::__nothrow_default_constructible)
  {}

  _CCCL_HIDE_FROM_ABI tuple(tuple const&) = default;
  _CCCL_HIDE_FROM_ABI tuple(tuple&&)      = default;

  template <class _Alloc,
            class _Constraints                                               = __tuple_constraints<_Tp...>,
            enable_if_t<_Constraints::__implicit_default_constructible, int> = 0>
  _CCCL_API inline tuple(allocator_arg_t, _Alloc const& __a) noexcept(_Constraints::__nothrow_default_constructible)
      : __base_(allocator_arg_t(), __a)
  {}

  template <class _Alloc,
            class _Constraints                                               = __tuple_constraints<_Tp...>,
            enable_if_t<_Constraints::__explicit_default_constructible, int> = 0>
  explicit
    _CCCL_API inline tuple(allocator_arg_t, _Alloc const& __a) noexcept(_Constraints::__nothrow_default_constructible)
      : __base_(allocator_arg_t(), __a)
  {}

  template <class _Constraints                                                     = __tuple_constraints<_Tp...>,
            enable_if_t<_Constraints::__implicit_variadic_copy_constructible, int> = 0>
  _CCCL_API constexpr tuple(const _Tp&... __t) noexcept(_Constraints::__nothrow_variadic_copy_constructible)
      : __base_(__tuple_variadic_constructor_tag{}, __t...)
  {}

  template <class _Constraints                                                     = __tuple_constraints<_Tp...>,
            enable_if_t<_Constraints::__explicit_variadic_copy_constructible, int> = 0>
  _CCCL_API constexpr explicit tuple(const _Tp&... __t) noexcept(_Constraints::__nothrow_variadic_copy_constructible)
      : __base_(__tuple_variadic_constructor_tag{}, __t...)
  {}

  template <class _Alloc,
            class _Constraints                                                     = __tuple_constraints<_Tp...>,
            enable_if_t<_Constraints::__implicit_variadic_copy_constructible, int> = 0>
  _CCCL_API inline tuple(allocator_arg_t, const _Alloc& __a, const _Tp&... __t) noexcept(
    _Constraints::__nothrow_variadic_copy_constructible)
      : __base_(allocator_arg_t(), __a, __tuple_variadic_constructor_tag{}, __t...)
  {}

  template <class _Alloc,
            class _Constraints                                                     = __tuple_constraints<_Tp...>,
            enable_if_t<_Constraints::__explicit_variadic_copy_constructible, int> = 0>
  _CCCL_API inline explicit tuple(allocator_arg_t, const _Alloc& __a, const _Tp&... __t) noexcept(
    _Constraints::__nothrow_variadic_copy_constructible)
      : __base_(allocator_arg_t(), __a, __tuple_variadic_constructor_tag{}, __t...)
  {}

  template <class... _Args>
  static constexpr bool __expands_to_this_tuple =
    (sizeof...(_Args) == 1) && (is_same_v<remove_cvref_t<_Args>, tuple> && ...);

  template <class... _Up>
  using __variadic_constraints =
    _If<!__expands_to_this_tuple<_Up...> && sizeof...(_Up) == sizeof...(_Tp),
        typename __tuple_constraints<_Tp...>::template __variadic_constraints<_Up...>,
        __invalid_tuple_constraints>;

  template <class... _Up,
            class _Constraints                                       = __variadic_constraints<_Up...>,
            enable_if_t<_Constraints::__implicit_constructible, int> = 0>
  _CCCL_API constexpr tuple(_Up&&... __u) noexcept(_Constraints::__nothrow_constructible)
      : __base_(__tuple_variadic_constructor_tag{}, ::cuda::std::forward<_Up>(__u)...)
  {}

  template <class... _Up,
            class _Constraints                                       = __variadic_constraints<_Up...>,
            enable_if_t<_Constraints::__explicit_constructible, int> = 0>
  _CCCL_API constexpr explicit tuple(_Up&&... __u) noexcept(_Constraints::__nothrow_constructible)
      : __base_(__tuple_variadic_constructor_tag{}, ::cuda::std::forward<_Up>(__u)...)
  {}

  template <class... _Up>
  using __variadic_constraints_less_rank =
    _If<!__expands_to_this_tuple<_Up...>,
        typename __tuple_constraints<_Tp...>::template __variadic_constraints_less_rank<_Up...>,
        __invalid_tuple_constraints>;

  template <class... _Up,
            class _Constraints                                       = __variadic_constraints_less_rank<_Up...>,
            enable_if_t<sizeof...(_Up) < sizeof...(_Tp), int>        = 0,
            enable_if_t<_Constraints::__implicit_constructible, int> = 0>
  _CCCL_API constexpr explicit tuple(_Up&&... __u) noexcept(is_nothrow_constructible_v<_BaseT, _Up...>)
      : __base_(__tuple_variadic_constructor_tag{}, ::cuda::std::forward<_Up>(__u)...)
  {}

  template <class _Alloc,
            class... _Up,
            class _Constraints                                       = __variadic_constraints<_Up...>,
            enable_if_t<_Constraints::__implicit_constructible, int> = 0>
  _CCCL_API inline tuple(allocator_arg_t, const _Alloc& __a, _Up&&... __u) noexcept(
    _Constraints::__nothrow_constructible)
      : __base_(allocator_arg_t(), __a, __tuple_variadic_constructor_tag{}, ::cuda::std::forward<_Up>(__u)...)
  {}

  template <class _Alloc,
            class... _Up,
            class _Constraints                                       = __variadic_constraints<_Up...>,
            enable_if_t<_Constraints::__explicit_constructible, int> = 0>
  _CCCL_API inline explicit tuple(allocator_arg_t, const _Alloc& __a, _Up&&... __u) noexcept(
    _Constraints::__nothrow_constructible)
      : __base_(allocator_arg_t(), __a, __tuple_variadic_constructor_tag{}, ::cuda::std::forward<_Up>(__u)...)
  {}

  template <class _Tuple>
  using __tuple_like_constraints =
    _If<__tuple_like_with_size<_Tuple, sizeof...(_Tp)>,
        typename __tuple_constraints<_Tp...>::template __tuple_like_constraints<_Tuple>,
        __invalid_tuple_constraints>;

  // Horrible hack to make tuple_of_iterator_references work
  template <class _TupleOfIteratorReferences,
            enable_if_t<__is_tuple_of_iterator_references_v<_TupleOfIteratorReferences>, int>   = 0,
            enable_if_t<(tuple_size<_TupleOfIteratorReferences>::value == sizeof...(_Tp)), int> = 0>
  _CCCL_API constexpr tuple(_TupleOfIteratorReferences&& __t)
      : tuple(::cuda::std::forward<_TupleOfIteratorReferences>(__t), __make_tuple_indices_t<sizeof...(_Tp)>{})
  {}

private:
  template <class _TupleOfIteratorReferences,
            size_t... _Indices,
            enable_if_t<__is_tuple_of_iterator_references_v<_TupleOfIteratorReferences>, int> = 0>
  _CCCL_API constexpr tuple(_TupleOfIteratorReferences&& __t, __tuple_indices<_Indices...>)
      : tuple(::cuda::std::get<_Indices>(::cuda::std::forward<_TupleOfIteratorReferences>(__t))...)
  {}

public:
  template <class _Tuple,
            class _Constraints                                       = __tuple_like_constraints<_Tuple>,
            enable_if_t<!__expands_to_this_tuple<_Tuple>, int>       = 0,
            enable_if_t<!is_lvalue_reference_v<_Tuple>, int>         = 0,
            enable_if_t<_Constraints::__implicit_constructible, int> = 0>
  _CCCL_API constexpr tuple(_Tuple&& __t) noexcept(is_nothrow_constructible_v<_BaseT, _Tuple>)
      : __base_(::cuda::std::forward<_Tuple>(__t))
  {}

  template <class _Tuple,
            class _Constraints                                       = __tuple_like_constraints<const _Tuple&>,
            enable_if_t<!__expands_to_this_tuple<_Tuple>, int>       = 0,
            enable_if_t<_Constraints::__implicit_constructible, int> = 0>
  _CCCL_API constexpr tuple(const _Tuple& __t) noexcept(is_nothrow_constructible_v<_BaseT, const _Tuple&>)
      : __base_(__t)
  {}

  template <class _Tuple,
            class _Constraints                                       = __tuple_like_constraints<_Tuple>,
            enable_if_t<!__expands_to_this_tuple<_Tuple>, int>       = 0,
            enable_if_t<!is_lvalue_reference_v<_Tuple>, int>         = 0,
            enable_if_t<_Constraints::__explicit_constructible, int> = 0>
  _CCCL_API constexpr explicit tuple(_Tuple&& __t) noexcept(is_nothrow_constructible_v<_BaseT, _Tuple>)
      : __base_(::cuda::std::forward<_Tuple>(__t))
  {}

  template <class _Tuple,
            class _Constraints                                       = __tuple_like_constraints<const _Tuple&>,
            enable_if_t<!__expands_to_this_tuple<_Tuple>, int>       = 0,
            enable_if_t<_Constraints::__explicit_constructible, int> = 0>
  _CCCL_API constexpr explicit tuple(const _Tuple& __t) noexcept(is_nothrow_constructible_v<_BaseT, const _Tuple&>)
      : __base_(__t)
  {}

  template <class _Alloc,
            class _Tuple,
            class _Constraints                                       = __tuple_like_constraints<_Tuple>,
            enable_if_t<_Constraints::__implicit_constructible, int> = 0>
  _CCCL_API inline tuple(allocator_arg_t, const _Alloc& __a, _Tuple&& __t)
      : __base_(allocator_arg_t(), __a, ::cuda::std::forward<_Tuple>(__t))
  {}

  template <class _Alloc,
            class _Tuple,
            class _Constraints                                       = __tuple_like_constraints<_Tuple>,
            enable_if_t<_Constraints::__explicit_constructible, int> = 0>
  _CCCL_API inline explicit tuple(allocator_arg_t, const _Alloc& __a, _Tuple&& __t)
      : __base_(allocator_arg_t(), __a, ::cuda::std::forward<_Tuple>(__t))
  {}

  _CCCL_HIDE_FROM_ABI tuple& operator=(const tuple& __t) = default;
  _CCCL_HIDE_FROM_ABI tuple& operator=(tuple&& __t)      = default;

  template <class _Tuple, enable_if_t<__tuple_assignable<_Tuple, tuple>, int> = 0>
  _CCCL_API inline tuple& operator=(_Tuple&& __t) noexcept(is_nothrow_assignable_v<_BaseT&, _Tuple>)
  {
    __base_.operator=(::cuda::std::forward<_Tuple>(__t));
    return *this;
  }

  _CCCL_API void swap(tuple& __t)
  {
    __base_.swap(__t.__base_);
  }

  _CCCL_API friend void swap(tuple& __t, tuple& __u)
  {
    __t.swap(__u);
  }
};

template <>
class _CCCL_TYPE_VISIBILITY_DEFAULT tuple<>
{
public:
  _CCCL_HIDE_FROM_ABI constexpr tuple() noexcept = default;
  template <class _Alloc>
  _CCCL_API inline tuple(allocator_arg_t, const _Alloc&) noexcept
  {}
  template <class _Alloc>
  _CCCL_API inline tuple(allocator_arg_t, const _Alloc&, const tuple&) noexcept
  {}
  template <class _Up>
  _CCCL_API inline tuple(array<_Up, 0>) noexcept
  {}
  template <class _Alloc, class _Up>
  _CCCL_API inline tuple(allocator_arg_t, const _Alloc&, array<_Up, 0>) noexcept
  {}
  _CCCL_API inline void swap(tuple&) noexcept {}
};

template <class... _Tp>
_CCCL_HOST_DEVICE tuple(_Tp...) -> tuple<_Tp...>;
template <class _Tp1, class _Tp2>
_CCCL_HOST_DEVICE tuple(pair<_Tp1, _Tp2>) -> tuple<_Tp1, _Tp2>;
template <class _Alloc, class... _Tp>
_CCCL_HOST_DEVICE tuple(allocator_arg_t, _Alloc, _Tp...) -> tuple<_Tp...>;
template <class _Alloc, class _Tp1, class _Tp2>
_CCCL_HOST_DEVICE tuple(allocator_arg_t, _Alloc, pair<_Tp1, _Tp2>) -> tuple<_Tp1, _Tp2>;
template <class _Alloc, class... _Tp>
_CCCL_HOST_DEVICE tuple(allocator_arg_t, _Alloc, tuple<_Tp...>) -> tuple<_Tp...>;

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_TUPLE_H
