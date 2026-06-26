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

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__fwd/get.h>
#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__memory/allocator_arg_t.h>
#include <cuda/std/__tuple_dir/tuple_constraints.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_indices.h>
#include <cuda/std/__tuple_dir/tuple_leaf.h>
#include <cuda/std/__tuple_dir/tuple_size.h>
#include <cuda/std/__tuple_dir/tuple_types.h>
#include <cuda/std/__type_traits/common_reference.h>
#include <cuda/std/__type_traits/common_type.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_nothrow_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/pair.h>
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
    return static_cast<__tuple_leaf<_Ip, type>&>(__base_).__get();
  }

  template <size_t _Ip>
  _CCCL_API constexpr const tuple_element_t<_Ip, tuple>& __get_impl() const& noexcept
  {
    using type _CCCL_NODEBUG_ALIAS = tuple_element_t<_Ip, tuple>;
    return static_cast<const __tuple_leaf<_Ip, type>&>(__base_).__get();
  }

  template <size_t _Ip>
  _CCCL_API constexpr tuple_element_t<_Ip, tuple>&& __get_impl() && noexcept
  {
    using type _CCCL_NODEBUG_ALIAS = tuple_element_t<_Ip, tuple>;
    return static_cast<type&&>(static_cast<__tuple_leaf<_Ip, type>&&>(__base_).__get());
  }

  template <size_t _Ip>
  _CCCL_API constexpr const tuple_element_t<_Ip, tuple>&& __get_impl() const&& noexcept
  {
    using type _CCCL_NODEBUG_ALIAS = tuple_element_t<_Ip, tuple>;
    return static_cast<const type&&>(static_cast<const __tuple_leaf<_Ip, type>&&>(__base_).__get());
  }

  // Going through an inline variable forces instantiation of the default constructors of all _Tp fr old GCC
  template <__select_constructor _Trait = ::cuda::std::__tuple_select_default_constructible(__tuple_types<_Tp...>{}),
            enable_if_t<__can_construct_implicitly<_Trait>, int> = 0>
  _CCCL_API constexpr tuple() noexcept((is_nothrow_default_constructible_v<_Tp> && ...))
  {}

  template <__select_constructor _Trait = ::cuda::std::__tuple_select_default_constructible(__tuple_types<_Tp...>{}),
            enable_if_t<__can_construct_explicitly<_Trait>, int> = 0>
  _CCCL_API explicit constexpr tuple() noexcept((is_nothrow_default_constructible_v<_Tp> && ...))
  {}

  _CCCL_HIDE_FROM_ABI tuple(tuple const&) = default;
  _CCCL_HIDE_FROM_ABI tuple(tuple&&)      = default;

  template <class _Alloc,
            __select_constructor _Trait = ::cuda::std::__tuple_select_default_constructible(__tuple_types<_Tp...>{}),
            enable_if_t<__can_construct_implicitly<_Trait>, int> = 0>
  _CCCL_API constexpr tuple(allocator_arg_t,
                            _Alloc const& __a) noexcept((is_nothrow_default_constructible_v<_Tp> && ...))
      : __base_(allocator_arg_t(), __a)
  {}

  template <class _Alloc,
            __select_constructor _Trait = ::cuda::std::__tuple_select_default_constructible(__tuple_types<_Tp...>{}),
            enable_if_t<__can_construct_explicitly<_Trait>, int> = 0>
  _CCCL_API explicit constexpr tuple(allocator_arg_t,
                                     _Alloc const& __a) noexcept((is_nothrow_default_constructible_v<_Tp> && ...))
      : __base_(allocator_arg_t(), __a)
  {}

  template <__select_constructor _Trait = __tuple_select_variadic_copy_constructible_v<__tuple_types<_Tp...>>,
            enable_if_t<__can_construct_implicitly<_Trait>, int> = 0>
  _CCCL_API constexpr tuple(const _Tp&... __t) noexcept(__tuple_all_nothrow_copy_constructible_v<_Tp...>)
      : __base_(__tuple_variadic_constructor_tag{}, __t...)
  {}

  template <__select_constructor _Trait = __tuple_select_variadic_copy_constructible_v<__tuple_types<_Tp...>>,
            enable_if_t<__can_construct_explicitly<_Trait>, int> = 0>
  _CCCL_API constexpr explicit tuple(const _Tp&... __t) noexcept(__tuple_all_nothrow_copy_constructible_v<_Tp...>)
      : __base_(__tuple_variadic_constructor_tag{}, __t...)
  {}

  template <class _Alloc,
            enable_if_t<sizeof...(_Tp) != 0, int> = 0, // Help Clang disambiguate for CTAD
            __select_constructor _Trait           = __tuple_select_variadic_copy_constructible_v<__tuple_types<_Tp...>>,
            enable_if_t<__can_construct_implicitly<_Trait>, int> = 0>
  _CCCL_API constexpr tuple(allocator_arg_t, const _Alloc& __a, const _Tp&... __t) noexcept(
    __tuple_all_nothrow_copy_constructible_v<_Tp...>)
      : __base_(allocator_arg_t(), __a, __tuple_variadic_constructor_tag{}, __t...)
  {}

  template <class _Alloc,
            enable_if_t<sizeof...(_Tp) != 0, int> = 0, // Help Clang disambiguate for CTAD
            __select_constructor _Trait           = __tuple_select_variadic_copy_constructible_v<__tuple_types<_Tp...>>,
            enable_if_t<__can_construct_explicitly<_Trait>, int> = 0>
  _CCCL_API explicit constexpr tuple(allocator_arg_t, const _Alloc& __a, const _Tp&... __t) noexcept(
    __tuple_all_nothrow_copy_constructible_v<_Tp...>)
      : __base_(allocator_arg_t(), __a, __tuple_variadic_constructor_tag{}, __t...)
  {}

  template <class _Alloc,
            __select_constructor _Trait = __tuple_select_variadic_copy_constructible_v<__tuple_types<_Tp...>>,
            enable_if_t<__can_construct<_Trait>, int> = 0>
  _CCCL_API constexpr tuple(allocator_arg_t, const _Alloc& __a, const tuple& __t) noexcept(
    __tuple_all_nothrow_copy_constructible_v<_Tp...>)
      : __base_(__tuple_like_constructor_tag{}, allocator_arg_t(), __a, __t)
  {}

  template <class _Alloc,
            __select_constructor _Trait = __tuple_select_variadic_move_constructible_v<__tuple_types<_Tp...>>,
            enable_if_t<__can_construct<_Trait>, int> = 0>
  _CCCL_API constexpr tuple(allocator_arg_t, const _Alloc& __a, tuple&& __t) noexcept(
    __tuple_all_nothrow_move_constructible_v<_Tp...>)
      : __base_(__tuple_like_constructor_tag{}, allocator_arg_t(), __a, ::cuda::std::move(__t))
  {}

  // Old MSVC chokes about a static constexpr variable needing an initializer. Work around by using a type
  template <class... _UTypes>
  using _VariadicConstraints =
    integral_constant<__select_constructor,
                      __tuple_select_variadic_constructible_v<__tuple_types<_Tp...>, __tuple_types<_UTypes...>>>;

  template <class... _UTypes>
  using _NothrowVariadic =
    bool_constant<__tuple_all_nothrow_constructible_v<__tuple_types<_Tp...>, __tuple_types<_UTypes...>>>;

  template <class... _UTypes,
            enable_if_t<sizeof...(_Tp) != 0, int>                = 0, // Help Clang disambiguate for CTAD
            __select_constructor _Trait                          = _VariadicConstraints<_UTypes...>::value,
            enable_if_t<__can_construct_implicitly<_Trait>, int> = 0>
  _CCCL_API constexpr tuple(_UTypes&&... __u) noexcept(_NothrowVariadic<_UTypes...>::value)
      : __base_(__tuple_variadic_constructor_tag{}, ::cuda::std::forward<_UTypes>(__u)...)
  {}

  template <class... _UTypes,
            enable_if_t<sizeof...(_Tp) != 0, int>                = 0, // Help Clang disambiguate for CTAD
            __select_constructor _Trait                          = _VariadicConstraints<_UTypes...>::value,
            enable_if_t<__can_construct_explicitly<_Trait>, int> = 0>
  _CCCL_API constexpr explicit tuple(_UTypes&&... __u) noexcept(_NothrowVariadic<_UTypes...>::value)
      : __base_(__tuple_variadic_constructor_tag{}, ::cuda::std::forward<_UTypes>(__u)...)
  {}

  template <class _Alloc,
            class... _UTypes,
            __select_constructor _Trait                          = _VariadicConstraints<_UTypes...>::value,
            enable_if_t<__can_construct_implicitly<_Trait>, int> = 0>
  _CCCL_API inline tuple(allocator_arg_t, const _Alloc& __a, _UTypes&&... __u) noexcept(
    _NothrowVariadic<_UTypes...>::value)
      : __base_(allocator_arg_t(), __a, __tuple_variadic_constructor_tag{}, ::cuda::std::forward<_UTypes>(__u)...)
  {}

  template <class _Alloc,
            class... _UTypes,
            __select_constructor _Trait                          = _VariadicConstraints<_UTypes...>::value,
            enable_if_t<__can_construct_explicitly<_Trait>, int> = 0>
  _CCCL_API inline explicit tuple(allocator_arg_t, const _Alloc& __a, _UTypes&&... __u) noexcept(
    _NothrowVariadic<_UTypes...>::value)
      : __base_(allocator_arg_t(), __a, __tuple_variadic_constructor_tag{}, ::cuda::std::forward<_UTypes>(__u)...)
  {}

  template <class... _UTypes>
  using _VariadicConstraintsLessRank = integral_constant<
    __select_constructor,
    __tuple_select_variadic_constructible_less_rank_v<__tuple_types<_Tp...>, __tuple_types<_UTypes...>>>;

  template <class... _UTypes,
            enable_if_t<(sizeof...(_UTypes) < sizeof...(_Tp)), int> = 0,
            enable_if_t<(sizeof...(_UTypes) != 0), int>             = 0,
            __select_constructor _Trait                             = _VariadicConstraintsLessRank<_UTypes...>::value,
            enable_if_t<__can_construct_explicitly<_Trait>, int>    = 0>
  _CCCL_API constexpr explicit tuple(_UTypes&&... __u)
      : __base_(__tuple_variadic_constructor_tag{}, ::cuda::std::forward<_UTypes>(__u)...)
  {}

  // Horrible hack to make tuple_of_iterator_references work
  // NOLINTBEGIN(bugprone-forwarding-reference-overload)
  template <class _TupleOfIteratorReferences,
            // clang-tidy has fallen off its rocker and claims we can use the non-existent
            // __tuple_of_iterato_references_v here.
            // NOLINTBEGIN(modernize-type-traits)
            enable_if_t<__is_tuple_of_iterator_references_v<_TupleOfIteratorReferences>, int> = 0,
            // NOLINTEND(modernize-type-traits)
            enable_if_t<(tuple_size<_TupleOfIteratorReferences>::value == sizeof...(_Tp)), int> = 0>
  _CCCL_API constexpr tuple(_TupleOfIteratorReferences&& __t)
      : tuple(::cuda::std::forward<_TupleOfIteratorReferences>(__t), __make_tuple_indices_t<sizeof...(_Tp)>{})
  {}
  // NOLINTEND(bugprone-forwarding-reference-overload)

private:
  template <class _TupleOfIteratorReferences,
            size_t... _Indices,
            enable_if_t<__is_tuple_of_iterator_references_v<_TupleOfIteratorReferences>, int> = 0>
  _CCCL_API constexpr tuple(_TupleOfIteratorReferences&& __t, __tuple_indices<_Indices...>)
      // clang-tidy incorrectly reports "'__t' used after it was forwarded".
      // Each expansion forwards the tuple only to select get<I>'s cvref-qualified overload for a distinct element.
      // NOLINTNEXTLINE(bugprone-use-after-move)
      : tuple(::cuda::std::get<_Indices>(::cuda::std::forward<_TupleOfIteratorReferences>(__t))...)
  {}

public:
  template <class _UTuple>
  using _TupleLikeConstraints = integral_constant<
    __select_constructor,
    __tuple_select_tuple_like_constructible_v<_UTuple, __tuple_types<_Tp...>, __make_tuple_indices_t<sizeof...(_Tp)>>>;

  template <class _UTuple>
  using _NothrowTupleLike = bool_constant<
    __tuple_nothrow_tuple_like_constructible_v<_UTuple, __tuple_types<_Tp...>, __make_tuple_indices_t<sizeof...(_Tp)>>>;

  template <class... _UTypes,
            __select_constructor _Trait                          = _TupleLikeConstraints<tuple<_UTypes...>&>::value,
            enable_if_t<__can_construct_implicitly<_Trait>, int> = 0>
  _CCCL_API constexpr tuple(tuple<_UTypes...>& __t) noexcept(_NothrowTupleLike<tuple<_UTypes...>&>::value)
      : __base_(__tuple_like_constructor_tag{}, __t)
  {}

  template <class... _UTypes,
            __select_constructor _Trait                          = _TupleLikeConstraints<tuple<_UTypes...>&>::value,
            enable_if_t<__can_construct_explicitly<_Trait>, int> = 0>
  _CCCL_API explicit constexpr tuple(tuple<_UTypes...>& __t) noexcept(_NothrowTupleLike<tuple<_UTypes...>&>::value)
      : __base_(__tuple_like_constructor_tag{}, __t)
  {}

  template <class... _UTypes,
            __select_constructor _Trait = _TupleLikeConstraints<const tuple<_UTypes...>&>::value,
            enable_if_t<__can_construct_implicitly<_Trait>, int> = 0>
  _CCCL_API constexpr tuple(const tuple<_UTypes...>& __t) noexcept(_NothrowTupleLike<const tuple<_UTypes...>&>::value)
      : __base_(__tuple_like_constructor_tag{}, __t)
  {}

  template <class... _UTypes,
            __select_constructor _Trait = _TupleLikeConstraints<const tuple<_UTypes...>&>::value,
            enable_if_t<__can_construct_explicitly<_Trait>, int> = 0>
  _CCCL_API explicit constexpr tuple(const tuple<_UTypes...>& __t) noexcept(
    _NothrowTupleLike<const tuple<_UTypes...>&>::value)
      : __base_(__tuple_like_constructor_tag{}, __t)
  {}
  template <class... _UTypes,
            __select_constructor _Trait                          = _TupleLikeConstraints<tuple<_UTypes...>&&>::value,
            enable_if_t<__can_construct_implicitly<_Trait>, int> = 0>
  _CCCL_API constexpr tuple(tuple<_UTypes...>&& __t) noexcept(_NothrowTupleLike<tuple<_UTypes...>&&>::value)
      : __base_(__tuple_like_constructor_tag{}, ::cuda::std::move(__t))
  {}

  template <class... _UTypes,
            __select_constructor _Trait                          = _TupleLikeConstraints<tuple<_UTypes...>&&>::value,
            enable_if_t<__can_construct_explicitly<_Trait>, int> = 0>
  _CCCL_API explicit constexpr tuple(tuple<_UTypes...>&& __t) noexcept(_NothrowTupleLike<tuple<_UTypes...>&&>::value)
      : __base_(__tuple_like_constructor_tag{}, ::cuda::std::move(__t))
  {}

  template <class... _UTypes,
            __select_constructor _Trait = _TupleLikeConstraints<const tuple<_UTypes...>&&>::value,
            enable_if_t<__can_construct_implicitly<_Trait>, int> = 0>
  _CCCL_API constexpr tuple(const tuple<_UTypes...>&& __t) noexcept(_NothrowTupleLike<const tuple<_UTypes...>&&>::value)
      : __base_(__tuple_like_constructor_tag{}, ::cuda::std::move(__t))
  {}

  template <class... _UTypes,
            __select_constructor _Trait = _TupleLikeConstraints<const tuple<_UTypes...>&&>::value,
            enable_if_t<__can_construct_explicitly<_Trait>, int> = 0>
  _CCCL_API explicit constexpr tuple(const tuple<_UTypes...>&& __t) noexcept(
    _NothrowTupleLike<const tuple<_UTypes...>&&>::value)
      : __base_(__tuple_like_constructor_tag{}, ::cuda::std::move(__t))
  {}

  // We cannot instantiate _TupleLikeConstraints eagerly because the leads to recursive constraints
  // We need to SFINAE the constructor away before instantiating the traits
  template <class _Tuple>
  using __disambiguate_tuple_like =
    bool_constant<!is_same_v<remove_cvref_t<_Tuple>, tuple> && __tuple_like_with_size<_Tuple, sizeof...(_Tp)>>;

  // NOLINTBEGIN(bugprone-forwarding-reference-overload)
  template <class _Tuple,
            enable_if_t<__disambiguate_tuple_like<_Tuple>::value, int> = 0,
            __select_constructor _Trait                                = _TupleLikeConstraints<_Tuple>::value,
            enable_if_t<__can_construct_implicitly<_Trait>, int>       = 0>
  _CCCL_API constexpr tuple(_Tuple&& __t) noexcept(_NothrowTupleLike<_Tuple>::value)
      : __base_(__tuple_like_constructor_tag{}, ::cuda::std::forward<_Tuple>(__t))
  {}
  // NOLINTEND(bugprone-forwarding-reference-overload)

  // NOLINTBEGIN(bugprone-forwarding-reference-overload)
  template <class _Tuple,
            enable_if_t<__disambiguate_tuple_like<_Tuple>::value, int> = 0,
            __select_constructor _Trait                                = _TupleLikeConstraints<_Tuple>::value,
            enable_if_t<__can_construct_explicitly<_Trait>, int>       = 0>
  _CCCL_API explicit constexpr tuple(_Tuple&& __t) noexcept(_NothrowTupleLike<_Tuple>::value)
      : __base_(__tuple_like_constructor_tag{}, ::cuda::std::forward<_Tuple>(__t))
  {}
  // NOLINTEND(bugprone-forwarding-reference-overload)

  template <class _Alloc,
            class _Tuple,
            enable_if_t<__disambiguate_tuple_like<_Tuple>::value, int> = 0, // Help Clang disambiguate for CTAD
            __select_constructor _Trait                                = _TupleLikeConstraints<_Tuple>::value,
            enable_if_t<__can_construct_implicitly<_Trait>, int>       = 0>
  _CCCL_API constexpr tuple(allocator_arg_t, const _Alloc& __a, _Tuple&& __t) noexcept(_NothrowTupleLike<_Tuple>::value)
      : __base_(__tuple_like_constructor_tag{}, allocator_arg_t(), __a, ::cuda::std::forward<_Tuple>(__t))
  {}

  template <class _Alloc,
            class _Tuple,
            enable_if_t<__disambiguate_tuple_like<_Tuple>::value, int> = 0, // Help Clang disambiguate for CTAD
            __select_constructor _Trait                                = _TupleLikeConstraints<_Tuple>::value,
            enable_if_t<__can_construct_explicitly<_Trait>, int>       = 0>
  _CCCL_API explicit constexpr tuple(allocator_arg_t, const _Alloc& __a, _Tuple&& __t) noexcept(
    _NothrowTupleLike<_Tuple>::value)
      : __base_(__tuple_like_constructor_tag{}, allocator_arg_t(), __a, ::cuda::std::forward<_Tuple>(__t))
  {}

  // [tuple.assign]
  _CCCL_HIDE_FROM_ABI tuple& operator=(const tuple& __t) = default;
  _CCCL_HIDE_FROM_ABI tuple& operator=(tuple&& __t)      = default;

  // [tuple.assign]-5
  template <__select_assignment _Trait             = __tuple_select_const_copy_assignable_v<__tuple_types<_Tp...>>,
            enable_if_t<__can_assign<_Trait>, int> = 0>
  _CCCL_API constexpr const tuple& operator=(const tuple& __t) const noexcept(__can_nothrow_assign<_Trait>)
  {
    ::cuda::std::__memberwise_copy_assign(*this, __t, __make_tuple_indices_t<sizeof...(_Tp)>{});
    return *this;
  }

  // [tuple.assign]-12
  template <__select_assignment _Trait             = __tuple_select_const_move_assignable_v<__tuple_types<_Tp...>>,
            enable_if_t<__can_assign<_Trait>, int> = 0>
  _CCCL_API constexpr const tuple& operator=(tuple&& __t) const noexcept(__can_nothrow_assign<_Trait>)
  {
    ::cuda::std::__memberwise_forward_assign(
      *this, ::cuda::std::move(__t), __type_list<_Tp...>{}, __make_tuple_indices_t<sizeof...(_Tp)>{});
    return *this;
  }

  // [tuple.assign]-15
  template <class... _UTypes,
            __select_assignment _Trait             = __tuple_select_converting_assignable_v</*__is_const=*/false,
                                                                                __tuple_types<_Tp...>,
                                                                                __tuple_types<const _UTypes&...>>,
            enable_if_t<__can_assign<_Trait>, int> = 0>
  _CCCL_API constexpr tuple& operator=(const tuple<_UTypes...>& __t) noexcept(__can_nothrow_assign<_Trait>)
  {
    ::cuda::std::__memberwise_copy_assign(*this, __t, __make_tuple_indices_t<sizeof...(_Tp)>{});
    return *this;
  }

  // [tuple.assign]-18
  template <class... _UTypes,
            __select_assignment _Trait             = __tuple_select_converting_assignable_v</*__is_const=*/true,
                                                                                __tuple_types<_Tp...>,
                                                                                __tuple_types<const _UTypes&...>>,
            enable_if_t<__can_assign<_Trait>, int> = 0>
  _CCCL_API constexpr const tuple& operator=(const tuple<_UTypes...>& __t) const noexcept(__can_nothrow_assign<_Trait>)
  {
    ::cuda::std::__memberwise_copy_assign(*this, __t, __make_tuple_indices_t<sizeof...(_Tp)>{});
    return *this;
  }

  // [tuple.assign]-21
  template <
    class... _UTypes,
    __select_assignment _Trait =
      __tuple_select_converting_assignable_v</*__is_const=*/false, __tuple_types<_Tp...>, __tuple_types<_UTypes...>>,
    enable_if_t<__can_assign<_Trait>, int> = 0>
  _CCCL_API constexpr tuple& operator=(tuple<_UTypes...>&& __t) noexcept(__can_nothrow_assign<_Trait>)
  {
    ::cuda::std::__memberwise_forward_assign(
      *this, ::cuda::std::move(__t), __type_list<_UTypes...>(), __make_tuple_indices_t<sizeof...(_Tp)>{});
    return *this;
  }

  // [tuple.assign]-24
  template <
    class... _UTypes,
    __select_assignment _Trait =
      __tuple_select_converting_assignable_v</*__is_const=*/true, __tuple_types<_Tp...>, __tuple_types<_UTypes...>>,
    enable_if_t<__can_assign<_Trait>, int> = 0>
  _CCCL_API constexpr const tuple& operator=(tuple<_UTypes...>&& __t) const noexcept(__can_nothrow_assign<_Trait>)
  {
    ::cuda::std::__memberwise_forward_assign(
      *this, ::cuda::std::move(__t), __type_list<_UTypes...>(), __make_tuple_indices_t<sizeof...(_Tp)>{});
    return *this;
  }

  // [tuple.assign]-39
  template <class _UTuple,
            enable_if_t<!__is_cuda_std_tuple<remove_cvref_t<_UTuple>>, int> = 0,
            __select_assignment _Trait             = __tuple_select_tuple_like_assignable_v</*__is_const=*/false,
                                                                                _UTuple,
                                                                                __tuple_types<_Tp...>,
                                                                                __make_tuple_indices_t<sizeof...(_Tp)>>,
            enable_if_t<__can_assign<_Trait>, int> = 0>
  _CCCL_API constexpr tuple& operator=(_UTuple&& __t) noexcept(__can_nothrow_assign<_Trait>)
  {
    ::cuda::std::__memberwise_tuple_assign(
      *this, ::cuda::std::forward<_UTuple>(__t), __make_tuple_indices_t<sizeof...(_Tp)>{});
    return *this;
  }

  // [tuple.assign]-42
  template <class _UTuple,
            enable_if_t<!__is_cuda_std_tuple<remove_cvref_t<_UTuple>>, int> = 0,
            __select_assignment _Trait             = __tuple_select_tuple_like_assignable_v</*__is_const=*/true,
                                                                                _UTuple,
                                                                                __tuple_types<_Tp...>,
                                                                                __make_tuple_indices_t<sizeof...(_Tp)>>,
            enable_if_t<__can_assign<_Trait>, int> = 0>
  _CCCL_API constexpr const tuple& operator=(_UTuple&& __t) const noexcept(__can_nothrow_assign<_Trait>)
  {
    ::cuda::std::__memberwise_tuple_assign(
      *this, ::cuda::std::forward<_UTuple>(__t), __make_tuple_indices_t<sizeof...(_Tp)>{});
    return *this;
  }

  _CCCL_API void swap(tuple& __t) noexcept(noexcept(__base_.swap(__t.__base_)))
  {
    __base_.swap(__t.__base_);
  }

  _CCCL_API friend void swap(tuple& __t, tuple& __u) noexcept(noexcept(__t.swap(__u)))
  {
    __t.swap(__u);
  }

  template <class... _UTypes>
  using _ComparisonConstraints =
    decltype(::cuda::std::__tuple_is_comparable(__tuple_types<_Tp...>{}, __tuple_types<_UTypes...>{}));

  _CCCL_EXEC_CHECK_DISABLE
  template <class... _UTypes, size_t... _Indices, class _Constraints = _ComparisonConstraints<_UTypes...>>
  [[nodiscard]] _CCCL_API constexpr bool __equal(const tuple<_UTypes...>& __other, __tuple_indices<_Indices...>) const
    noexcept(_Constraints::__nothrow_equality_comparable)
  {
    using ::cuda::std::get;
    return ((get<_Indices>(*this) == get<_Indices>(__other)) && ...);
  }

  // Not a friend function because MSVC has issues with nested namespaces and thrust::tuple
  _CCCL_TEMPLATE(class... _UTypes, class _Constraints = _ComparisonConstraints<_UTypes...>)
  _CCCL_REQUIRES(_Constraints::__equality_comparable)
  [[nodiscard]] _CCCL_API constexpr bool operator==(const tuple<_UTypes...>& __rhs) const
    noexcept(_Constraints::__nothrow_equality_comparable)
  {
    return __equal(__rhs, __make_tuple_indices_t<sizeof...(_Tp)>{});
  }

  _CCCL_TEMPLATE(class... _UTypes, class _Constraints = _ComparisonConstraints<_UTypes...>)
  _CCCL_REQUIRES(_Constraints::__equality_comparable)
  [[nodiscard]] _CCCL_API constexpr bool operator!=(const tuple<_UTypes...>& __rhs) const
    noexcept(_Constraints::__nothrow_equality_comparable)
  {
    return !__equal(__rhs, __make_tuple_indices_t<sizeof...(_Tp)>{});
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class... _UTypes,
            size_t _CurrentIndex,
            size_t... _Indices,
            class _Constraints = _ComparisonConstraints<_UTypes...>>
  [[nodiscard]] _CCCL_API constexpr bool
  __tuple_less_than(const tuple<_UTypes...>& __other, __tuple_indices<_CurrentIndex, _Indices...>) const
    noexcept(_Constraints::__nothrow_less_than_comparable)
  {
    using ::cuda::std::get;
    if constexpr (sizeof...(_Indices) == 0)
    {
      return get<_CurrentIndex>(*this) < get<_CurrentIndex>(__other);
    }
    else
    {
      if (get<_CurrentIndex>(*this) < get<_CurrentIndex>(__other))
      {
        return true;
      }
      if (get<_CurrentIndex>(__other) < get<_CurrentIndex>(*this))
      {
        return false;
      }
      return this->__tuple_less_than(__other, __tuple_indices<_Indices...>{});
    }
  }

  _CCCL_TEMPLATE(class... _UTypes, class _Constraints = _ComparisonConstraints<_UTypes...>)
  _CCCL_REQUIRES(_Constraints::__less_than_comparable)
  [[nodiscard]] _CCCL_API constexpr bool operator<(const tuple<_UTypes...>& __rhs) const
    noexcept(_Constraints::__nothrow_less_than_comparable)
  {
    return __tuple_less_than(__rhs, __make_tuple_indices_t<sizeof...(_Tp)>{});
  }

  _CCCL_TEMPLATE(class... _UTypes, class _Constraints = _ComparisonConstraints<_UTypes...>)
  _CCCL_REQUIRES(_Constraints::__less_than_comparable)
  [[nodiscard]] _CCCL_API constexpr bool operator>(const tuple<_UTypes...>& __rhs) const
    noexcept(_Constraints::__nothrow_less_than_comparable)
  {
    return __rhs.__tuple_less_than(*this, __make_tuple_indices_t<sizeof...(_Tp)>{});
  }

  _CCCL_TEMPLATE(class... _UTypes, class _Constraints = _ComparisonConstraints<_UTypes...>)
  _CCCL_REQUIRES(_Constraints::__less_than_comparable)
  [[nodiscard]] _CCCL_API constexpr bool operator>=(const tuple<_UTypes...>& __rhs) const
    noexcept(_Constraints::__nothrow_less_than_comparable)
  {
    return !__tuple_less_than(__rhs, __make_tuple_indices_t<sizeof...(_Tp)>{});
  }

  _CCCL_TEMPLATE(class... _UTypes, class _Constraints = _ComparisonConstraints<_UTypes...>)
  _CCCL_REQUIRES(_Constraints::__less_than_comparable)
  [[nodiscard]] _CCCL_API constexpr bool operator<=(const tuple<_UTypes...>& __rhs) const
    noexcept(_Constraints::__nothrow_less_than_comparable)
  {
    return !__rhs.__tuple_less_than(*this, __make_tuple_indices_t<sizeof...(_Tp)>{});
  }
};

template <>
class _CCCL_TYPE_VISIBILITY_DEFAULT tuple<>
{
public:
  _CCCL_HIDE_FROM_ABI constexpr tuple() noexcept = default;
  template <class _Alloc>
  _CCCL_API constexpr tuple(allocator_arg_t, const _Alloc&) noexcept
  {}
  template <class _Alloc>
  _CCCL_API constexpr tuple(allocator_arg_t, const _Alloc&, const tuple&) noexcept
  {}
  template <class _Up>
  _CCCL_API constexpr tuple(array<_Up, 0>) noexcept
  {}
  template <class _Alloc, class _Up>
  _CCCL_API constexpr tuple(allocator_arg_t, const _Alloc&, array<_Up, 0>) noexcept
  {}

  template <class _UTuple,
            enable_if_t<!__is_cuda_std_tuple<remove_cvref_t<_UTuple>>, int> = 0,
            enable_if_t<__tuple_like_with_size<_UTuple, 0>, int>            = 0>
  _CCCL_API constexpr tuple& operator=(_UTuple&&) noexcept
  {
    return *this;
  }

  template <class _UTuple,
            enable_if_t<!__is_cuda_std_tuple<remove_cvref_t<_UTuple>>, int> = 0,
            enable_if_t<__tuple_like_with_size<_UTuple, 0>, int>            = 0>
  _CCCL_API constexpr const tuple& operator=(_UTuple&&) const noexcept
  {
    return *this;
  }

  _CCCL_API constexpr void swap(tuple&) noexcept {}

  [[nodiscard]] _CCCL_API friend constexpr bool operator==(const tuple&, const tuple&) noexcept
  {
    return true;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator!=(const tuple&, const tuple&) noexcept
  {
    return false;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator<(const tuple&, const tuple&) noexcept
  {
    return false;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator>(const tuple&, const tuple&) noexcept
  {
    return false;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator>=(const tuple&, const tuple&) noexcept
  {
    return true;
  }

  [[nodiscard]] _CCCL_API friend constexpr bool operator<=(const tuple&, const tuple&) noexcept
  {
    return true;
  }
};

namespace __tuple_common_ref
{
// Equivalent to __type_pair in type_list.h, but reimplemented here because we don't want to
// pull in 1k lines of templates just for this.
template <class _Tp, class _Up>
struct __type_pair
{
  using __first _CCCL_NODEBUG_ALIAS  = _Tp;
  using __second _CCCL_NODEBUG_ALIAS = _Up;
};
} // namespace __tuple_common_ref

template <class... _TypePairs>
_CCCL_CONCEPT __tuple_of_common_references = _CCCL_REQUIRES_EXPR((variadic _TypePairs), )(
  typename(tuple<common_reference_t<typename _TypePairs::__first, typename _TypePairs::__second>...>));

template <class... _TTypes, class... _UTypes, template <class> class _TQual, template <class> class _UQual>
struct basic_common_reference<
  tuple<_TTypes...>,
  tuple<_UTypes...>,
  _TQual,
  _UQual,
  enable_if_t<__tuple_of_common_references<__tuple_common_ref::__type_pair<_TQual<_TTypes>, _UQual<_UTypes>>...>>>
{
  using type _CCCL_NODEBUG_ALIAS = tuple<common_reference_t<_TQual<_TTypes>, _UQual<_UTypes>>...>;
};

template <class... _TypePairs>
_CCCL_CONCEPT __tuple_of_common_types = _CCCL_REQUIRES_EXPR((variadic _TypePairs), )(
  typename(tuple<common_type_t<typename _TypePairs::__first, typename _TypePairs::__second>...>));

template <class, class, class = void>
struct __tuple_common_type
{};

template <class... _TTypes, class... _UTypes>
struct __tuple_common_type<tuple<_TTypes...>,
                           tuple<_UTypes...>,
                           enable_if_t<__tuple_of_common_types<__tuple_common_ref::__type_pair<_TTypes, _UTypes>...>>>
{
  using type _CCCL_NODEBUG_ALIAS = tuple<common_type_t<_TTypes, _UTypes>...>;
};

template <class... _TTypes, class... _UTypes>
struct common_type<tuple<_TTypes...>, tuple<_UTypes...>> : __tuple_common_type<tuple<_TTypes...>, tuple<_UTypes...>>
{};

template <class... _Tp>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES tuple(_Tp...) -> tuple<_Tp...>;
template <class _Tp1, class _Tp2>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES tuple(pair<_Tp1, _Tp2>) -> tuple<_Tp1, _Tp2>;
template <class _Alloc, class... _Tp>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES tuple(allocator_arg_t, _Alloc, _Tp...) -> tuple<_Tp...>;
template <class _Alloc, class _Tp1, class _Tp2>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES tuple(allocator_arg_t, _Alloc, pair<_Tp1, _Tp2>) -> tuple<_Tp1, _Tp2>;
template <class _Alloc, class... _Tp>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES tuple(allocator_arg_t, _Alloc, tuple<_Tp...>) -> tuple<_Tp...>;

template <class _T1, class _T2, bool _IsRef>
template <class... _Args1, class... _Args2, size_t... _I1, size_t... _I2>
_CCCL_API constexpr __pair_base<_T1, _T2, _IsRef>::__pair_base(
  piecewise_construct_t,
  tuple<_Args1...>& __first_args,
  tuple<_Args2...>& __second_args,
  __tuple_indices<_I1...>,
  __tuple_indices<_I2...>)
    : first(::cuda::std::forward<_Args1>(::cuda::std::get<_I1>(__first_args))...)
    , second(::cuda::std::forward<_Args2>(::cuda::std::get<_I2>(__second_args))...)
{}

// specialize cuda::std::tuple_size and cuda::std::tuple_element for std::tuple and cuda::std::tuple

#if _CCCL_HAS_HOST_STD_LIB()
template <class... _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_size<::std::tuple<_Tp...>> : integral_constant<size_t, sizeof...(_Tp)>
{};

template <size_t _Ip, class... _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, ::std::tuple<_Tp...>>
{
  static_assert(_Ip < sizeof...(_Tp), "Index out of bounds in cuda::std::tuple_element<> (std::tuple)");
  using type _CCCL_NODEBUG_ALIAS = tuple_element_t<_Ip, __tuple_types<_Tp...>>;
};
#endif // _CCCL_HAS_HOST_STD_LIB()

template <class... _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_size<tuple<_Tp...>> : integral_constant<size_t, sizeof...(_Tp)>
{};

template <size_t _Ip, class... _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT tuple_element<_Ip, tuple<_Tp...>>
{
  static_assert(_Ip < sizeof...(_Tp), "Index out of bounds in cuda::std::tuple_element<> (cuda::std::tuple)");
  using type _CCCL_NODEBUG_ALIAS = tuple_element_t<_Ip, __tuple_types<_Tp...>>;
};

_CCCL_END_NAMESPACE_CUDA_STD

// tuple protocol for cuda::std::tuple

_CCCL_BEGIN_NAMESPACE_STD

template <class... _Tp>
struct tuple_size<::cuda::std::tuple<_Tp...>> : ::cuda::std::integral_constant<::cuda::std::size_t, sizeof...(_Tp)>
{};

template <::cuda::std::size_t _Ip, class... _Tp>
struct tuple_element<_Ip, ::cuda::std::tuple<_Tp...>>
{
  static_assert(_Ip < sizeof...(_Tp), "Index out of bounds in std::tuple_element<> (cuda::std::tuple)");
  using type _CCCL_NODEBUG_ALIAS = ::cuda::std::tuple_element_t<_Ip, ::cuda::std::__tuple_types<_Tp...>>;
};

_CCCL_END_NAMESPACE_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_TUPLE_H
