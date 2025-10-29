//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___TUPLE_TUPLE_LEAF_H
#define _CUDA_STD___TUPLE_TUPLE_LEAF_H

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
#include <cuda/std/__tuple_dir/make_tuple_types.h>
#include <cuda/std/__tuple_dir/sfinae_helpers.h>
#include <cuda/std/__tuple_dir/tuple_constraints.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__tuple_dir/tuple_indices.h>
#include <cuda/std/__tuple_dir/tuple_types.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_assignable.h>
#include <cuda/std/__type_traits/is_empty.h>
#include <cuda/std/__type_traits/is_final.h>
#include <cuda/std/__type_traits/is_nothrow_assignable.h>
#include <cuda/std/__type_traits/is_nothrow_constructible.h>
#include <cuda/std/__type_traits/is_nothrow_default_constructible.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_swappable.h>
#include <cuda/std/__type_traits/reference_constructs_from_temporary.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/swap.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

struct __tuple_leaf_default_constructor_tag
{};

//! @brief Different types of `__tuple_leaf` specialization. We need those to ensure we are trivially copyable
enum class __tuple_leaf_specialization
{
  __default,
  __synthesize_assignment,
  __empty_non_final,
};

//! @brief Detects whether we need to synthesize the assignment operator for reference types or can use EBCO
template <class _Tp>
_CCCL_API constexpr __tuple_leaf_specialization __tuple_leaf_choose()
{
  return is_empty_v<_Tp> && !is_final_v<_Tp> ? __tuple_leaf_specialization::__empty_non_final
       : __must_synthesize_assignment_v<_Tp>
         ? __tuple_leaf_specialization::__synthesize_assignment
         : __tuple_leaf_specialization::__default;
}

template <size_t _Ip, class _Hp, __tuple_leaf_specialization = __tuple_leaf_choose<_Hp>()>
class __tuple_leaf;

_CCCL_EXEC_CHECK_DISABLE
template <size_t _Ip, class _Hp, __tuple_leaf_specialization _Ep>
_CCCL_API inline void swap(__tuple_leaf<_Ip, _Hp, _Ep>& __x,
                           __tuple_leaf<_Ip, _Hp, _Ep>& __y) noexcept(is_nothrow_swappable_v<_Hp>)
{
  swap(__x.get(), __y.get());
}

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4244) // conversion from '_Tp' to '_Hp', possible loss of data

template <class _Tp, class _Up, class _Current, bool = !is_same_v<remove_cvref_t<_Tp>, _Current>>
inline constexpr bool __tuple_leaf_can_forward = false;

template <class _Tp, class _Up, class _Current>
inline constexpr bool __tuple_leaf_can_forward<_Tp, _Up, _Current, true> = is_constructible_v<_Up, _Tp>;

template <size_t _Ip, class _Hp>
class __tuple_leaf<_Ip, _Hp, __tuple_leaf_specialization::__default>
{
  _Hp __value_;

#if defined(_CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY)
  template <class _Up>
  static constexpr bool __can_bind_reference = !reference_constructs_from_temporary_v<_Hp&, _Up>;
#else
  template <class _Up>
  static constexpr bool __can_bind_reference = true;
#endif // !_CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY

public:
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr __tuple_leaf() noexcept(is_nothrow_default_constructible_v<_Hp>)
      : __value_()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr __tuple_leaf(__tuple_leaf_default_constructor_tag) noexcept(
    is_nothrow_default_constructible_v<_Hp>)
      : __value_()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Alloc>
  _CCCL_API inline __tuple_leaf(integral_constant<int, 0>, const _Alloc&)
      : __value_()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Alloc>
  _CCCL_API inline __tuple_leaf(integral_constant<int, 1>, const _Alloc& __a)
      : __value_(allocator_arg_t(), __a)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Alloc>
  _CCCL_API inline __tuple_leaf(integral_constant<int, 2>, const _Alloc& __a)
      : __value_(__a)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp, enable_if_t<__tuple_leaf_can_forward<_Tp, _Hp, __tuple_leaf>, int> = 0>
  _CCCL_API constexpr explicit __tuple_leaf(_Tp&& __t) noexcept(is_nothrow_constructible_v<_Hp, _Tp>)
      : __value_(::cuda::std::forward<_Tp>(__t))
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp, class _Alloc>
  _CCCL_API inline explicit __tuple_leaf(integral_constant<int, 0>, const _Alloc&, _Tp&& __t)
      : __value_(::cuda::std::forward<_Tp>(__t))
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp, class _Alloc>
  _CCCL_API inline explicit __tuple_leaf(integral_constant<int, 1>, const _Alloc& __a, _Tp&& __t)
      : __value_(allocator_arg_t(), __a, ::cuda::std::forward<_Tp>(__t))
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp, class _Alloc>
  _CCCL_API inline explicit __tuple_leaf(integral_constant<int, 2>, const _Alloc& __a, _Tp&& __t)
      : __value_(::cuda::std::forward<_Tp>(__t), __a)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp>
  _CCCL_API inline __tuple_leaf& operator=(_Tp&& __t) noexcept(is_nothrow_assignable_v<_Hp&, _Tp>)
  {
    __value_ = ::cuda::std::forward<_Tp>(__t);
    return *this;
  }

  _CCCL_API inline void swap(__tuple_leaf& __t) noexcept(is_nothrow_swappable_v<_Hp>)
  {
    ::cuda::std::swap(*this, __t);
  }

  [[nodiscard]] _CCCL_API constexpr _Hp& get() noexcept
  {
    return __value_;
  }
  [[nodiscard]] _CCCL_API constexpr const _Hp& get() const noexcept
  {
    return __value_;
  }
};

template <size_t _Ip, class _Hp>
class __tuple_leaf<_Ip, _Hp, __tuple_leaf_specialization::__synthesize_assignment>
{
  _Hp __value_;

#if defined(_CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY)
  template <class _Up>
  static constexpr bool __can_bind_reference = !reference_constructs_from_temporary_v<_Hp&, _Up>;
#else
  template <class _Up>
  static constexpr bool __can_bind_reference = true;
#endif // !_CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY

public:
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr __tuple_leaf() noexcept(is_nothrow_default_constructible_v<_Hp>)
      : __value_()
  {
    static_assert(!is_reference_v<_Hp>, "Attempted to default construct a reference element in a tuple");
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp, enable_if_t<__tuple_leaf_can_forward<_Tp, _Hp, __tuple_leaf>, int> = 0>
  _CCCL_API constexpr explicit __tuple_leaf(_Tp&& __t) noexcept(is_nothrow_constructible_v<_Hp, _Tp>)
      : __value_(::cuda::std::forward<_Tp>(__t))
  {
    static_assert(__can_bind_reference<_Tp&&>,
                  "Attempted construction of reference element "
                  "binds to a temporary whose lifetime has ended");
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp, class _Alloc>
  _CCCL_API inline explicit __tuple_leaf(integral_constant<int, 0>, const _Alloc&, _Tp&& __t)
      : __value_(::cuda::std::forward<_Tp>(__t))
  {
    static_assert(__can_bind_reference<_Tp&&>,
                  "Attempted construction of reference element binds to a "
                  "temporary whose lifetime has ended");
  }

  _CCCL_EXEC_CHECK_DISABLE
  __tuple_leaf(const __tuple_leaf& __t) = default;
  _CCCL_EXEC_CHECK_DISABLE
  __tuple_leaf(__tuple_leaf&& __t) = default;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr __tuple_leaf& operator=(const __tuple_leaf& __t) noexcept
  {
    __value_ = __t.__value_;
    return *this;
  }
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr __tuple_leaf& operator=(__tuple_leaf&& __t) noexcept
  {
    __value_ = ::cuda::std::move(__t.__value_);
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp>
  _CCCL_API inline __tuple_leaf& operator=(_Tp&& __t) noexcept(is_nothrow_assignable_v<_Hp&, _Tp>)
  {
    __value_ = ::cuda::std::forward<_Tp>(__t);
    return *this;
  }

  _CCCL_API inline void swap(__tuple_leaf& __t) noexcept(is_nothrow_swappable_v<_Hp>)
  {
    ::cuda::std::swap(*this, __t);
  }

  [[nodiscard]] _CCCL_API constexpr _Hp& get() noexcept
  {
    return __value_;
  }
  [[nodiscard]] _CCCL_API constexpr const _Hp& get() const noexcept
  {
    return __value_;
  }
};

template <size_t _Ip, class _Hp>
class __tuple_leaf<_Ip, _Hp, __tuple_leaf_specialization::__empty_non_final> : private remove_const_t<_Hp>
{
public:
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr __tuple_leaf() noexcept(is_nothrow_default_constructible_v<_Hp>)
      : _Hp()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API constexpr __tuple_leaf(__tuple_leaf_default_constructor_tag) noexcept(
    is_nothrow_default_constructible_v<_Hp>)
      : _Hp()
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Alloc>
  _CCCL_API inline __tuple_leaf(integral_constant<int, 0>, const _Alloc&)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Alloc>
  _CCCL_API inline __tuple_leaf(integral_constant<int, 1>, const _Alloc& __a)
      : _Hp(allocator_arg_t(), __a)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Alloc>
  _CCCL_API inline __tuple_leaf(integral_constant<int, 2>, const _Alloc& __a)
      : _Hp(__a)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp, enable_if_t<__tuple_leaf_can_forward<_Tp, _Hp, __tuple_leaf>, int> = 0>
  _CCCL_API constexpr explicit __tuple_leaf(_Tp&& __t) noexcept(is_nothrow_constructible_v<_Hp, _Tp>)
      : _Hp(::cuda::std::forward<_Tp>(__t))
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp, class _Alloc>
  _CCCL_API inline explicit __tuple_leaf(integral_constant<int, 0>, const _Alloc&, _Tp&& __t)
      : _Hp(::cuda::std::forward<_Tp>(__t))
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp, class _Alloc>
  _CCCL_API inline explicit __tuple_leaf(integral_constant<int, 1>, const _Alloc& __a, _Tp&& __t)
      : _Hp(allocator_arg_t(), __a, ::cuda::std::forward<_Tp>(__t))
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp, class _Alloc>
  _CCCL_API inline explicit __tuple_leaf(integral_constant<int, 2>, const _Alloc& __a, _Tp&& __t)
      : _Hp(::cuda::std::forward<_Tp>(__t), __a)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp, enable_if_t<is_assignable_v<_Hp&, const _Tp&>, int> = 0>
  _CCCL_API inline __tuple_leaf& operator=(const _Tp& __t) noexcept(is_nothrow_assignable_v<_Hp&, const _Tp&>)
  {
    _Hp::operator=(__t);
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp, enable_if_t<is_assignable_v<_Hp&, _Tp>, int> = 0>
  _CCCL_API inline __tuple_leaf& operator=(_Tp&& __t) noexcept(is_nothrow_assignable_v<_Hp&, _Tp>)
  {
    _Hp::operator=(::cuda::std::forward<_Tp>(__t));
    return *this;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_API inline void swap(__tuple_leaf& __t) noexcept(is_nothrow_swappable_v<_Hp>)
  {
    ::cuda::std::swap(*this, __t);
  }

  [[nodiscard]] _CCCL_API constexpr _Hp& get() noexcept
  {
    return static_cast<_Hp&>(*this);
  }
  [[nodiscard]] _CCCL_API constexpr const _Hp& get() const noexcept
  {
    return static_cast<const _Hp&>(*this);
  }
};

_CCCL_DIAG_POP

struct __tuple_variadic_constructor_tag
{};

// __tuple_impl

template <class _Indx, class... _Tp>
struct _CCCL_DECLSPEC_EMPTY_BASES __tuple_impl;

template <size_t... _Indx, class... _Tp>
struct _CCCL_DECLSPEC_EMPTY_BASES __tuple_impl<__tuple_indices<_Indx...>, _Tp...>
    : public __tuple_leaf<_Indx, _Tp>...
    , public __tuple_impl_sfinae_helper<__tuple_impl<__tuple_indices<_Indx...>, _Tp...>,
                                        __tuple_all_copy_assignable_v<_Tp...>,
                                        __tuple_all_move_assignable_v<_Tp...>>
{
  using _Constraints = __tuple_constraints<_Tp...>;

  _CCCL_API constexpr __tuple_impl() noexcept(_Constraints::__nothrow_default_constructible)
      : __tuple_leaf<_Indx, _Tp>()...
  {}

  // Handle non-allocator, full initialization
  // Old MSVC cannot handle the noexept specifier outside of template arguments
  template <class... _Up,
            class _Constraints = typename _Constraints::template __variadic_constraints<_Up...>,
            enable_if_t<sizeof...(_Up) == sizeof...(_Tp), int> = 0>
  _CCCL_API constexpr explicit __tuple_impl(__tuple_variadic_constructor_tag,
                                            _Up&&... __u) noexcept(_Constraints::__nothrow_constructible)
      : __tuple_leaf<_Indx, _Tp>(::cuda::std::forward<_Up>(__u))...
  {}

  // Handle non-allocator, partial default initialization
  // Recursively delegate until we have full rank
  template <class... _Up, enable_if_t<sizeof...(_Up) < sizeof...(_Tp), int> = 0>
  _CCCL_API constexpr explicit __tuple_impl(__tuple_variadic_constructor_tag __tag, _Up&&... __u) noexcept(
    noexcept(__tuple_impl(__tag, ::cuda::std::forward<_Up>(__u)..., __tuple_leaf_default_constructor_tag{})))
      : __tuple_impl(__tag, ::cuda::std::forward<_Up>(__u)..., __tuple_leaf_default_constructor_tag{})
  {}

  // Handle allocator aware, full initialization
  template <class _Alloc, class... _Up, enable_if_t<sizeof...(_Up) == sizeof...(_Tp), int> = 0>
  _CCCL_API inline explicit __tuple_impl(
    allocator_arg_t, const _Alloc& __a, __tuple_variadic_constructor_tag, _Up&&... __u)
      : __tuple_leaf<_Indx, _Tp>(__uses_alloc_ctor<_Tp, _Alloc, _Up>(), __a, ::cuda::std::forward<_Up>(__u))...
  {}

  // Handle allocator aware, full default initialization
  template <class _Alloc>
  _CCCL_API inline explicit __tuple_impl(allocator_arg_t, const _Alloc& __a)
      : __tuple_leaf<_Indx, _Tp>(__uses_alloc_ctor<_Tp, _Alloc>(), __a)...
  {}

  template <class _Tuple, size_t _Indx2>
  using __tuple_elem_at = tuple_element_t<_Indx2, __make_tuple_types_t<_Tuple>>;

  // cannot use inline variable here
  template <class _Tuple, enable_if_t<__tuple_constructible_struct<_Tuple, __tuple_types<_Tp...>>::value, int> = 0>
  _CCCL_API constexpr __tuple_impl(_Tuple&& __t) noexcept(__tuple_nothrow_constructible<_Tuple, __tuple_types<_Tp...>>)
      : __tuple_leaf<_Indx, _Tp>(::cuda::std::forward<__tuple_elem_at<_Tuple, _Indx>>(::cuda::std::get<_Indx>(__t)))...
  {}

  // cannot use inline variable here
  template <class _Alloc,
            class _Tuple,
            enable_if_t<__tuple_constructible_struct<_Tuple, __tuple_types<_Tp...>>::value, int> = 0>
  _CCCL_API inline __tuple_impl(allocator_arg_t, const _Alloc& __a, _Tuple&& __t)
      : __tuple_leaf<_Indx, _Tp>(__uses_alloc_ctor<_Tp, _Alloc, __tuple_elem_at<_Tuple, _Indx>>(),
                                 __a,
                                 ::cuda::std::forward<__tuple_elem_at<_Tuple, _Indx>>(::cuda::std::get<_Indx>(__t)))...
  {}

  template <class _Tuple, enable_if_t<__tuple_assignable<_Tuple, __tuple_types<_Tp...>>, int> = 0>
  _CCCL_API inline __tuple_impl&
  operator=(_Tuple&& __t) noexcept(__tuple_nothrow_assignable<_Tuple, __tuple_types<_Tp...>>)
  {
    (__tuple_leaf<_Indx, _Tp>::operator=(
       ::cuda::std::forward<__tuple_elem_at<_Tuple, _Indx>>(::cuda::std::get<_Indx>(__t))),
     ...);
    return *this;
  }

  _CCCL_HIDE_FROM_ABI __tuple_impl(const __tuple_impl&)            = default;
  _CCCL_HIDE_FROM_ABI __tuple_impl(__tuple_impl&&)                 = default;
  _CCCL_HIDE_FROM_ABI __tuple_impl& operator=(const __tuple_impl&) = default;
  _CCCL_HIDE_FROM_ABI __tuple_impl& operator=(__tuple_impl&&)      = default;

  // Using a fold exppression here breaks nvrtc
  _CCCL_API inline void swap(__tuple_impl& __t) noexcept(__all<is_nothrow_swappable_v<_Tp>...>::value)
  {
    (__tuple_leaf<_Indx, _Tp>::swap(static_cast<__tuple_leaf<_Indx, _Tp>&>(__t)), ...);
  }
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_TUPLE_LEAF_H
