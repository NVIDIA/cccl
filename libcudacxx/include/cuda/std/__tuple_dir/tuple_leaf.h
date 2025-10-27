//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
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

#include <cuda/std/__fwd/tuple.h>
#include <cuda/std/__memory/allocator_arg_t.h>
#include <cuda/std/__tuple_dir/sfinae_helpers.h>
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

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___TUPLE_TUPLE_LEAF_H
