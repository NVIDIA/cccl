//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___VARIANT_COMPARISON_H
#define _CUDA_STD___VARIANT_COMPARISON_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__functional/operations.h>
#include <cuda/std/__fwd/variant.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__variant/variant.h>
#include <cuda/std/__variant/visit.h>

// [variant.syn]
#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()
#  include <cuda/std/compare>
#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Operator>
struct __convert_to_bool
{
  template <class _T1, class _T2>
  [[nodiscard]] _CCCL_API constexpr bool operator()(_T1&& __t1, _T2&& __t2) const
  {
    static_assert(
      is_convertible_v<decltype(_Operator{}(::cuda::std::forward<_T1>(__t1), ::cuda::std::forward<_T2>(__t2))), bool>,
      "the relational operator does not return a type which is "
      "implicitly convertible to bool");
    return _Operator{}(::cuda::std::forward<_T1>(__t1), ::cuda::std::forward<_T2>(__t2));
  }
};

template <class... _Types>
[[nodiscard]] _CCCL_API constexpr bool operator==(const variant<_Types...>& __lhs, const variant<_Types...>& __rhs)
{
  const auto __index_ = __lhs.index();
  if (__index_ != __rhs.index())
  {
    return false;
  }
  if (__lhs.valueless_by_exception())
  {
    return true;
  }

  return __variant_binary_visitor::__visit(__index_, __convert_to_bool<equal_to<>>{}, __lhs, __rhs);
}

#if _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

template <class... _Types>
  requires(three_way_comparable<_Types> && ...)
[[nodiscard]] _CCCL_API constexpr common_comparison_category_t<compare_three_way_result_t<_Types>...>
operator<=>(const variant<_Types...>& __lhs, const variant<_Types...>& __rhs)
{
  using __result_t = common_comparison_category_t<compare_three_way_result_t<_Types>...>;
  if (__lhs.valueless_by_exception() && __rhs.valueless_by_exception())
  {
    return strong_ordering::equal;
  }
  if (__lhs.valueless_by_exception())
  {
    return strong_ordering::less;
  }
  if (__rhs.valueless_by_exception())
  {
    return strong_ordering::greater;
  }
  if (auto __c = __lhs.index() <=> __rhs.index(); __c != 0)
  {
    return __c;
  }
  auto __three_way = []<class _Type>(const _Type& __v, const _Type& __w) -> __result_t {
    return __v <=> __w;
  };
  return __variant_binary_visitor::__visit(__lhs.index(), __three_way, __lhs, __rhs);
}

#endif // _LIBCUDACXX_HAS_SPACESHIP_OPERATOR()

template <class... _Types>
[[nodiscard]] _CCCL_API constexpr bool operator!=(const variant<_Types...>& __lhs, const variant<_Types...>& __rhs)
{
  if (__lhs.index() != __rhs.index())
  {
    return true;
  }
  if (__lhs.valueless_by_exception())
  {
    return false;
  }
  return __variant_binary_visitor::__visit(__lhs.index(), __convert_to_bool<not_equal_to<>>{}, __lhs, __rhs);
}

template <class... _Types>
[[nodiscard]] _CCCL_API constexpr bool operator<(const variant<_Types...>& __lhs, const variant<_Types...>& __rhs)
{
  if (__rhs.valueless_by_exception())
  {
    return false;
  }
  if (__lhs.valueless_by_exception())
  {
    return true;
  }
  if (__lhs.index() < __rhs.index())
  {
    return true;
  }
  if (__lhs.index() > __rhs.index())
  {
    return false;
  }
  return __variant_binary_visitor::__visit(__lhs.index(), __convert_to_bool<less<>>{}, __lhs, __rhs);
}

template <class... _Types>
[[nodiscard]] _CCCL_API constexpr bool operator>(const variant<_Types...>& __lhs, const variant<_Types...>& __rhs)
{
  if (__lhs.valueless_by_exception())
  {
    return false;
  }
  if (__rhs.valueless_by_exception())
  {
    return true;
  }
  if (__lhs.index() > __rhs.index())
  {
    return true;
  }
  if (__lhs.index() < __rhs.index())
  {
    return false;
  }
  return __variant_binary_visitor::__visit(__lhs.index(), __convert_to_bool<greater<>>{}, __lhs, __rhs);
}

template <class... _Types>
[[nodiscard]] _CCCL_API constexpr bool operator<=(const variant<_Types...>& __lhs, const variant<_Types...>& __rhs)
{
  if (__lhs.valueless_by_exception())
  {
    return true;
  }
  if (__rhs.valueless_by_exception())
  {
    return false;
  }
  if (__lhs.index() < __rhs.index())
  {
    return true;
  }
  if (__lhs.index() > __rhs.index())
  {
    return false;
  }
  return __variant_binary_visitor::__visit(__lhs.index(), __convert_to_bool<less_equal<>>{}, __lhs, __rhs);
}

template <class... _Types>
[[nodiscard]] _CCCL_API constexpr bool operator>=(const variant<_Types...>& __lhs, const variant<_Types...>& __rhs)
{
  if (__rhs.valueless_by_exception())
  {
    return true;
  }
  if (__lhs.valueless_by_exception())
  {
    return false;
  }
  if (__lhs.index() > __rhs.index())
  {
    return true;
  }
  if (__lhs.index() < __rhs.index())
  {
    return false;
  }
  return __variant_binary_visitor::__visit(__lhs.index(), __convert_to_bool<greater_equal<>>{}, __lhs, __rhs);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___VARIANT_COMPARISON_H
