//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___VARIANT_VISIT_H
#define _CUDA_STD___VARIANT_VISIT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/variant.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__utility/unreachable.h>
#include <cuda/std/__variant/bad_variant_access.h>
#include <cuda/std/__variant/get.h>
#include <cuda/std/__variant/variant.h>
#include <cuda/std/__variant/variant_visit.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

struct __variant_binary_visitor
{
  template <class _BinaryOp, class _LeftVariant, class _RightVariant>
  [[nodiscard]] _CCCL_API static constexpr auto
  __visit(const size_t __index_, _BinaryOp&& __op, const _LeftVariant& __lhs, const _RightVariant& __rhs)
  {
    return __visit(integral_constant<size_t, variant_size_v<_LeftVariant> - 1>{},
                   __index_,
                   ::cuda::std::forward<_BinaryOp>(__op),
                   __lhs,
                   __rhs);
  }

private:
  template <size_t _CurrentIndex, class _BinaryOp, class _LeftVariant, class _RightVariant>
  [[nodiscard]] _CCCL_API static constexpr auto
  __visit(integral_constant<size_t, _CurrentIndex>,
          const size_t __index_,
          _BinaryOp&& __op,
          const _LeftVariant& __lhs,
          const _RightVariant& __rhs)
  {
    if (__index_ == _CurrentIndex)
    {
      return __op(::cuda::std::get<_CurrentIndex>(__lhs), ::cuda::std::get<_CurrentIndex>(__rhs));
    }
    return __visit(
      integral_constant<size_t, _CurrentIndex - 1>{}, __index_, ::cuda::std::forward<_BinaryOp>(__op), __lhs, __rhs);
  }

  template <class _BinaryOp, class _LeftVariant, class _RightVariant>
  [[nodiscard]] _CCCL_API static constexpr auto
  __visit(integral_constant<size_t, 0>,
          const size_t __index_,
          _BinaryOp&& __op,
          const _LeftVariant& __lhs,
          const _RightVariant& __rhs)
  {
    if (__index_ == 0)
    {
      return __op(::cuda::std::get<0>(__lhs), ::cuda::std::get<0>(__rhs));
    }
    // We already checked that every variant has a value, so we should never reach this line
#if _CCCL_COMPILER(MSVC) // MSVC needs this to be wrapped in a function or it will error
    ::cuda::std::unreachable();
#else // ^^^ _CCCL_COMPILER(MSVC) ^^^ / vvv !_CCCL_COMPILER(MSVC) vvv
    _CCCL_UNREACHABLE();
#endif // !_CCCL_COMPILER(MSVC)
  }
};

template <class... _Types>
[[nodiscard]] _CCCL_API constexpr variant<_Types...>& __as_variant(variant<_Types...>& __vs) noexcept
{
  return __vs;
}

template <class... _Types>
[[nodiscard]] _CCCL_API constexpr const variant<_Types...>& __as_variant(const variant<_Types...>& __vs) noexcept
{
  return __vs;
}

template <class... _Types>
[[nodiscard]] _CCCL_API constexpr variant<_Types...>&& __as_variant(variant<_Types...>&& __vs) noexcept
{
  return ::cuda::std::move(__vs);
}

template <class... _Types>
[[nodiscard]] _CCCL_API constexpr const variant<_Types...>&& __as_variant(const variant<_Types...>&& __vs) noexcept
{
  return ::cuda::std::move(__vs);
}

template <class... _Vs>
_CCCL_API constexpr void __throw_if_valueless(_Vs&&... __vs)
{
  [[maybe_unused]] int __unused[] = {
    (::cuda::std::__as_variant(__vs).valueless_by_exception() ? ::cuda::std::__throw_bad_variant_access() : void(),
     0)...,
    0};
}

template <class _Visitor,
          class... _Vs,
          typename = void_t<decltype(::cuda::std::__as_variant(::cuda::std::declval<_Vs>()))...>>
_CCCL_API constexpr decltype(auto) visit(_Visitor&& __visitor, _Vs&&... __vs)
{
  using __variant_detail::__visitation::__variant;
  ::cuda::std::__throw_if_valueless(::cuda::std::forward<_Vs>(__vs)...);
  return __variant::__visit_value(::cuda::std::forward<_Visitor>(__visitor), ::cuda::std::forward<_Vs>(__vs)...);
}

template <class _Rp,
          class _Visitor,
          class... _Vs,
          typename = void_t<decltype(::cuda::std::__as_variant(::cuda::std::declval<_Vs>()))...>>
_CCCL_API constexpr _Rp visit(_Visitor&& __visitor, _Vs&&... __vs)
{
  using __variant_detail::__visitation::__variant;
  ::cuda::std::__throw_if_valueless(::cuda::std::forward<_Vs>(__vs)...);
  return __variant::__visit_value<_Rp>(::cuda::std::forward<_Visitor>(__visitor), ::cuda::std::forward<_Vs>(__vs)...);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___VARIANT_VISIT_H
