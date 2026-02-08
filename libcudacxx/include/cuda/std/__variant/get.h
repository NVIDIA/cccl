//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___VARIANT_GET_H
#define _CUDA_STD___VARIANT_GET_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__fwd/get.h>
#include <cuda/std/__fwd/variant.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__tuple_dir/get.h>
#include <cuda/std/__type_traits/add_pointer.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/__variant/bad_variant_access.h>
#include <cuda/std/__variant/variant.h>
#include <cuda/std/__variant/variant_access.h>
#include <cuda/std/__variant/variant_match.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <size_t _Ip, class... _Types>
[[nodiscard]] _CCCL_API constexpr bool __holds_alternative(const variant<_Types...>& __v) noexcept
{
  return __v.index() == _Ip;
}

template <class _Tp, class... _Types>
[[nodiscard]] _CCCL_API constexpr bool holds_alternative(const variant<_Types...>& __v) noexcept
{
  return ::cuda::std::__holds_alternative<__find_exactly_one_t<_Tp, _Types...>::value>(__v);
}

template <size_t _Ip, class _Vp>
[[nodiscard]] _CCCL_API constexpr auto&& __generic_get(_Vp&& __v)
{
  using __variant_detail::__access::__variant;
  if (!::cuda::std::__holds_alternative<_Ip>(__v))
  {
    ::cuda::std::__throw_bad_variant_access();
  }
  return __variant::__get_alt<_Ip>(::cuda::std::forward<_Vp>(__v)).__value;
}

template <size_t _Ip, class... _Types>
[[nodiscard]] _CCCL_API constexpr variant_alternative_t<_Ip, variant<_Types...>>& get(variant<_Types...>& __v)
{
  static_assert(_Ip < sizeof...(_Types), "");
  static_assert(!is_void_v<variant_alternative_t<_Ip, variant<_Types...>>>, "");
  return ::cuda::std::__generic_get<_Ip>(__v);
}

template <size_t _Ip, class... _Types>
[[nodiscard]] _CCCL_API constexpr variant_alternative_t<_Ip, variant<_Types...>>&& get(variant<_Types...>&& __v)
{
  static_assert(_Ip < sizeof...(_Types), "");
  static_assert(!is_void_v<variant_alternative_t<_Ip, variant<_Types...>>>, "");
  return ::cuda::std::__generic_get<_Ip>(::cuda::std::move(__v));
}

template <size_t _Ip, class... _Types>
[[nodiscard]] _CCCL_API constexpr const variant_alternative_t<_Ip, variant<_Types...>>&
get(const variant<_Types...>& __v)
{
  static_assert(_Ip < sizeof...(_Types), "");
  static_assert(!is_void_v<variant_alternative_t<_Ip, variant<_Types...>>>, "");
  return ::cuda::std::__generic_get<_Ip>(__v);
}

template <size_t _Ip, class... _Types>
[[nodiscard]] _CCCL_API constexpr const variant_alternative_t<_Ip, variant<_Types...>>&&
get(const variant<_Types...>&& __v)
{
  static_assert(_Ip < sizeof...(_Types), "");
  static_assert(!is_void_v<variant_alternative_t<_Ip, variant<_Types...>>>, "");
  return ::cuda::std::__generic_get<_Ip>(::cuda::std::move(__v));
}

template <class _Tp, class... _Types>
[[nodiscard]] _CCCL_API constexpr _Tp& get(variant<_Types...>& __v)
{
  static_assert(!is_void_v<_Tp>, "");
  return ::cuda::std::get<__find_exactly_one_t<_Tp, _Types...>::value>(__v);
}

template <class _Tp, class... _Types>
[[nodiscard]] _CCCL_API constexpr _Tp&& get(variant<_Types...>&& __v)
{
  static_assert(!is_void_v<_Tp>, "");
  return ::cuda::std::get<__find_exactly_one_t<_Tp, _Types...>::value>(::cuda::std::move(__v));
}

template <class _Tp, class... _Types>
[[nodiscard]] _CCCL_API constexpr const _Tp& get(const variant<_Types...>& __v)
{
  static_assert(!is_void_v<_Tp>, "");
  return ::cuda::std::get<__find_exactly_one_t<_Tp, _Types...>::value>(__v);
}

template <class _Tp, class... _Types>
[[nodiscard]] _CCCL_API constexpr const _Tp&& get(const variant<_Types...>&& __v)
{
  static_assert(!is_void_v<_Tp>, "");
  return ::cuda::std::get<__find_exactly_one_t<_Tp, _Types...>::value>(::cuda::std::move(__v));
}

template <size_t _Ip, class _Vp>
[[nodiscard]] _CCCL_API constexpr auto* __generic_get_if(_Vp* __v) noexcept
{
  using __variant_detail::__access::__variant;
  return __v && ::cuda::std::__holds_alternative<_Ip>(*__v)
         ? ::cuda::std::addressof(__variant::__get_alt<_Ip>(*__v).__value)
         : nullptr;
}

template <size_t _Ip, class... _Types>
[[nodiscard]] _CCCL_API constexpr add_pointer_t<variant_alternative_t<_Ip, variant<_Types...>>>
get_if(variant<_Types...>* __v) noexcept
{
  static_assert(_Ip < sizeof...(_Types), "");
  static_assert(!is_void_v<variant_alternative_t<_Ip, variant<_Types...>>>, "");
  return ::cuda::std::__generic_get_if<_Ip>(__v);
}

template <size_t _Ip, class... _Types>
[[nodiscard]] _CCCL_API constexpr add_pointer_t<const variant_alternative_t<_Ip, variant<_Types...>>>
get_if(const variant<_Types...>* __v) noexcept
{
  static_assert(_Ip < sizeof...(_Types), "");
  static_assert(!is_void_v<variant_alternative_t<_Ip, variant<_Types...>>>, "");
  return ::cuda::std::__generic_get_if<_Ip>(__v);
}

template <class _Tp, class... _Types>
[[nodiscard]] _CCCL_API constexpr add_pointer_t<_Tp> get_if(variant<_Types...>* __v) noexcept
{
  static_assert(!is_void_v<_Tp>, "");
  return ::cuda::std::get_if<__find_exactly_one_t<_Tp, _Types...>::value>(__v);
}

template <class _Tp, class... _Types>
[[nodiscard]] _CCCL_API constexpr add_pointer_t<const _Tp> get_if(const variant<_Types...>* __v) noexcept
{
  static_assert(!is_void_v<_Tp>, "");
  return ::cuda::std::get_if<__find_exactly_one_t<_Tp, _Types...>::value>(__v);
}

// __unchecked_get is the same as ::cuda::std::get, except, it is UB to use it
// with the wrong type whereas ::cuda::std::get will throw or returning nullptr.
// This makes it faster than ::cuda::std::get.
template <size_t _Ip, class _Vp>
[[nodiscard]] _CCCL_API constexpr auto&& __unchecked_get(_Vp&& __v) noexcept
{
  using __variant_detail::__access::__variant;
  return __variant::__get_alt<_Ip>(::cuda::std::forward<_Vp>(__v)).__value;
}

template <class _Tp, class... _Types>
[[nodiscard]] _CCCL_API constexpr auto&& __unchecked_get(const variant<_Types...>& __v) noexcept
{
  return ::cuda::std::__unchecked_get<__find_exactly_one_t<_Tp, _Types...>::value>(__v);
}

template <class _Tp, class... _Types>
[[nodiscard]] _CCCL_API constexpr auto&& __unchecked_get(variant<_Types...>& __v) noexcept
{
  return ::cuda::std::__unchecked_get<__find_exactly_one_t<_Tp, _Types...>::value>(__v);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___VARIANT_GET_H
