// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___MEMORY_POINTER_TRAITS_H
#define _CUDA_STD___MEMORY_POINTER_TRAITS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/derived_from.h>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/conjunction.h>
#include <cuda/std/__type_traits/decay.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_class.h>
#include <cuda/std/__type_traits/is_function.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Tp, class = void>
inline constexpr bool __has_element_type = false;

template <class _Tp>
inline constexpr bool __has_element_type<_Tp, void_t<typename _Tp::element_type>> = true;

template <class _Ptr, bool = __has_element_type<_Ptr>>
struct __pointer_traits_element_type;

template <class _Ptr>
struct __pointer_traits_element_type<_Ptr, true>
{
  using type _CCCL_NODEBUG = typename _Ptr::element_type;
};

template <template <class, class...> class _Sp, class _Tp, class... _Args>
struct __pointer_traits_element_type<_Sp<_Tp, _Args...>, true>
{
  using type _CCCL_NODEBUG = typename _Sp<_Tp, _Args...>::element_type;
};

template <template <class, class...> class _Sp, class _Tp, class... _Args>
struct __pointer_traits_element_type<_Sp<_Tp, _Args...>, false>
{
  using type _CCCL_NODEBUG = _Tp;
};

template <class _Tp, class = void>
inline constexpr bool __has_difference_type = false;

template <class _Tp>
inline constexpr bool __has_difference_type<_Tp, void_t<typename _Tp::difference_type>> = true;

template <class _Ptr, bool = __has_difference_type<_Ptr>>
struct __pointer_traits_difference_type
{
  using type _CCCL_NODEBUG = ptrdiff_t;
};

template <class _Ptr>
struct __pointer_traits_difference_type<_Ptr, true>
{
  using type _CCCL_NODEBUG = typename _Ptr::difference_type;
};

template <class _Tp, class _Up, class = void>
inline constexpr bool __has_rebind = false;

template <class _Tp, class _Up>
inline constexpr bool __has_rebind<_Tp, _Up, void_t<typename _Tp::template rebind<_Up>>> = true;

template <class _Tp, class _Up, bool = __has_rebind<_Tp, _Up>>
struct __pointer_traits_rebind
{
  using type _CCCL_NODEBUG = typename _Tp::template rebind<_Up>;
};

template <template <class, class...> class _Sp, class _Tp, class... _Args, class _Up>
struct __pointer_traits_rebind<_Sp<_Tp, _Args...>, _Up, true>
{
  using type _CCCL_NODEBUG = typename _Sp<_Tp, _Args...>::template rebind<_Up>;
};

template <template <class, class...> class _Sp, class _Tp, class... _Args, class _Up>
struct __pointer_traits_rebind<_Sp<_Tp, _Args...>, _Up, false>
{
  using type = _Sp<_Up, _Args...>;
};

template <class _Ptr, class = void>
struct __pointer_traits_impl
{};

template <class _Ptr>
struct __pointer_traits_impl<_Ptr, void_t<typename __pointer_traits_element_type<_Ptr>::type>>
{
  using pointer         = _Ptr;
  using element_type    = typename __pointer_traits_element_type<pointer>::type;
  using difference_type = typename __pointer_traits_difference_type<pointer>::type;

  template <class _Up>
  using rebind = typename __pointer_traits_rebind<pointer, _Up>::type;

private:
  struct __nat
  {};

public:
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 static pointer
  pointer_to(conditional_t<is_void_v<element_type>, __nat, element_type>& __r)
  {
    return pointer::pointer_to(__r);
  }
};

template <class _Ptr, class = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT pointer_traits : __pointer_traits_impl<_Ptr>
{};

template <class _Tp>
struct _CCCL_TYPE_VISIBILITY_DEFAULT pointer_traits<_Tp*>
{
  using pointer         = _Tp*;
  using element_type    = _Tp;
  using difference_type = ptrdiff_t;

  template <class _Up>
  using rebind = _Up*;

private:
  struct __nat
  {};

public:
  _CCCL_API inline _CCCL_CONSTEXPR_CXX20 static pointer
  pointer_to(conditional_t<is_void_v<element_type>, __nat, element_type>& __r) noexcept
  {
    return ::cuda::std::addressof(__r);
  }
};

// to_address
template <class _Pointer>
_CCCL_CONCEPT __has_cuda_std_to_address = _CCCL_REQUIRES_EXPR((_Pointer), const _Pointer& __ptr)(
  typename(typename ::cuda::std::pointer_traits<_Pointer>), (::cuda::std::pointer_traits<_Pointer>::to_address(__ptr)));

template <class _Pointer>
_CCCL_CONCEPT __has_const_operator_arrow = _CCCL_REQUIRES_EXPR((_Pointer), const _Pointer& __ptr)((__ptr.operator->()));

template <class _Pointer>
inline constexpr bool __is_fancy_pointer = __has_cuda_std_to_address<_Pointer> || __has_const_operator_arrow<_Pointer>;

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr auto to_address(_Tp* const __ptr) noexcept
{
  static_assert(!is_function_v<_Tp>, "_Tp is a function type");
  return __ptr;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Pointer, enable_if_t<__is_fancy_pointer<_Pointer>, int> = 0>
[[nodiscard]] _CCCL_API constexpr auto to_address(const _Pointer& __ptr) noexcept
{
  if constexpr (__has_cuda_std_to_address<_Pointer>)
  {
    return ::cuda::std::pointer_traits<_Pointer>::to_address(__ptr);
  }
  else
  {
    return ::cuda::std::to_address(__ptr.operator->());
  }
}

template <class _Pointer>
_CCCL_CONCEPT __can_to_address =
  _CCCL_REQUIRES_EXPR((_Pointer), const _Pointer& __ptr)((::cuda::std::to_address(__ptr)));

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___MEMORY_POINTER_TRAITS_H
