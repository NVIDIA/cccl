// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___MEMORY_CONSTRUCT_AT_H
#define _CUDA_STD___MEMORY_CONSTRUCT_AT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__iterator/access.h>
#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__memory/voidify.h>
#include <cuda/std/__new/device_new.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_array.h>
#include <cuda/std/__type_traits/is_constructible.h>
#include <cuda/std/__type_traits/is_trivially_constructible.h>
#include <cuda/std/__type_traits/is_trivially_destructible.h>
#include <cuda/std/__type_traits/is_trivially_move_assignable.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/__utility/move.h>

#if _CCCL_STD_VER >= 2020 // need to backfill ::std::construct_at
#  include <cuda/std/__cccl/memory_wrapper.h>

#  ifndef __cpp_lib_constexpr_dynamic_alloc
namespace std
{
_CCCL_EXEC_CHECK_DISABLE
template <class _Tp,
          class... _Args,
          class = decltype(::new(::cuda::std::declval<void*>()) _Tp(::cuda::std::declval<_Args>()...))>
_CCCL_API constexpr _Tp* construct_at(_Tp* __location, _Args&&... __args)
{
#    if defined(_CCCL_BUILTIN_ADDRESSOF)
  return ::new (::cuda::std::__voidify(*__location)) _Tp(::cuda::std::forward<_Args>(__args)...);
#    else
  return ::new (const_cast<void*>(static_cast<const volatile void*>(__location)))
    _Tp(::cuda::std::forward<_Args>(__args)...);
#    endif
}
} // namespace std
#  endif // __cpp_lib_constexpr_dynamic_alloc
#endif // _CCCL_STD_VER >= 2020

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// There is a performance issue with placement new, where EDG based compiler insert a nullptr check that is superfluous
// Because this is a noticeable performance regression, we specialize it for certain types
// This is possible because we are calling ::new ignoring any user defined overloads of operator placement new
namespace __detail
{
#if _CCCL_COMPILER(NVHPC, <, 25, 5) // NVHPC has issues determining the narrowing conversions
template <class _To, class...>
struct __check_narrowing : true_type
{};

template <class _To, class _From>
struct __check_narrowing<_To, _From> : false_type
{};

// This is a bit hacky, but we rely on the fact that arithmetic types cannot have more than one argument to their
// constructor
template <class _To, class _From>
struct __check_narrowing<_To, _From, void_t<decltype(_To{::cuda::std::declval<_From>()})>> : true_type
{};
#else // ^^^ _CCCL_COMPILER(NVHPC, <, 25, 5) ^^^ / vvv !_CCCL_COMPILER(NVHPC, <, 25, 5) vvv
// We cannot allow narrowing conversions between arithmetic types as the assignment will generate an error
template <class _Tp, class... _Args>
using __check_narrowing =
  conditional_t<sizeof...(_Args) == 1, __cccl_internal::__is_non_narrowing_convertible<_Tp, _Args...>, true_type>;
#endif // !_CCCL_COMPILER(NVHPC, <, 25, 5)

template <class _Tp, class... _Args>
_CCCL_CONCEPT __can_optimize_construct_at = _CCCL_REQUIRES_EXPR((_Tp, variadic _Args))(
  requires(is_trivially_constructible_v<_Tp, _Args...>),
  requires(is_trivially_move_assignable_v<_Tp>),
  requires(__check_narrowing<_Tp, _Args...>::value));
} // namespace __detail

// construct_at
#if _CCCL_STD_VER >= 2020

_CCCL_EXEC_CHECK_DISABLE
_CCCL_TEMPLATE(class _Tp, class... _Args)
_CCCL_REQUIRES(is_constructible_v<_Tp, _Args...>)
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 _Tp* construct_at(_Tp* __location, _Args&&... __args)
{
  _CCCL_ASSERT(__location != nullptr, "null pointer given to construct_at");
  // Need to go through `std::construct_at` as that is the explicitly blessed function
  _CCCL_IF_CONSTEVAL
  {
    return ::std::construct_at(__location, ::cuda::std::forward<_Args>(__args)...);
  }
  if constexpr (__detail::__can_optimize_construct_at<_Tp, _Args...>)
  {
    *__location = _Tp{::cuda::std::forward<_Args>(__args)...};
    return __location;
  }
  else
  {
    return ::new (::cuda::std::__voidify(*__location)) _Tp(::cuda::std::forward<_Args>(__args)...);
  }
}
#endif // _CCCL_STD_VER >= 2020

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class... _Args>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 _Tp* __construct_at(_Tp* __location, _Args&&... __args)
{
  _CCCL_ASSERT(__location != nullptr, "null pointer given to construct_at");
#if _CCCL_STD_VER >= 2020
  // Need to go through `std::construct_at` as that is the explicitly blessed function
  _CCCL_IF_CONSTEVAL
  {
    return ::std::construct_at(__location, ::cuda::std::forward<_Args>(__args)...);
  }
#endif // _CCCL_STD_VER >= 2020
  if constexpr (__detail::__can_optimize_construct_at<_Tp, _Args...>)
  {
    *__location = _Tp{::cuda::std::forward<_Args>(__args)...};
    return __location;
  }
  else
  {
    return ::new (::cuda::std::__voidify(*__location)) _Tp(::cuda::std::forward<_Args>(__args)...);
  }
}

// destroy_at

// The internal functions are available regardless of the language version (with the exception of the `__destroy_at`
// taking an array).
template <class _ForwardIterator>
_CCCL_API constexpr _ForwardIterator __destroy(_ForwardIterator, _ForwardIterator);

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp>
_CCCL_API constexpr void __destroy_at([[maybe_unused]] _Tp* __loc)
{
  _CCCL_ASSERT(__loc != nullptr, "null pointer given to __destroy_at");
  if constexpr (is_trivially_destructible_v<_Tp>)
  {
    return;
  }
  else if constexpr (is_array_v<_Tp>)
  {
    ::cuda::std::__destroy(::cuda::std::begin(*__loc), ::cuda::std::end(*__loc));
  }
  else
  {
    __loc->~_Tp();
  }
}

_CCCL_EXEC_CHECK_DISABLE
template <class _ForwardIterator>
_CCCL_API constexpr _ForwardIterator __destroy(_ForwardIterator __first, _ForwardIterator __last)
{
  for (; __first != __last; ++__first)
  {
    ::cuda::std::__destroy_at(::cuda::std::addressof(*__first));
  }
  return __first;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _BidirectionalIterator>
_CCCL_API constexpr _BidirectionalIterator
__reverse_destroy(_BidirectionalIterator __first, _BidirectionalIterator __last)
{
  while (__last != __first)
  {
    --__last;
    ::cuda::std::__destroy_at(::cuda::std::addressof(*__last));
  }
  return __last;
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 void destroy_at([[maybe_unused]] _Tp* __loc)
{
  _CCCL_ASSERT(__loc != nullptr, "null pointer given to __destroy_at");
  if constexpr (is_trivially_destructible_v<_Tp>)
  {
    return;
  }
  else if constexpr (is_array_v<_Tp>)
  {
    ::cuda::std::__destroy(::cuda::std::begin(*__loc), ::cuda::std::end(*__loc));
  }
  else
  {
    __loc->~_Tp();
  }
}

template <class _ForwardIterator>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 void destroy(_ForwardIterator __first, _ForwardIterator __last) noexcept
{
  (void) ::cuda::std::__destroy(::cuda::std::move(__first), ::cuda::std::move(__last));
}

_CCCL_EXEC_CHECK_DISABLE
template <class _ForwardIterator, class _Size>
_CCCL_API inline _CCCL_CONSTEXPR_CXX20 _ForwardIterator destroy_n(_ForwardIterator __first, _Size __n)
{
  for (; __n > 0; (void) ++__first, --__n)
  {
    ::cuda::std::__destroy_at(::cuda::std::addressof(*__first));
  }
  return __first;
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___MEMORY_CONSTRUCT_AT_H
