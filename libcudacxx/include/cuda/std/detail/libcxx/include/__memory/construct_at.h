// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MEMORY_CONSTRUCT_AT_H
#define _LIBCUDACXX___MEMORY_CONSTRUCT_AT_H

#ifndef __cuda_std__
#  include <__config>
#endif //__cuda_std__

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include "../__assert"
#include "../__concepts/__concept_macros.h"
#include "../__iterator/access.h"
#include "../__memory/addressof.h"
#include "../__memory/voidify.h"
#include "../__type_traits/enable_if.h"
#include "../__type_traits/integral_constant.h"
#include "../__type_traits/is_arithmetic.h"
#include "../__type_traits/is_array.h"
#include "../__type_traits/is_constant_evaluated.h"
#include "../__type_traits/is_trivially_move_assignable.h"
#include "../__type_traits/is_trivially_constructible.h"
#include "../__type_traits/void_t.h"
#include "../__utility/declval.h"
#include "../__utility/forward.h"
#include "../__utility/move.h"

#ifdef _CCCL_CUDA_COMPILER_CLANG
#  include <new>
#endif // _CCCL_CUDA_COMPILER_CLANG

#if defined(__cuda_std__) && _CCCL_STD_VER > 2017 // need to backfill ::std::construct_at
#  ifndef _CCCL_COMPILER_NVRTC
#    include <memory>
#  endif // _CCCL_COMPILER_NVRTC

#  ifndef __cpp_lib_constexpr_dynamic_alloc
namespace std
{
template <class _Tp,
          class... _Args,
          class = decltype(::new(_CUDA_VSTD::declval<void*>()) _Tp(_CUDA_VSTD::declval<_Args>()...))>
_LIBCUDACXX_INLINE_VISIBILITY constexpr _Tp* construct_at(_Tp* __location, _Args&&... __args)
{
#    if defined(_LIBCUDACXX_ADDRESSOF)
  return ::new (_CUDA_VSTD::__voidify(*__location)) _Tp(_CUDA_VSTD::forward<_Args>(__args)...);
#    else
  return ::new (const_cast<void*>(static_cast<const volatile void*>(__location)))
    _Tp(_CUDA_VSTD::forward<_Args>(__args)...);
#    endif
}
} // namespace std
#  endif // __cpp_lib_constexpr_dynamic_alloc
#endif // __cuda_std__ && _CCCL_STD_VER > 2017

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// There is a performance issue with placement new, where EDG based compiler insert a nullptr check that is superfluous
// Because this is a noticable performance regression, we specialize it for certain types
// This is possible because we are calling ::new ignoring any user defined overloads of operator placement new
namespace __detail
{
// We cannot allow narrowing conversions between arithmetic types as the assignment will give errors
template <class _To, class...>
struct __is_narrowing_impl : false_type
{};

template <class _To, class _From>
struct __is_narrowing_impl<_To, _From> : true_type
{};

// This is a bit hacky, but we rely on the fact that arithmetic types cannot have more than one argument to their constructor
template <class _To, class _From>
struct __is_narrowing_impl<_To, _From, __void_t<decltype(_To{_CUDA_VSTD::declval<_From>()})>> : false_type
{};

template <class _Tp, class... _Args>
using __is_narrowing = _If<_LIBCUDACXX_TRAIT(is_arithmetic, _Tp), __is_narrowing_impl<_Tp, _Args...>, false_type>;

// The destination type must be trivially constructible from the arguments and also trivially assignable, because we
// technically move assign in the optimization
template <class _Tp, class... _Args>
struct __can_optimize_construct_at
    : integral_constant<bool,
                        _LIBCUDACXX_TRAIT(is_trivially_constructible, _Tp, _Args...)
                          && _LIBCUDACXX_TRAIT(is_trivially_move_assignable, _Tp)
                          && !__is_narrowing<_Tp, _Args...>::value>
{};
} // namespace __detail

// construct_at
#if _CCCL_STD_VER > 2017

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp,
          class... _Args,
          class = decltype(::new(_CUDA_VSTD::declval<void*>()) _Tp(_CUDA_VSTD::declval<_Args>()...))>
_LIBCUDACXX_INLINE_VISIBILITY
  _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 __enable_if_t<!__detail::__can_optimize_construct_at<_Tp, _Args...>::value, _Tp*>
  construct_at(_Tp* __location, _Args&&... __args)
{
  _LIBCUDACXX_ASSERT(__location != nullptr, "null pointer given to construct_at");
#  if defined(__cuda_std__)
  // Need to go through `std::construct_at` as that is the explicitly blessed function
  if (__libcpp_is_constant_evaluated())
  {
    return ::std::construct_at(__location, _CUDA_VSTD::forward<_Args>(__args)...);
  }
#  endif // __cuda_std__
  return ::new (_CUDA_VSTD::__voidify(*__location)) _Tp(_CUDA_VSTD::forward<_Args>(__args)...);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp,
          class... _Args,
          class = decltype(::new(_CUDA_VSTD::declval<void*>()) _Tp(_CUDA_VSTD::declval<_Args>()...))>
_LIBCUDACXX_INLINE_VISIBILITY
  _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 __enable_if_t<__detail::__can_optimize_construct_at<_Tp, _Args...>::value, _Tp*>
  construct_at(_Tp* __location, _Args&&... __args)
{
  _LIBCUDACXX_ASSERT(__location != nullptr, "null pointer given to construct_at");
#  if defined(__cuda_std__)
  // Need to go through `std::construct_at` as that is the explicitly blessed function
  if (__libcpp_is_constant_evaluated())
  {
    return ::std::construct_at(__location, _CUDA_VSTD::forward<_Args>(__args)...);
  }
  *__location = _Tp{_CUDA_VSTD::forward<_Args>(__args)...};
  return __location;
#  else // ^^^ __cuda_std__ ^^^ / vvv !__cuda_std__ vvv
  // NVCC always considers construction + move assignment, other compilers are smarter using copy construction
  // So rather than adding all kinds of workarounds simply fall back to the correct implementation for libcxx mode
  return ::new (_CUDA_VSTD::__voidify(*__location)) _Tp(_CUDA_VSTD::forward<_Args>(__args)...);
#  endif // !__cuda_std__
}

#endif // _CCCL_STD_VER > 2017

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class... _Args>
_LIBCUDACXX_INLINE_VISIBILITY
  _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 __enable_if_t<!__detail::__can_optimize_construct_at<_Tp, _Args...>::value, _Tp*>
  __construct_at(_Tp* __location, _Args&&... __args)
{
  _LIBCUDACXX_ASSERT(__location != nullptr, "null pointer given to construct_at");
#if defined(__cuda_std__) && _CCCL_STD_VER > 2017
  // Need to go through `std::construct_at` as that is the explicitly blessed function
  if (__libcpp_is_constant_evaluated())
  {
    return ::std::construct_at(__location, _CUDA_VSTD::forward<_Args>(__args)...);
  }
#endif // __cuda_std__ && _CCCL_STD_VER > 2017
  return ::new (_CUDA_VSTD::__voidify(*__location)) _Tp(_CUDA_VSTD::forward<_Args>(__args)...);
}

_CCCL_EXEC_CHECK_DISABLE
template <class _Tp, class... _Args>
_LIBCUDACXX_INLINE_VISIBILITY
  _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 __enable_if_t<__detail::__can_optimize_construct_at<_Tp, _Args...>::value, _Tp*>
  __construct_at(_Tp* __location, _Args&&... __args)
{
  _LIBCUDACXX_ASSERT(__location != nullptr, "null pointer given to construct_at");
#if defined(__cuda_std__) && _CCCL_STD_VER > 2017
  // Need to go through `std::construct_at` as that is the explicitly blessed function
  if (__libcpp_is_constant_evaluated())
  {
    return ::std::construct_at(__location, _CUDA_VSTD::forward<_Args>(__args)...);
  }
#endif // __cuda_std__ && _CCCL_STD_VER > 2017
  *__location = _Tp{_CUDA_VSTD::forward<_Args>(__args)...};
  return __location;
}

// destroy_at

// The internal functions are available regardless of the language version (with the exception of the `__destroy_at`
// taking an array).
template <class _ForwardIterator>
_LIBCUDACXX_INLINE_VISIBILITY
  _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 _ForwardIterator __destroy(_ForwardIterator, _ForwardIterator);

template <class _Tp, __enable_if_t<!is_array<_Tp>::value, int> = 0>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 void __destroy_at(_Tp* __loc)
{
  _LIBCUDACXX_ASSERT(__loc != nullptr, "null pointer given to destroy_at");
  __loc->~_Tp();
}

#if _CCCL_STD_VER > 2017
template <class _Tp, __enable_if_t<is_array<_Tp>::value, int> = 0>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 void __destroy_at(_Tp* __loc)
{
  _LIBCUDACXX_ASSERT(__loc != nullptr, "null pointer given to destroy_at");
  _CUDA_VSTD::__destroy(_CUDA_VSTD::begin(*__loc), _CUDA_VSTD::end(*__loc));
}
#endif

template <class _ForwardIterator>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 _ForwardIterator
__destroy(_ForwardIterator __first, _ForwardIterator __last)
{
  for (; __first != __last; ++__first)
  {
    _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(*__first));
  }
  return __first;
}

template <class _BidirectionalIterator>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 _BidirectionalIterator
__reverse_destroy(_BidirectionalIterator __first, _BidirectionalIterator __last)
{
  while (__last != __first)
  {
    --__last;
    _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(*__last));
  }
  return __last;
}

#if _CCCL_STD_VER > 2014

template <class _Tp, enable_if_t<!is_array_v<_Tp>, int> = 0>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 void destroy_at(_Tp* __loc)
{
  _LIBCUDACXX_ASSERT(__loc != nullptr, "null pointer given to destroy_at");
  __loc->~_Tp();
}

#  if _CCCL_STD_VER > 2017
template <class _Tp, enable_if_t<is_array_v<_Tp>, int> = 0>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 void destroy_at(_Tp* __loc)
{
  _CUDA_VSTD::__destroy_at(__loc);
}
#  endif // _CCCL_STD_VER > 2017

template <class _ForwardIterator>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 void
destroy(_ForwardIterator __first, _ForwardIterator __last)
{
  (void) _CUDA_VSTD::__destroy(_CUDA_VSTD::move(__first), _CUDA_VSTD::move(__last));
}

template <class _ForwardIterator, class _Size>
_LIBCUDACXX_INLINE_VISIBILITY _LIBCUDACXX_CONSTEXPR_AFTER_CXX17 _ForwardIterator
destroy_n(_ForwardIterator __first, _Size __n)
{
  for (; __n > 0; (void) ++__first, --__n)
  {
    _CUDA_VSTD::__destroy_at(_CUDA_VSTD::addressof(*__first));
  }
  return __first;
}

#endif // _CCCL_STD_VER > 2014

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___MEMORY_CONSTRUCT_AT_H
