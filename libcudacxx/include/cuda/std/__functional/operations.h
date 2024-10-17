// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___FUNCTIONAL_OPERATIONS_H
#define _LIBCUDACXX___FUNCTIONAL_OPERATIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__functional/address_stability.h>
#include <cuda/std/__functional/binary_function.h>
#include <cuda/std/__functional/unary_function.h>
#include <cuda/std/__type_traits/conjunction.h>
#include <cuda/std/__type_traits/is_class.h>
#include <cuda/std/__type_traits/is_enum.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__utility/forward.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Arithmetic operations

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT plus : __binary_function<_Tp, _Tp, _Tp>
{
  typedef _Tp __result_type; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  constexpr _LIBCUDACXX_HIDE_FROM_ABI _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x + __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(plus);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT plus<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  constexpr _LIBCUDACXX_HIDE_FROM_ABI auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) + _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) + _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) + _CUDA_VSTD::forward<_T2>(__u);
  }
  typedef void is_transparent;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT minus : __binary_function<_Tp, _Tp, _Tp>
{
  typedef _Tp __result_type; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  constexpr _LIBCUDACXX_HIDE_FROM_ABI _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x - __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(minus);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT minus<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  constexpr _LIBCUDACXX_HIDE_FROM_ABI auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) - _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) - _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) - _CUDA_VSTD::forward<_T2>(__u);
  }
  typedef void is_transparent;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT multiplies : __binary_function<_Tp, _Tp, _Tp>
{
  typedef _Tp __result_type; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  constexpr _LIBCUDACXX_HIDE_FROM_ABI _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x * __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(multiplies);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT multiplies<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  constexpr _LIBCUDACXX_HIDE_FROM_ABI auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) * _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) * _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) * _CUDA_VSTD::forward<_T2>(__u);
  }
  typedef void is_transparent;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT divides : __binary_function<_Tp, _Tp, _Tp>
{
  typedef _Tp __result_type; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  constexpr _LIBCUDACXX_HIDE_FROM_ABI _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x / __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(divides);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT divides<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  constexpr _LIBCUDACXX_HIDE_FROM_ABI auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) / _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) / _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) / _CUDA_VSTD::forward<_T2>(__u);
  }
  typedef void is_transparent;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT modulus : __binary_function<_Tp, _Tp, _Tp>
{
  typedef _Tp __result_type; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  constexpr _LIBCUDACXX_HIDE_FROM_ABI _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x % __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(modulus);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT modulus<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  constexpr _LIBCUDACXX_HIDE_FROM_ABI auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) % _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) % _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) % _CUDA_VSTD::forward<_T2>(__u);
  }
  typedef void is_transparent;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT negate : __unary_function<_Tp, _Tp>
{
  typedef _Tp __result_type; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_HIDE_FROM_ABI _Tp operator()(const _Tp& __x) const
  {
    return -__x;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(negate);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT negate<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp>
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_HIDE_FROM_ABI auto operator()(_Tp&& __x) const
    noexcept(noexcept(-_CUDA_VSTD::forward<_Tp>(__x))) -> decltype(-_CUDA_VSTD::forward<_Tp>(__x))
  {
    return -_CUDA_VSTD::forward<_Tp>(__x);
  }
  typedef void is_transparent;
};

// Bitwise operations

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT bit_and : __binary_function<_Tp, _Tp, _Tp>
{
  typedef _Tp __result_type; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  constexpr _LIBCUDACXX_HIDE_FROM_ABI _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x & __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(bit_and);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT bit_and<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  constexpr _LIBCUDACXX_HIDE_FROM_ABI auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) & _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) & _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) & _CUDA_VSTD::forward<_T2>(__u);
  }
  typedef void is_transparent;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT bit_not : __unary_function<_Tp, _Tp>
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_HIDE_FROM_ABI _Tp operator()(const _Tp& __x) const
  {
    return ~__x;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(bit_not);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT bit_not<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp>
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_HIDE_FROM_ABI auto operator()(_Tp&& __x) const
    noexcept(noexcept(~_CUDA_VSTD::forward<_Tp>(__x))) -> decltype(~_CUDA_VSTD::forward<_Tp>(__x))
  {
    return ~_CUDA_VSTD::forward<_Tp>(__x);
  }
  typedef void is_transparent;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT bit_or : __binary_function<_Tp, _Tp, _Tp>
{
  typedef _Tp __result_type; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_HIDE_FROM_ABI _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x | __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(bit_or);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT bit_or<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  constexpr _LIBCUDACXX_HIDE_FROM_ABI auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) | _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) | _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) | _CUDA_VSTD::forward<_T2>(__u);
  }
  typedef void is_transparent;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT bit_xor : __binary_function<_Tp, _Tp, _Tp>
{
  typedef _Tp __result_type; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  constexpr _LIBCUDACXX_HIDE_FROM_ABI _Tp operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x ^ __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(bit_xor);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT bit_xor<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  constexpr _LIBCUDACXX_HIDE_FROM_ABI auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) ^ _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) ^ _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) ^ _CUDA_VSTD::forward<_T2>(__u);
  }
  typedef void is_transparent;
};

// Comparison operations

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT equal_to : __binary_function<_Tp, _Tp, bool>
{
  typedef bool __result_type; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  constexpr _LIBCUDACXX_HIDE_FROM_ABI bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x == __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(equal_to);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT equal_to<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  constexpr _LIBCUDACXX_HIDE_FROM_ABI auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) == _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) == _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) == _CUDA_VSTD::forward<_T2>(__u);
  }
  typedef void is_transparent;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT not_equal_to : __binary_function<_Tp, _Tp, bool>
{
  typedef bool __result_type; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  constexpr _LIBCUDACXX_HIDE_FROM_ABI bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x != __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(not_equal_to);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT not_equal_to<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  constexpr _LIBCUDACXX_HIDE_FROM_ABI auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) != _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) != _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) != _CUDA_VSTD::forward<_T2>(__u);
  }
  typedef void is_transparent;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT less : __binary_function<_Tp, _Tp, bool>
{
  typedef bool __result_type; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  constexpr _LIBCUDACXX_HIDE_FROM_ABI bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x < __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(less);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT less<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  constexpr _LIBCUDACXX_HIDE_FROM_ABI auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) < _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) < _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) < _CUDA_VSTD::forward<_T2>(__u);
  }
  typedef void is_transparent;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT less_equal : __binary_function<_Tp, _Tp, bool>
{
  typedef bool __result_type; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  constexpr _LIBCUDACXX_HIDE_FROM_ABI bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x <= __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(less_equal);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT less_equal<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  constexpr _LIBCUDACXX_HIDE_FROM_ABI auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) <= _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) <= _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) <= _CUDA_VSTD::forward<_T2>(__u);
  }
  typedef void is_transparent;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT greater_equal : __binary_function<_Tp, _Tp, bool>
{
  typedef bool __result_type; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  constexpr _LIBCUDACXX_HIDE_FROM_ABI bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x >= __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(greater_equal);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT greater_equal<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  constexpr _LIBCUDACXX_HIDE_FROM_ABI auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) >= _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) >= _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) >= _CUDA_VSTD::forward<_T2>(__u);
  }
  typedef void is_transparent;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT greater : __binary_function<_Tp, _Tp, bool>
{
  typedef bool __result_type; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  constexpr _LIBCUDACXX_HIDE_FROM_ABI bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x > __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(greater);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT greater<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  constexpr _LIBCUDACXX_HIDE_FROM_ABI auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) > _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) > _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) > _CUDA_VSTD::forward<_T2>(__u);
  }
  typedef void is_transparent;
};

// Logical operations

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT logical_and : __binary_function<_Tp, _Tp, bool>
{
  typedef bool __result_type; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  constexpr _LIBCUDACXX_HIDE_FROM_ABI bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x && __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(logical_and);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT logical_and<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  constexpr _LIBCUDACXX_HIDE_FROM_ABI auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) && _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) && _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) && _CUDA_VSTD::forward<_T2>(__u);
  }
  typedef void is_transparent;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT logical_not : __unary_function<_Tp, bool>
{
  typedef bool __result_type; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  constexpr _LIBCUDACXX_HIDE_FROM_ABI bool operator()(const _Tp& __x) const
  {
    return !__x;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(logical_not);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT logical_not<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _Tp>
  _CCCL_CONSTEXPR_CXX14 _LIBCUDACXX_HIDE_FROM_ABI auto operator()(_Tp&& __x) const
    noexcept(noexcept(!_CUDA_VSTD::forward<_Tp>(__x))) -> decltype(!_CUDA_VSTD::forward<_Tp>(__x))
  {
    return !_CUDA_VSTD::forward<_Tp>(__x);
  }
  typedef void is_transparent;
};

template <class _Tp = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT logical_or : __binary_function<_Tp, _Tp, bool>
{
  typedef bool __result_type; // used by valarray
  _CCCL_EXEC_CHECK_DISABLE
  constexpr _LIBCUDACXX_HIDE_FROM_ABI bool operator()(const _Tp& __x, const _Tp& __y) const
  {
    return __x || __y;
  }
};
_LIBCUDACXX_CTAD_SUPPORTED_FOR_TYPE(logical_or);

template <>
struct _CCCL_TYPE_VISIBILITY_DEFAULT logical_or<void>
{
  _CCCL_EXEC_CHECK_DISABLE
  template <class _T1, class _T2>
  constexpr _LIBCUDACXX_HIDE_FROM_ABI auto operator()(_T1&& __t, _T2&& __u) const
    noexcept(noexcept(_CUDA_VSTD::forward<_T1>(__t) || _CUDA_VSTD::forward<_T2>(__u)))
      -> decltype(_CUDA_VSTD::forward<_T1>(__t) || _CUDA_VSTD::forward<_T2>(__u))
  {
    return _CUDA_VSTD::forward<_T1>(__t) || _CUDA_VSTD::forward<_T2>(__u);
  }
  typedef void is_transparent;
};

_LIBCUDACXX_END_NAMESPACE_STD

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

template <typename _T>
struct __has_builtin_operators
    : _CUDA_VSTD::bool_constant<!_CUDA_VSTD::is_class<_T>::value && !_CUDA_VSTD::is_enum<_T>::value
                                && !_CUDA_VSTD::is_void<_T>::value>
{};

#define _LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(functor)                                               \
  /*we know what plus<T> etc. do if T is not a type that could have a weird operatorX() */         \
  template <typename _T, typename... _Args>                                                        \
  struct __allows_copied_arguments_impl<functor<_T>, void, _Args...> : __has_builtin_operators<_T> \
  {};                                                                                              \
  /*we know what plus<void> etc. do if T is not a type that could have a weird operatorX() */      \
  template <typename... _Args>                                                                     \
  struct __allows_copied_arguments_impl<functor<void>, void, _Args...>                             \
      : _CUDA_VSTD::conjunction<__has_builtin_operators<_Args>...>                                 \
  {};

_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(_CUDA_VSTD::plus);
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(_CUDA_VSTD::minus);
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(_CUDA_VSTD::multiplies);
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(_CUDA_VSTD::divides);
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(_CUDA_VSTD::modulus);
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(_CUDA_VSTD::negate);
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(_CUDA_VSTD::bit_and);
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(_CUDA_VSTD::bit_not);
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(_CUDA_VSTD::bit_or);
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(_CUDA_VSTD::bit_xor);
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(_CUDA_VSTD::equal_to);
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(_CUDA_VSTD::not_equal_to);
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(_CUDA_VSTD::less);
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(_CUDA_VSTD::less_equal);
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(_CUDA_VSTD::greater_equal);
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(_CUDA_VSTD::greater);
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(_CUDA_VSTD::logical_and);
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(_CUDA_VSTD::logical_not);
_LIBCUDACXX_MARK_CAN_COPY_ARGUMENTS(_CUDA_VSTD::logical_or);

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _LIBCUDACXX___FUNCTIONAL_OPERATIONS_H
