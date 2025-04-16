//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___STRING_CONSTEXPR_C_FUNCTIONS_H
#define _LIBCUDACXX___STRING_CONSTEXPR_C_FUNCTIONS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/__type_traits/make_nbit_int.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/climits>

#if !_CCCL_COMPILER(NVRTC)
#  include <cstring>
#endif // !_CCCL_COMPILER(NVRTC)

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <class _CharT>
_LIBCUDACXX_HIDE_FROM_ABI constexpr _CharT*
__cccl_constexpr_strcpy(_CharT* _CCCL_RESTRICT __dst, const _CharT* _CCCL_RESTRICT __src) noexcept
{
  if constexpr (sizeof(_CharT) == 1)
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_HOST,
                   (return reinterpret_cast<_CharT*>(
                             ::strcpy(reinterpret_cast<char*>(__dst), reinterpret_cast<const char*>(__src)));))
    }
  }

  _CharT* __dst_it = __dst;
  while ((*__dst_it++ = *__src++) != _CharT('\0'))
  {
  }
  return __dst;
}

template <class _CharT>
_LIBCUDACXX_HIDE_FROM_ABI constexpr _CharT*
__cccl_constexpr_strncpy(_CharT* _CCCL_RESTRICT __dst, const _CharT* _CCCL_RESTRICT __src, size_t __n) noexcept
{
  if constexpr (sizeof(_CharT) == 1)
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_HOST,
                   (return reinterpret_cast<_CharT*>(
                             ::strncpy(reinterpret_cast<char*>(__dst), reinterpret_cast<const char*>(__src), __n));))
    }
  }

  _CharT* __dst_it = __dst;
  while (__n--)
  {
    if ((*__dst_it++ = *__src++) == _CharT('\0'))
    {
      while (__n--)
      {
        *__dst_it++ = _CharT('\0');
      }
      break;
    }
  }
  return __dst;
}

template <class _CharT>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr size_t __cccl_constexpr_strlen(const _CharT* __ptr) noexcept
{
  if constexpr (sizeof(_CharT) == 1)
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_HOST, (return ::strlen(reinterpret_cast<const char*>(__ptr));))
    }
  }

  size_t __len = 0;
  while (*__ptr++ != _CharT('\0'))
  {
    ++__len;
  }
  return __len;
}

template <class _CharT>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr int
__cccl_constexpr_strcmp(const _CharT* __lhs, const _CharT* __rhs) noexcept
{
  if constexpr (sizeof(_CharT) == 1)
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_HOST,
                   (return ::strcmp(reinterpret_cast<const char*>(__lhs), reinterpret_cast<const char*>(__rhs));))
    }
  }

  using _UCharT = __make_nbit_uint_t<sizeof(_CharT) * CHAR_BIT>;

  while (*__lhs == *__rhs)
  {
    if (*__lhs == _CharT('\0'))
    {
      return 0;
    }

    ++__lhs;
    ++__rhs;
  }
  return (static_cast<_UCharT>(*__lhs) < static_cast<_UCharT>(*__rhs)) ? -1 : 1;
}

template <class _CharT>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr int
__cccl_constexpr_strncmp(const _CharT* __lhs, const _CharT* __rhs, size_t __n) noexcept
{
  if constexpr (sizeof(_CharT) == 1)
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_HOST,
                   (return ::strncmp(reinterpret_cast<const char*>(__lhs), reinterpret_cast<const char*>(__rhs), __n);))
    }
  }

  using _UCharT = __make_nbit_uint_t<sizeof(_CharT) * CHAR_BIT>;

  while (__n--)
  {
    if (*__lhs != *__rhs)
    {
      return (static_cast<_UCharT>(*__lhs) < static_cast<_UCharT>(*__rhs)) ? -1 : 1;
    }

    if (*__lhs == _CharT('\0'))
    {
      return 0;
    }

    ++__lhs;
    ++__rhs;
  }
  return 0;
}

template <class _CharT>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _CharT* __cccl_constexpr_strchr(_CharT* __ptr, _CharT __c) noexcept
{
  if constexpr (sizeof(_CharT) == 1)
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_HOST,
                   (return reinterpret_cast<_CharT*>(::strchr(
                     reinterpret_cast<char*>(const_cast<remove_const_t<_CharT>*>(__ptr)), static_cast<int>(__c)));))
    }
  }

  while (*__ptr != __c)
  {
    if (*__ptr == _CharT('\0'))
    {
      return nullptr;
    }
    ++__ptr;
  }
  return __ptr;
}

template <class _CharT>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _CharT* __cccl_constexpr_strrchr(_CharT* __ptr, _CharT __c) noexcept
{
  if constexpr (sizeof(_CharT) == 1)
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_HOST,
                   (return reinterpret_cast<_CharT*>(::strrchr(
                     reinterpret_cast<char*>(const_cast<remove_const_t<_CharT>*>(__ptr)), static_cast<int>(__c)));))
    }
  }

  if (__c == _CharT('\0'))
  {
    return __ptr + _CUDA_VSTD::__cccl_constexpr_strlen(__ptr);
  }

  _CharT* __last{};
  while (*__ptr != _CharT('\0'))
  {
    if (*__ptr == __c)
    {
      __last = __ptr;
    }
    ++__ptr;
  }
  return __last;
}

template <class _Tp>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp* __cccl_constexpr_memchr(_Tp* __ptr, _Tp __c, size_t __n) noexcept
{
  if constexpr (sizeof(_Tp) == 1)
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(
        NV_IS_HOST,
        (return reinterpret_cast<_Tp*>(::memchr(const_cast<remove_const_t<_Tp>*>(__ptr), static_cast<int>(__c), __n));))
    }
  }

  while (__n--)
  {
    if (*__ptr == static_cast<unsigned char>(__c))
    {
      return __ptr;
    }
    ++__ptr;
  }
  return nullptr;
}

template <class _Tp>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp*
__cccl_constexpr_memmove(_Tp* __dst, const _Tp* __src, size_t __n) noexcept
{
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
#if defined(_CCCL_BUILTIN_MEMMOVE)
    return reinterpret_cast<_Tp*>(_CCCL_BUILTIN_MEMMOVE(__dst, __src, __n * sizeof(_Tp)));
#else // ^^^ _CCCL_BUILTIN_MEMMOVE ^^^ / vvv !_CCCL_BUILTIN_MEMMOVE vvv
    NV_IF_TARGET(NV_IS_HOST, (return reinterpret_cast<_Tp*>(::memmove(__dst, __src, __n * sizeof(_Tp)));))
#endif // ^^^ !_CCCL_BUILTIN_MEMMOVE ^^^
  }

  const auto __dst_copy = __dst;

  if (__src < __dst && __dst < __src + __n)
  {
    __dst += __n;
    __src += __n;

    while (__n-- > 0)
    {
      *--__dst = *--__src;
    }
  }
  else
  {
    while (__n-- > 0)
    {
      *__dst++ = *__src++;
    }
  }
  return __dst_copy;
}

template <class _Tp>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr int
__cccl_constexpr_memcmp(const _Tp* __lhs, const _Tp* __rhs, size_t __n) noexcept
{
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
#if defined(_CCCL_BUILTIN_MEMCMP)
    return _CCCL_BUILTIN_MEMCMP(__lhs, __rhs, __n * sizeof(_Tp));
#else // ^^^ _CCCL_BUILTIN_MEMCMP ^^^ / vvv !_CCCL_BUILTIN_MEMCMP vvv
    NV_IF_TARGET(NV_IS_HOST, (return ::memcmp(__lhs, __rhs, __n * sizeof(_Tp));))
#endif // ^^^ !_CCCL_BUILTIN_MEMCMP ^^^
  }

  while (__n--)
  {
    if (*__lhs != *__rhs)
    {
      return *__lhs < *__rhs ? -1 : 1;
    }
    ++__lhs;
    ++__rhs;
  }
  return 0;
}

template <class _Tp>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp*
__cccl_constexpr_memcpy(_Tp* _CCCL_RESTRICT __dst, const _Tp* _CCCL_RESTRICT __src, size_t __n) noexcept
{
  if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
  {
#if defined(_CCCL_BUILTIN_MEMCPY)
    return reinterpret_cast<_Tp*>(_CCCL_BUILTIN_MEMCPY(__dst, __src, __n * sizeof(_Tp)));
#else // ^^^ _CCCL_BUILTIN_MEMCPY ^^^ / vvv !_CCCL_BUILTIN_MEMCPY vvv
    NV_IF_TARGET(NV_IS_HOST, (return reinterpret_cast<_Tp*>(::memcpy(__dst, __src, __n * sizeof(_Tp)));))
#endif // ^^^ !_CCCL_BUILTIN_MEMCPY ^^^
  }

  const auto __dst_copy = __dst;

  while (__n--)
  {
    *__dst++ = *__src++;
  }
  return __dst_copy;
}

template <class _Tp>
[[nodiscard]] _LIBCUDACXX_HIDE_FROM_ABI constexpr _Tp* __cccl_constexpr_memset(_Tp* __ptr, _Tp __c, size_t __n) noexcept
{
  if constexpr (sizeof(_Tp) == 1)
  {
    if (!_CUDA_VSTD::__cccl_default_is_constant_evaluated())
    {
      NV_IF_TARGET(NV_IS_HOST,
                   (return reinterpret_cast<_Tp*>(::memset(__ptr, static_cast<int>(__c), __n * sizeof(_Tp)));))
    }
  }

  const auto __ptr_copy = __ptr;

  while (__n--)
  {
    *__ptr++ = __c;
  }
  return __ptr_copy;
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___STRING_CONSTEXPR_C_FUNCTIONS_H
