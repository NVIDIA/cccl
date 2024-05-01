//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CCCL_SEQUENCE_ACCESS_H
#define __CCCL_SEQUENCE_ACCESS_H

#include <cuda/std/__cccl/compiler.h>
#include <cuda/std/__cccl/system_header.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// We need to define hidden _CCCL_NODISCARD_FRIEND _LIBCUDACXX_INLINE_VISIBILITYs for {cr,r,}{begin,end} of our
// containers as we will otherwise encounter ambigouities
#define _CCCL_SYNTHESIZE_SEQUENCE_ACCESS(_ClassName, _ConstIter)                                                 \
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_INLINE_VISIBILITY iterator begin(_ClassName& __sequence) noexcept(          \
    noexcept(__sequence.begin()))                                                                                \
  {                                                                                                              \
    return __sequence.begin();                                                                                   \
  }                                                                                                              \
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_INLINE_VISIBILITY _ConstIter begin(const _ClassName& __sequence) noexcept(  \
    noexcept(__sequence.begin()))                                                                                \
  {                                                                                                              \
    return __sequence.begin();                                                                                   \
  }                                                                                                              \
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_INLINE_VISIBILITY iterator end(_ClassName& __sequence) noexcept(            \
    noexcept(__sequence.end()))                                                                                  \
  {                                                                                                              \
    return __sequence.end();                                                                                     \
  }                                                                                                              \
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_INLINE_VISIBILITY _ConstIter end(const _ClassName& __sequence) noexcept(    \
    noexcept(__sequence.end()))                                                                                  \
  {                                                                                                              \
    return __sequence.end();                                                                                     \
  }                                                                                                              \
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_INLINE_VISIBILITY _ConstIter cbegin(const _ClassName& __sequence) noexcept( \
    noexcept(__sequence.begin()))                                                                                \
  {                                                                                                              \
    return __sequence.begin();                                                                                   \
  }                                                                                                              \
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_INLINE_VISIBILITY _ConstIter cend(const _ClassName& __sequence) noexcept(   \
    noexcept(__sequence.end()))                                                                                  \
  {                                                                                                              \
    return __sequence.end();                                                                                     \
  }
#define _CCCL_SYNTHESIZE_SEQUENCE_REVERSE_ACCESS(_ClassName, _ConstIter)                                          \
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_INLINE_VISIBILITY reverse_iterator rbegin(_ClassName& __sequence) noexcept(  \
    noexcept(__sequence.rbegin()))                                                                                \
  {                                                                                                               \
    return __sequence.rbegin();                                                                                   \
  }                                                                                                               \
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_INLINE_VISIBILITY _ConstIter rbegin(const _ClassName& __sequence) noexcept(  \
    noexcept(__sequence.rbegin()))                                                                                \
  {                                                                                                               \
    return __sequence.rbegin();                                                                                   \
  }                                                                                                               \
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_INLINE_VISIBILITY reverse_iterator rend(_ClassName& __sequence) noexcept(    \
    noexcept(__sequence.rend()))                                                                                  \
  {                                                                                                               \
    return __sequence.rend();                                                                                     \
  }                                                                                                               \
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_INLINE_VISIBILITY _ConstIter rend(const _ClassName& __sequence) noexcept(    \
    noexcept(__sequence.rend()))                                                                                  \
  {                                                                                                               \
    return __sequence.rend();                                                                                     \
  }                                                                                                               \
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_INLINE_VISIBILITY _ConstIter crbegin(const _ClassName& __sequence) noexcept( \
    noexcept(__sequence.rbegin()))                                                                                \
  {                                                                                                               \
    return __sequence.rbegin();                                                                                   \
  }                                                                                                               \
  _CCCL_NODISCARD_FRIEND _LIBCUDACXX_INLINE_VISIBILITY _ConstIter crend(const _ClassName& __sequence) noexcept(   \
    noexcept(__sequence.rend()))                                                                                  \
  {                                                                                                               \
    return __sequence.rend();                                                                                     \
  }

#endif // __CCCL_SEQUENCE_ACCESS_H
