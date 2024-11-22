// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___MEMORY_ASSUME_ALIGNED_H
#define _LIBCUDACXX___MEMORY_ASSUME_ALIGNED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/__type_traits/is_constant_evaluated.h>
#include <cuda/std/cstddef> // size_t
#include <cuda/std/cstdint> // uintptr_t

_LIBCUDACXX_BEGIN_NAMESPACE_STD

template <size_t _Align, class _Tp>
_CCCL_NODISCARD _LIBCUDACXX_HIDE_FROM_ABI _CCCL_CONSTEXPR_CXX14 _Tp* assume_aligned(_Tp* __ptr) noexcept
{
  static_assert(_CUDA_VSTD::has_single_bit(_Align), "std::assume_aligned requires the alignment to be a power of 2!");
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED) && defined(_CCCL_BUILTIN_ASSUME_ALIGNED)
  if (!_CCCL_BUILTIN_IS_CONSTANT_EVALUATED())
  {
#  if !_CCCL_COMPILER(MSVC) // MSVC checks within the builtin
    _CCCL_ASSERT(reinterpret_cast<uintptr_t>(__ptr) % _Align == 0, "Alignment assumption is violated");
#  endif // !_CCCL_COMPILER(MSVC)
    return static_cast<_Tp*>(_CCCL_BUILTIN_ASSUME_ALIGNED(__ptr, _Align));
  }
  else
#endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED && _CCCL_BUILTIN_ASSUME_ALIGNED
  {
    return __ptr;
  }
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___MEMORY_ASSUME_ALIGNED_H
