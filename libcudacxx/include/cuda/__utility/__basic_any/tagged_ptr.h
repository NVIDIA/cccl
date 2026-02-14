//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_BASIC_ANY_TAGGED_PTR_H
#define _CUDA___UTILITY_BASIC_ANY_TAGGED_PTR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__utility/__basic_any/basic_any_fwd.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <class _Ptr>
struct __tagged_ptr;

template <class _Tp>
struct __tagged_ptr<_Tp*>
{
  _CCCL_API void __set(_Tp* __pv, bool __flag) noexcept
  {
    __ptr_ = reinterpret_cast<uintptr_t>(__pv) | uintptr_t(__flag);
  }

  [[nodiscard]] _CCCL_API auto __get() const noexcept -> _Tp*
  {
    return reinterpret_cast<_Tp*>(__ptr_ & ~uintptr_t(1));
  }

  [[nodiscard]] _CCCL_API auto __flag() const noexcept -> bool
  {
    return static_cast<bool>(__ptr_ & uintptr_t(1));
  }

  uintptr_t __ptr_ = 0;
};

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___UTILITY_BASIC_ANY_TAGGED_PTR_H
