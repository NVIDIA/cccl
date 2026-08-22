//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ATOMIC_FUNCTIONS_CUDA_PTX_BACKEND_H
#define _CUDA_STD___ATOMIC_FUNCTIONS_CUDA_PTX_BACKEND_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/functions/backend.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_scalar.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

struct __cuda_atomic_ptx_backend
{
  template <class _Type>
  static constexpr bool __use_direct_bitwise = sizeof(_Type) < 16;

  template <class _Type>
  static constexpr bool __use_direct_arithmetic = is_scalar_v<_Type> && (sizeof(_Type) < 16);

  template <class _Type>
  static constexpr bool __use_direct_minmax = is_integral_v<_Type> && (sizeof(_Type) < 16);

  template <class _Type>
  static constexpr bool __use_fallback_bitwise = sizeof(_Type) == 16;

  template <class _Type>
  static constexpr bool __use_fallback_arithmetic = is_scalar_v<_Type> && (sizeof(_Type) == 16);

  template <class _Type>
  static constexpr bool __use_fallback_minmax = !is_integral_v<_Type> || (is_scalar_v<_Type> && sizeof(_Type) == 16);

  static constexpr bool __needs_constant_order             = true;
  static constexpr bool __requires_local_memory_workaround = true;
  static constexpr size_t __smallest_cas                   = 32;
  static constexpr size_t __widest_cas                     = 128;
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ATOMIC_FUNCTIONS_CUDA_PTX_BACKEND_H
