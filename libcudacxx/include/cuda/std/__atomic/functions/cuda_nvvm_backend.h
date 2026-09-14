//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ATOMIC_FUNCTIONS_CUDA_NVVM_BACKEND_H
#define _CUDA_STD___ATOMIC_FUNCTIONS_CUDA_NVVM_BACKEND_H

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

struct __cuda_atomic_nvvm_backend
{
private:
  template <class _Type>
  static constexpr bool __has_builtin_subword_rmw = sizeof(_Type) >= 4 || _CCCL_PTX_ARCH() >= 1000;

public:
  template <class _Order>
  _CCCL_HOST_DEVICE_API static constexpr _Order __collapse_cas_order(_Order __order)
  {
    return __order;
  }

  template <class _Operation, class _Fn, class _Order, class _Sco, class... _Args>
  _CCCL_DEVICE_API static auto
  __with_transformed_order(_Operation, _Fn& __fn, _Order __order, _Sco __scope, _Args... __args)
    -> decltype(__fn(__order, __args..., __scope))
  {
    return __fn(__order, __args..., __scope);
  }

  template <class _Type>
  static constexpr bool __use_direct_bitwise = sizeof(_Type) < 16 && __has_builtin_subword_rmw<_Type>;

  template <class _Type>
  static constexpr bool __use_direct_arithmetic =
    is_scalar_v<_Type> && sizeof(_Type) < 16 && __has_builtin_subword_rmw<_Type>;

  template <class _Type>
  static constexpr bool __use_direct_minmax =
    is_integral_v<_Type> && sizeof(_Type) < 16 && __has_builtin_subword_rmw<_Type>;

  template <class _Type>
  static constexpr bool __use_fallback_bitwise = sizeof(_Type) == 16 || !__has_builtin_subword_rmw<_Type>;

  template <class _Type>
  static constexpr bool __use_fallback_arithmetic =
    is_scalar_v<_Type> && (sizeof(_Type) == 16 || !__has_builtin_subword_rmw<_Type>);

  template <class _Type>
  static constexpr bool __use_fallback_minmax =
    !is_integral_v<_Type> || (is_scalar_v<_Type> && (sizeof(_Type) == 16 || !__has_builtin_subword_rmw<_Type>) );

  static constexpr bool __needs_constant_order             = true;
  static constexpr bool __requires_local_memory_workaround = true;
  static constexpr size_t __smallest_cas                   = 32;
  static constexpr size_t __widest_cas                     = 128;
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ATOMIC_FUNCTIONS_CUDA_NVVM_BACKEND_H
