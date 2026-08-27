//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ATOMIC_FUNCTIONS_CUDA_PTX_H
#define _CUDA_STD___ATOMIC_FUNCTIONS_CUDA_PTX_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/functions/cuda_ptx_backend.h>
#include <cuda/std/__atomic/functions/cuda_ptx_generated.h>
#include <cuda/std/__atomic/functions/generic.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/make_unsigned.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Type, enable_if_t<is_integral_v<_Type>, bool> = false>
[[nodiscard]] _CCCL_DEVICE_API _Type __cuda_atomic_ptx_negate(_Type __value)
{
  using __unsigned_type = make_unsigned_t<_Type>;
  const auto __bits     = ::cuda::std::bit_cast<__unsigned_type>(__value);
  const auto __negated  = static_cast<__unsigned_type>(__unsigned_type{} - __bits);
  return ::cuda::std::bit_cast<_Type>(__negated);
}

template <class _Type, enable_if_t<!is_integral_v<_Type>, bool> = false>
[[nodiscard]] _CCCL_DEVICE_API _Type __cuda_atomic_ptx_negate(_Type __value)
{
  return -__value;
}

template <class _Type, class _Order, class _Operand, class _Sco>
_CCCL_DEVICE_API void __cuda_atomic_fetch_sub(
  __cuda_atomic_ptx_backend __backend, _Type* __ptr, _Type& __dst, _Type __op, _Order __order, _Operand, _Sco __scope)
{
  __cuda_atomic_fetch_add(__backend, __ptr, __dst, __cuda_atomic_ptx_negate(__op), __order, _Operand{}, __scope);
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ATOMIC_FUNCTIONS_CUDA_PTX_H
