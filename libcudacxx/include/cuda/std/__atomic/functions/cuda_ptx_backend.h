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
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstddef>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

struct __cuda_atomic_ptx_backend
{
  template <class _Order>
  _CCCL_HOST_DEVICE_API static constexpr auto __transform_order(_Order)
  {
    if constexpr (is_same_v<_Order, __cuda_atomic_order_relaxed>)
    {
      return __cuda_atomic_ptx_order_relaxed{false};
    }
    else if constexpr (is_same_v<_Order, __cuda_atomic_order_release>)
    {
      return __cuda_atomic_ptx_order_release{false};
    }
    else if constexpr (is_same_v<_Order, __cuda_atomic_order_acquire>)
    {
      return __cuda_atomic_ptx_order_acquire{false};
    }
    else
    {
      static_assert(is_same_v<_Order, __cuda_atomic_order_acq_rel>, "invalid PTX atomic order");
      return __cuda_atomic_ptx_order_acq_rel{false};
    }
  }

  template <class _Order>
  _CCCL_HOST_DEVICE_API static constexpr _Order __collapse_cas_order(_Order __order)
  {
    return __order;
  }

  template <class _Success, class _Failure>
  _CCCL_HOST_DEVICE_API static constexpr auto __collapse_cas_order(__cuda_atomic_cas_order<_Success, _Failure>)
  {
    if constexpr (is_same_v<_Success, __cuda_atomic_order_seq_cst> || is_same_v<_Failure, __cuda_atomic_order_seq_cst>)
    {
      return __cuda_atomic_order_seq_cst{};
    }
    else if constexpr (is_same_v<_Success, __cuda_atomic_order_acq_rel>
                       || (is_same_v<_Success, __cuda_atomic_order_release>
                           && is_same_v<_Failure, __cuda_atomic_order_acquire>) )
    {
      return __cuda_atomic_order_acq_rel{};
    }
    else if constexpr (is_same_v<_Success, __cuda_atomic_order_release>)
    {
      return __cuda_atomic_order_release{};
    }
    else if constexpr (is_same_v<_Success, __cuda_atomic_order_acquire>
                       || is_same_v<_Failure, __cuda_atomic_order_acquire>)
    {
      return __cuda_atomic_order_acquire{};
    }
    else
    {
      return __cuda_atomic_order_relaxed{};
    }
  }

  template <class _Operation, class _Fn, class _Order, class _Sco, class... _Args>
  _CCCL_DEVICE_API static auto
  __with_transformed_order(_Operation, _Fn& __fn, _Order __order, _Sco __scope, _Args... __args)
    -> decltype(__fn(__order, __args..., __scope));

  static constexpr bool __needs_constant_order             = true;
  static constexpr bool __requires_local_memory_workaround = true;
  static constexpr size_t __smallest_cas                   = 32;
  static constexpr size_t __widest_cas                     = 128;
};

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ATOMIC_FUNCTIONS_CUDA_PTX_BACKEND_H
