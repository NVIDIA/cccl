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

#include <cuda/std/__atomic/functions/common.h>
#include <cuda/std/__atomic/functions/cuda_ptx_backend.h>
#include <cuda/std/__atomic/functions/cuda_ptx_generated.h>
#include <cuda/std/__atomic/functions/generic.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/make_unsigned.h>

#include <nv/target>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Operation, class _Fn, class _Order, class _Sco, class... _Args>
_CCCL_DEVICE_API auto __cuda_atomic_ptx_backend::__with_transformed_order(
  _Operation, _Fn& __fn, _Order __order, _Sco __scope, _Args... __args) -> decltype(__fn(__order, __args..., __scope))
{
  constexpr bool __is_load  = is_same_v<_Operation, __cuda_atomic_operation_load>;
  constexpr bool __is_store = is_same_v<_Operation, __cuda_atomic_operation_store>;
  constexpr bool __is_rmw   = is_same_v<_Operation, __cuda_atomic_operation_rmw>;
  static_assert(__is_load || __is_store || __is_rmw, "invalid atomic operation class");

  [[maybe_unused]] constexpr bool __is_seq_cst = is_same_v<_Order, __cuda_atomic_order_seq_cst>;

  NV_DISPATCH_TARGET(
    NV_PROVIDES_SM_70,
    ({
      if constexpr (__is_seq_cst)
      {
        if constexpr (__is_store)
        {
          return __fn(__cuda_atomic_ptx_order_relaxed{true}, __args..., __scope);
        }
        else
        {
          return __fn(__cuda_atomic_ptx_order_acquire{true}, __args..., __scope);
        }
      }
      else
      {
        return __fn(__transform_order(__order), __args..., __scope);
      }
    }),
    NV_IS_DEVICE,
    ({
      constexpr bool __is_release = is_same_v<_Order, __cuda_atomic_order_release>;
      constexpr bool __is_acq_rel = is_same_v<_Order, __cuda_atomic_order_acq_rel>;
      constexpr bool __is_acquire = is_same_v<_Order, __cuda_atomic_order_acquire>;
      constexpr bool __membar_before =
        __is_seq_cst || (__is_store && __is_release) || (__is_rmw && (__is_release || __is_acq_rel));
      constexpr bool __membar_after = (__is_load || __is_rmw) && (__is_acquire || __is_acq_rel || __is_seq_cst);

      if constexpr (__membar_before)
      {
        ::cuda::std::__cuda_atomic_membar(__scope);
      }
      if constexpr (__membar_after)
      {
        if constexpr (is_void_v<decltype(__fn(__cuda_atomic_order_volatile{}, __args..., __scope))>)
        {
          __fn(__cuda_atomic_order_volatile{}, __args..., __scope);
          ::cuda::std::__cuda_atomic_membar(__scope);
          return;
        }
        else
        {
          auto __result = __fn(__cuda_atomic_order_volatile{}, __args..., __scope);
          ::cuda::std::__cuda_atomic_membar(__scope);
          return __result;
        }
      }
      else
      {
        return __fn(__cuda_atomic_order_volatile{}, __args..., __scope);
      }
    }))
}

template <class _Type, enable_if_t<is_integral_v<_Type>, int> = 0>
[[nodiscard]] _CCCL_DEVICE_API _Type __cuda_atomic_ptx_negate(_Type __value)
{
  using __unsigned_type = make_unsigned_t<_Type>;
  const auto __bits     = ::cuda::std::bit_cast<__unsigned_type>(__value);
  const auto __negated  = static_cast<__unsigned_type>(__unsigned_type{} - __bits);
  return ::cuda::std::bit_cast<_Type>(__negated);
}

template <class _Type, enable_if_t<!is_integral_v<_Type>, int> = 0>
[[nodiscard]] _CCCL_DEVICE_API _Type __cuda_atomic_ptx_negate(_Type __value)
{
  return -__value;
}

template <class _Type,
          class _Order,
          class _Operand,
          class _Sco,
          enable_if_t<(_Operand::__size >= 32) && (_Operand::__size <= 64), int> = 0>
_CCCL_DEVICE_API void __cuda_atomic_fetch_sub(
  __cuda_atomic_ptx_backend __backend,
  _Type* __ptr,
  __unv<_Type>& __dst,
  __unv<_Type> __op,
  _Order __order,
  _Operand,
  _Sco __scope)
{
  ::cuda::std::__cuda_atomic_fetch_add(
    __backend, __ptr, __dst, ::cuda::std::__cuda_atomic_ptx_negate(__op), __order, _Operand{}, __scope);
}

#if _CCCL_CUDA_COMPILATION()
_CCCL_DEVICE_API inline void __cuda_atomic_signal_fence(__cuda_atomic_ptx_backend, memory_order)
{
  asm volatile("" ::: "memory");
}
#endif // _CCCL_CUDA_COMPILATION()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ATOMIC_FUNCTIONS_CUDA_PTX_H
