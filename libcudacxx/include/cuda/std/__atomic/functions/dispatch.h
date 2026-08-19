//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ATOMIC_FUNCTIONS_DISPATCH_H
#define _CUDA_STD___ATOMIC_FUNCTIONS_DISPATCH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/functions/cuda_ptx.h>
#include <cuda/std/__atomic/functions/host.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Backend, class _Type, class _Fn, class _Operand, class _Sco>
struct __cuda_atomic_bind_fetch_fallback
{
  _Backend __backend;
  _Type* __ptr;
  _Type* __dst;
  _Type* __op;
  _Fn __fn;

  template <class _Order>
  _CCCL_HOST_DEVICE_API void operator()(_Order __order)
  {
    __fn(__backend, __ptr, *__dst, *__op, __order, _Operand{}, _Sco{});
  }
};

template <class _Backend, class _Type, class _Operand, class _Sco>
struct __cuda_atomic_bind_fetch_sub
{
  _Backend __backend;
  _Type* __ptr;
  _Type* __dst;
  _Type* __op;

  template <class _Order>
  _CCCL_HOST_DEVICE_API void operator()(_Order __order)
  {
    __cuda_atomic_fetch_sub(__backend, __ptr, *__dst, *__op, __order, _Operand{}, _Sco{});
  }
};

template <class _Backend, class _Type, class _Fn, class _Sco>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Type __cuda_atomic_fetch_fallback_dispatch(
  _Backend __backend, _Type* __ptr, _Type __op, memory_order __order, _Sco __scope, _Fn __fn)
{
  using __operand = __cuda_atomic_operand_tag<__cuda_atomic_operand::_b, sizeof(_Type) * 8>;
  _Type __dst{};
  __cuda_atomic_bind_fetch_fallback<_Backend, _Type, _Fn, __operand, _Sco> __bound_fetch_fallback{
    __backend, __ptr, &__dst, &__op, __fn};
  __cuda_atomic_fetch_order_dispatch(__backend, __bound_fetch_fallback, __order, __scope);
  return __dst;
}

template <class _Backend,
          class _Type,
          class _Up,
          class _Sco,
          typename _Backend::template __enable_if_direct_arithmetic<_Type> = false>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Type
__cuda_atomic_fetch_sub_dispatch(_Backend __backend, _Type* __ptr, _Up __op, memory_order __order, _Sco __scope)
{
  constexpr auto __skip = __atomic_ptr_skip_t<_Type>::__skip;
  __op                  = __op * __skip;
  using __proxy_type    = __cuda_atomic_deduce_arithmetic_t<_Type>;
  using __proxy_operand = __cuda_atomic_deduce_arithmetic_tag_t<_Type>;
  _Type __dst{};
  auto* __ptr_proxy = reinterpret_cast<__proxy_type*>(__ptr);
  auto* __dst_proxy = reinterpret_cast<__proxy_type*>(&__dst);
  auto* __op_proxy  = reinterpret_cast<__proxy_type*>(&__op);
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    if (__cuda_atomic_fetch_sub_weak_if_local(__ptr_proxy, *__op_proxy, __dst_proxy))
    {
      return __dst;
    }
  }
  __cuda_atomic_bind_fetch_sub<_Backend, __proxy_type, __proxy_operand, _Sco> __bound_fetch_sub{
    __backend, __ptr_proxy, __dst_proxy, __op_proxy};
  __cuda_atomic_fetch_order_dispatch(__backend, __bound_fetch_sub, __order, __scope);
  return __dst;
}

template <class _Backend,
          class _Type,
          class _Up,
          class _Sco,
          typename _Backend::template __enable_if_direct_arithmetic<_Type> = false>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Type __cuda_atomic_fetch_sub_dispatch(
  _Backend __backend, volatile _Type* __ptr, _Up __op, memory_order __order, _Sco __scope)
{
  return __cuda_atomic_fetch_sub_dispatch(__backend, const_cast<_Type*>(__ptr), __op, __order, __scope);
}

template <class _Backend,
          class _Type,
          class _Up,
          class _Sco,
          typename _Backend::template __enable_if_fallback_arithmetic<_Type> = false>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Type
__cuda_atomic_fetch_add_dispatch(_Backend __backend, _Type* __ptr, _Up __val, memory_order __order, _Sco __scope)
{
  constexpr auto __skip = __atomic_ptr_skip_t<_Type>::__skip;
  const _Type __op      = static_cast<_Type>(__val * __skip);
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    _Type __dst{};
    if (__cuda_atomic_fetch_add_weak_if_local(__ptr, __op, &__dst))
    {
      return __dst;
    }
  }
  return __cuda_atomic_fetch_fallback_dispatch(__backend, __ptr, __op, __order, __scope, __cuda_atomic_fetch_add_op{});
}

template <class _Backend,
          class _Type,
          class _Up,
          class _Sco,
          typename _Backend::template __enable_if_fallback_arithmetic<_Type> = false>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Type __cuda_atomic_fetch_add_dispatch(
  _Backend __backend, volatile _Type* __ptr, _Up __val, memory_order __order, _Sco __scope)
{
  return __cuda_atomic_fetch_add_dispatch(__backend, const_cast<_Type*>(__ptr), __val, __order, __scope);
}

template <class _Backend,
          class _Type,
          class _Up,
          class _Sco,
          typename _Backend::template __enable_if_fallback_arithmetic<_Type> = false>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Type
__cuda_atomic_fetch_sub_dispatch(_Backend __backend, _Type* __ptr, _Up __val, memory_order __order, _Sco __scope)
{
  constexpr auto __skip = __atomic_ptr_skip_t<_Type>::__skip;
  const _Type __op      = static_cast<_Type>(__val * __skip);
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    _Type __dst{};
    if (__cuda_atomic_fetch_sub_weak_if_local(__ptr, __op, &__dst))
    {
      return __dst;
    }
  }
  return __cuda_atomic_fetch_fallback_dispatch(__backend, __ptr, __op, __order, __scope, __cuda_atomic_fetch_sub_op{});
}

template <class _Backend,
          class _Type,
          class _Up,
          class _Sco,
          typename _Backend::template __enable_if_fallback_arithmetic<_Type> = false>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Type __cuda_atomic_fetch_sub_dispatch(
  _Backend __backend, volatile _Type* __ptr, _Up __val, memory_order __order, _Sco __scope)
{
  return __cuda_atomic_fetch_sub_dispatch(__backend, const_cast<_Type*>(__ptr), __val, __order, __scope);
}

template <class _Backend,
          class _Type,
          class _Up,
          class _Sco,
          typename _Backend::template __enable_if_fallback_bitwise<_Type> = false>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Type
__cuda_atomic_fetch_and_dispatch(_Backend __backend, _Type* __ptr, _Up __val, memory_order __order, _Sco __scope)
{
  const _Type __op = static_cast<_Type>(__val);
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    _Type __dst{};
    if (__cuda_atomic_fetch_and_weak_if_local(__ptr, __op, &__dst))
    {
      return __dst;
    }
  }
  return __cuda_atomic_fetch_fallback_dispatch(__backend, __ptr, __op, __order, __scope, __cuda_atomic_fetch_and_op{});
}

template <class _Backend,
          class _Type,
          class _Up,
          class _Sco,
          typename _Backend::template __enable_if_fallback_bitwise<_Type> = false>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Type __cuda_atomic_fetch_and_dispatch(
  _Backend __backend, volatile _Type* __ptr, _Up __val, memory_order __order, _Sco __scope)
{
  return __cuda_atomic_fetch_and_dispatch(__backend, const_cast<_Type*>(__ptr), __val, __order, __scope);
}

template <class _Backend,
          class _Type,
          class _Up,
          class _Sco,
          typename _Backend::template __enable_if_fallback_bitwise<_Type> = false>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Type
__cuda_atomic_fetch_or_dispatch(_Backend __backend, _Type* __ptr, _Up __val, memory_order __order, _Sco __scope)
{
  const _Type __op = static_cast<_Type>(__val);
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    _Type __dst{};
    if (__cuda_atomic_fetch_or_weak_if_local(__ptr, __op, &__dst))
    {
      return __dst;
    }
  }
  return __cuda_atomic_fetch_fallback_dispatch(__backend, __ptr, __op, __order, __scope, __cuda_atomic_fetch_or_op{});
}

template <class _Backend,
          class _Type,
          class _Up,
          class _Sco,
          typename _Backend::template __enable_if_fallback_bitwise<_Type> = false>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Type __cuda_atomic_fetch_or_dispatch(
  _Backend __backend, volatile _Type* __ptr, _Up __val, memory_order __order, _Sco __scope)
{
  return __cuda_atomic_fetch_or_dispatch(__backend, const_cast<_Type*>(__ptr), __val, __order, __scope);
}

template <class _Backend,
          class _Type,
          class _Up,
          class _Sco,
          typename _Backend::template __enable_if_fallback_bitwise<_Type> = false>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Type
__cuda_atomic_fetch_xor_dispatch(_Backend __backend, _Type* __ptr, _Up __val, memory_order __order, _Sco __scope)
{
  const _Type __op = static_cast<_Type>(__val);
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    _Type __dst{};
    if (__cuda_atomic_fetch_xor_weak_if_local(__ptr, __op, &__dst))
    {
      return __dst;
    }
  }
  return __cuda_atomic_fetch_fallback_dispatch(__backend, __ptr, __op, __order, __scope, __cuda_atomic_fetch_xor_op{});
}

template <class _Backend,
          class _Type,
          class _Up,
          class _Sco,
          typename _Backend::template __enable_if_fallback_bitwise<_Type> = false>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Type __cuda_atomic_fetch_xor_dispatch(
  _Backend __backend, volatile _Type* __ptr, _Up __val, memory_order __order, _Sco __scope)
{
  return __cuda_atomic_fetch_xor_dispatch(__backend, const_cast<_Type*>(__ptr), __val, __order, __scope);
}

template <class _Backend,
          class _Type,
          class _Up,
          class _Sco,
          typename _Backend::template __enable_if_fallback_minmax<_Type> = false>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Type
__cuda_atomic_fetch_min_dispatch(_Backend __backend, _Type* __ptr, _Up __val, memory_order __order, _Sco __scope)
{
  const _Type __op = static_cast<_Type>(__val);
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    _Type __dst{};
    if (__cuda_atomic_fetch_min_weak_if_local(__ptr, __op, &__dst))
    {
      return __dst;
    }
  }
  return __cuda_atomic_fetch_fallback_dispatch(__backend, __ptr, __op, __order, __scope, __cuda_atomic_fetch_min_op{});
}

template <class _Backend,
          class _Type,
          class _Up,
          class _Sco,
          typename _Backend::template __enable_if_fallback_minmax<_Type> = false>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Type __cuda_atomic_fetch_min_dispatch(
  _Backend __backend, volatile _Type* __ptr, _Up __val, memory_order __order, _Sco __scope)
{
  return __cuda_atomic_fetch_min_dispatch(__backend, const_cast<_Type*>(__ptr), __val, __order, __scope);
}

template <class _Backend,
          class _Type,
          class _Up,
          class _Sco,
          typename _Backend::template __enable_if_fallback_minmax<_Type> = false>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Type
__cuda_atomic_fetch_max_dispatch(_Backend __backend, _Type* __ptr, _Up __val, memory_order __order, _Sco __scope)
{
  const _Type __op = static_cast<_Type>(__val);
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    _Type __dst{};
    if (__cuda_atomic_fetch_max_weak_if_local(__ptr, __op, &__dst))
    {
      return __dst;
    }
  }
  return __cuda_atomic_fetch_fallback_dispatch(__backend, __ptr, __op, __order, __scope, __cuda_atomic_fetch_max_op{});
}

template <class _Backend,
          class _Type,
          class _Up,
          class _Sco,
          typename _Backend::template __enable_if_fallback_minmax<_Type> = false>
[[nodiscard]] _CCCL_HOST_DEVICE_API _Type __cuda_atomic_fetch_max_dispatch(
  _Backend __backend, volatile _Type* __ptr, _Up __val, memory_order __order, _Sco __scope)
{
  return __cuda_atomic_fetch_max_dispatch(__backend, const_cast<_Type*>(__ptr), __val, __order, __scope);
}

#if _CCCL_CUDA_COMPILATION()
_CCCL_DEVICE static inline void __cuda_atomic_signal_fence(__cuda_atomic_ptx_backend, memory_order)
{
  asm volatile("" ::: "memory");
}
#endif // _CCCL_CUDA_COMPILATION()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ATOMIC_FUNCTIONS_DISPATCH_H
