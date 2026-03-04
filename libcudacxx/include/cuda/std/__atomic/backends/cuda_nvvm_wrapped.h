//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ATOMIC_BACKENDS_CUDA_NVVM_WRAPPED_H
#define _CUDA_STD___ATOMIC_BACKENDS_CUDA_NVVM_WRAPPED_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/order.h>
#include <cuda/std/__atomic/scopes.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_CUDA_COMPILATION()

#  define __ATOMIC_NVVM_WRAP(...)    (__VA_ARGS__)
#  define __ATOMIC_NVVM_UNWRAP1(...) __VA_ARGS__
#  define __ATOMIC_NVVM_UNWRAP(...)  __ATOMIC_NVVM_UNWRAP1 __VA_ARGS__

#  define __ATOMIC_SWITCH(val, ...) \
    switch (val)                    \
    {                               \
      __VA_ARGS__                   \
    }

#  define __ATOMIC_CASE(test, fn, ...) \
    case test:                         \
      _CCCL_PP_OBSTRUCT(fn)(test, ##__VA_ARGS__) break;

#  define __ATOMIC_SCOPE_CASES_SM90(...)                              \
    /* THREAD */ __ATOMIC_CASE(__NV_THREAD_SCOPE_THREAD, __VA_ARGS__) \
    /* BLOCK */ __ATOMIC_CASE(__NV_THREAD_SCOPE_BLOCK, __VA_ARGS__)   \
    /* DEVICE */ __ATOMIC_CASE(__NV_THREAD_SCOPE_DEVICE, __VA_ARGS__) \
    /* SYSTEM */ __ATOMIC_CASE(__NV_THREAD_SCOPE_SYSTEM, __VA_ARGS__) \
    /* CLUSTER */ __ATOMIC_CASE(__NV_THREAD_SCOPE_CLUSTER, __VA_ARGS__)

#  define __ATOMIC_SCOPE_CASES(...)                                   \
    /* THREAD */ __ATOMIC_CASE(__NV_THREAD_SCOPE_THREAD, __VA_ARGS__) \
    /* BLOCK */ __ATOMIC_CASE(__NV_THREAD_SCOPE_BLOCK, __VA_ARGS__)   \
    /* DEVICE */ __ATOMIC_CASE(__NV_THREAD_SCOPE_DEVICE, __VA_ARGS__) \
    /* SYSTEM */ __ATOMIC_CASE(__NV_THREAD_SCOPE_SYSTEM, __VA_ARGS__)

#  define __ATOMIC_ALL_ORDER_CASES(...)                           \
    /* RELAXED */ __ATOMIC_CASE(__NV_ATOMIC_RELAXED, __VA_ARGS__) \
    /* CONSUME */ __ATOMIC_CASE(__NV_ATOMIC_CONSUME, __VA_ARGS__) \
    /* ACQUIRE */ __ATOMIC_CASE(__NV_ATOMIC_ACQUIRE, __VA_ARGS__) \
    /* RELEASE */ __ATOMIC_CASE(__NV_ATOMIC_RELEASE, __VA_ARGS__) \
    /* ACQ_REL */ __ATOMIC_CASE(__NV_ATOMIC_ACQ_REL, __VA_ARGS__) \
    /* SEQ_CST */ __ATOMIC_CASE(__NV_ATOMIC_SEQ_CST, __VA_ARGS__)

#  define __ATOMIC_READ_CASES(...)                                \
    /* RELAXED */ __ATOMIC_CASE(__NV_ATOMIC_RELAXED, __VA_ARGS__) \
    /* CONSUME */ __ATOMIC_CASE(__NV_ATOMIC_CONSUME, __VA_ARGS__) \
    /* ACQUIRE */ __ATOMIC_CASE(__NV_ATOMIC_ACQUIRE, __VA_ARGS__) \
    /* SEQ_CST */ __ATOMIC_CASE(__NV_ATOMIC_SEQ_CST, __VA_ARGS__)

#  define __ATOMIC_WRITE_CASES(...)                               \
    /* RELAXED */ __ATOMIC_CASE(__NV_ATOMIC_RELAXED, __VA_ARGS__) \
    /* RELEASE */ __ATOMIC_CASE(__NV_ATOMIC_RELEASE, __VA_ARGS__) \
    /* SEQ_CST */ __ATOMIC_CASE(__NV_ATOMIC_SEQ_CST, __VA_ARGS__)

#  define __ATOMIC_FENCE_CASES(...)    __ATOMIC_ALL_ORDER_CASES(__VA_ARGS__)
#  define __ATOMIC_EXCHANGE_CASES(...) __ATOMIC_ALL_ORDER_CASES(__VA_ARGS__)
#  define __ATOMIC_FETCH_OP_CASES(...) __ATOMIC_ALL_ORDER_CASES(__VA_ARGS__)

#  define __ATOMIC_COMPARE_SUCCESS_CASES(...) __ATOMIC_ALL_ORDER_CASES(__VA_ARGS__)
#  define __ATOMIC_COMPARE_FAILURE_CASES(...) __ATOMIC_READ_CASES(__VA_ARGS__)

#  define __ATOMIC_SCOPES_SWITCH(scope, scopes, ...) __ATOMIC_SWITCH(scope, scopes(__VA_ARGS__))
#  define __ATOMIC_ORDER_SWITCH(order, orders, ...)  __ATOMIC_SWITCH(order, orders(__VA_ARGS__))

#  define __ATOMIC_NVVM_BUILTIN2(_scope, intrinsic, ...) intrinsic(__ATOMIC_NVVM_UNWRAP(__VA_ARGS__), _scope);
#  define __ATOMIC_NVVM_BUILTIN1(_order, intrinsic, scope, scopes, ...) \
    __ATOMIC_SCOPES_SWITCH(                                             \
      scope, scopes, __ATOMIC_NVVM_BUILTIN2, intrinsic, __ATOMIC_NVVM_WRAP(__ATOMIC_NVVM_UNWRAP(__VA_ARGS__), _order))
#  define __ATOMIC_NVVM_BUILTIN0(_order, intrinsic, order, orders, scope, scopes, ...) \
    __ATOMIC_ORDER_SWITCH(                                                             \
      order,                                                                           \
      orders,                                                                          \
      __ATOMIC_NVVM_BUILTIN1,                                                          \
      intrinsic,                                                                       \
      scope,                                                                           \
      scopes,                                                                          \
      __ATOMIC_NVVM_WRAP(__ATOMIC_NVVM_UNWRAP(__VA_ARGS__), _order))

// An attempted explanation:
// We pass down macro function names and arguments through functions that create switch statements, the cases expand
// them by eventually invoking the passed in `__ATOMIC_NVVM_BUILTIN#` with the now concrete case value selected in the
// switch - This then calls another switch builder, uses another macro function, and expands again. Arguments to the
// function are packed inside of `()` by __ATOMIC_NVVM_WRAP/UNWRAP in order to prevent any accidental escape.
#  define __ATOMIC_NVVM_BUILTIN(intrinsic, order, orders, scope, ...) \
    NV_IF_ELSE_TARGET(                                                \
      NV_PROVIDES_SM_90,                                              \
      ({__ATOMIC_ORDER_SWITCH(                                        \
        order,                                                        \
        orders,                                                       \
        __ATOMIC_NVVM_BUILTIN1,                                       \
        intrinsic,                                                    \
        scope,                                                        \
        __ATOMIC_SCOPE_CASES_SM90,                                    \
        __ATOMIC_NVVM_WRAP(__VA_ARGS__))}),                           \
      ({__ATOMIC_ORDER_SWITCH(                                        \
        order,                                                        \
        orders,                                                       \
        __ATOMIC_NVVM_BUILTIN1,                                       \
        intrinsic,                                                    \
        scope,                                                        \
        __ATOMIC_SCOPE_CASES,                                         \
        __ATOMIC_NVVM_WRAP(__VA_ARGS__))}))

// __ATOMIC_NVVM_BUILTIN_SF selects three times for compare_exchange
#  define __ATOMIC_NVVM_BUILTIN_SF(intrinsic, success, sorders, failure, forders, scope, ...) \
    NV_IF_ELSE_TARGET(                                                                        \
      NV_PROVIDES_SM_90,                                                                      \
      ({__ATOMIC_ORDER_SWITCH(                                                                \
        success,                                                                              \
        sorders,                                                                              \
        __ATOMIC_NVVM_BUILTIN0,                                                               \
        intrinsic,                                                                            \
        failure,                                                                              \
        forders,                                                                              \
        scope,                                                                                \
        __ATOMIC_SCOPE_CASES_SM90,                                                            \
        __ATOMIC_NVVM_WRAP(__VA_ARGS__))}),                                                   \
      ({__ATOMIC_ORDER_SWITCH(                                                                \
        success,                                                                              \
        sorders,                                                                              \
        __ATOMIC_NVVM_BUILTIN0,                                                               \
        intrinsic,                                                                            \
        failure,                                                                              \
        forders,                                                                              \
        scope,                                                                                \
        __ATOMIC_SCOPE_CASES,                                                                 \
        __ATOMIC_NVVM_WRAP(__VA_ARGS__))}))

template <class _Type>
_CCCL_ARTIFICIAL static inline _CCCL_DEVICE void
__atomic_load_nvvm_dispatch(const _Type* __ptr, _Type* __dst, int __memorder, int __sco)
{
  _CCCL_PP_EXPAND(__ATOMIC_NVVM_BUILTIN(__nv_atomic_load, __memorder, __ATOMIC_READ_CASES, __sco, __ptr, __dst));
}

template <class _Type>
_CCCL_ARTIFICIAL static inline _CCCL_DEVICE void
__atomic_store_nvvm_dispatch(_Type* __ptr, _Type* __val, int __memorder, int __sco)
{
  _CCCL_PP_EXPAND(__ATOMIC_NVVM_BUILTIN(__nv_atomic_store, __memorder, __ATOMIC_WRITE_CASES, __sco, __ptr, __val));
}

template <class _Type>
_CCCL_ARTIFICIAL static inline _CCCL_DEVICE bool __atomic_compare_exchange_nvvm_dispatch(
  _Type* __ptr, _Type* __exp, _Type* __des, bool __weak, int __success_memorder, int __failure_memorder, int __sco)
{
  _CCCL_PP_EXPAND(__ATOMIC_NVVM_BUILTIN_SF(
    return __nv_atomic_compare_exchange,
           __success_memorder,
           __ATOMIC_COMPARE_SUCCESS_CASES,
           __failure_memorder,
           __ATOMIC_COMPARE_FAILURE_CASES,
           __sco,
           __ptr,
           __exp,
           __des,
           __weak));
  _CCCL_UNREACHABLE();
  return {};
}

template <class _Type>
_CCCL_ARTIFICIAL static inline _CCCL_DEVICE void
__atomic_exchange_nvvm_dispatch(_Type* __atom, _Type* __val, _Type* __ret, int __memorder, int __sco)
{
  _CCCL_PP_EXPAND(
    __ATOMIC_NVVM_BUILTIN(__nv_atomic_exchange, __memorder, __ATOMIC_EXCHANGE_CASES, __sco, __atom, __val, __ret));
}

template <class _Type>
_CCCL_ARTIFICIAL static inline _CCCL_DEVICE _Type
__atomic_fetch_max_nvvm_dispatch(_Type* __ptr, _Type __op, int __memorder, int __sco)
{
  _CCCL_PP_EXPAND(
    __ATOMIC_NVVM_BUILTIN(return __nv_atomic_fetch_max, __memorder, __ATOMIC_EXCHANGE_CASES, __sco, __ptr, __op));
  _CCCL_UNREACHABLE();
  return {};
}

template <class _Type>
_CCCL_ARTIFICIAL static inline _CCCL_DEVICE _Type
__atomic_fetch_min_nvvm_dispatch(_Type* __ptr, _Type __op, int __memorder, int __sco)
{
  _CCCL_PP_EXPAND(
    __ATOMIC_NVVM_BUILTIN(return __nv_atomic_fetch_min, __memorder, __ATOMIC_EXCHANGE_CASES, __sco, __ptr, __op));
  _CCCL_UNREACHABLE();
  return {};
}

template <class _Type>
_CCCL_ARTIFICIAL static inline _CCCL_DEVICE _Type
__atomic_fetch_and_nvvm_dispatch(_Type* __ptr, _Type __op, int __memorder, int __sco)
{
  _CCCL_PP_EXPAND(
    __ATOMIC_NVVM_BUILTIN(return __nv_atomic_fetch_and, __memorder, __ATOMIC_EXCHANGE_CASES, __sco, __ptr, __op));
  _CCCL_UNREACHABLE();
  return {};
}

template <class _Type>
_CCCL_ARTIFICIAL static inline _CCCL_DEVICE _Type
__atomic_fetch_or_nvvm_dispatch(_Type* __ptr, _Type __op, int __memorder, int __sco)
{
  _CCCL_PP_EXPAND(
    __ATOMIC_NVVM_BUILTIN(return __nv_atomic_fetch_or, __memorder, __ATOMIC_EXCHANGE_CASES, __sco, __ptr, __op));
  _CCCL_UNREACHABLE();
  return {};
}

template <class _Type>
_CCCL_ARTIFICIAL static inline _CCCL_DEVICE _Type
__atomic_fetch_xor_nvvm_dispatch(_Type* __ptr, _Type __op, int __memorder, int __sco)
{
  _CCCL_PP_EXPAND(
    __ATOMIC_NVVM_BUILTIN(return __nv_atomic_fetch_xor, __memorder, __ATOMIC_EXCHANGE_CASES, __sco, __ptr, __op));
  _CCCL_UNREACHABLE();
  return {};
}

template <class _Type>
_CCCL_ARTIFICIAL static inline _CCCL_DEVICE _Type
__atomic_fetch_add_nvvm_dispatch(_Type* __ptr, _Type __op, int __memorder, int __sco)
{
  _CCCL_PP_EXPAND(
    __ATOMIC_NVVM_BUILTIN(return __nv_atomic_fetch_add, __memorder, __ATOMIC_EXCHANGE_CASES, __sco, __ptr, __op));
  _CCCL_UNREACHABLE();
  return {};
}

#endif // _CCCL_CUDA_COMPILATION

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ATOMIC_BACKENDS_CUDA_NVVM_WRAPPED_H
