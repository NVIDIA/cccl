//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ATOMIC_FUNCTIONS_CUDA_NVVM_H
#define _CUDA_STD___ATOMIC_FUNCTIONS_CUDA_NVVM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/functions/cuda_nvvm_backend.h>
#include <cuda/std/__atomic/functions/generic.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_CTK_AT_LEAST(13, 5) && _CCCL_HAS_NV_ATOMIC_BUILTINS()

extern "C" _CCCL_DEVICE void __cuda_atomic_nvvm_cas_128b_unsupported_before_SM_90();
extern "C" _CCCL_DEVICE void __cuda_atomic_nvvm_exchange_128b_unsupported_before_SM_90();

template <class _Order>
struct __cuda_atomic_nvvm_order;

template <>
struct __cuda_atomic_nvvm_order<__cuda_atomic_order_relaxed>
{
  static constexpr int __value = __NV_ATOMIC_RELAXED;
};

template <>
struct __cuda_atomic_nvvm_order<__cuda_atomic_order_release>
{
  static constexpr int __value = __NV_ATOMIC_RELEASE;
};

template <>
struct __cuda_atomic_nvvm_order<__cuda_atomic_order_acquire>
{
  static constexpr int __value = __NV_ATOMIC_ACQUIRE;
};

template <>
struct __cuda_atomic_nvvm_order<__cuda_atomic_order_acq_rel>
{
  static constexpr int __value = __NV_ATOMIC_ACQ_REL;
};

template <>
struct __cuda_atomic_nvvm_order<__cuda_atomic_order_seq_cst>
{
  static constexpr int __value = __NV_ATOMIC_SEQ_CST;
};

template <class _Scope>
struct __cuda_atomic_nvvm_scope;

template <>
struct __cuda_atomic_nvvm_scope<__thread_scope_block_tag>
{
  static constexpr int __value = __NV_THREAD_SCOPE_BLOCK;
};

template <>
struct __cuda_atomic_nvvm_scope<__thread_scope_cluster_tag>
{
  static constexpr int __value = __NV_THREAD_SCOPE_CLUSTER;
};

template <>
struct __cuda_atomic_nvvm_scope<__thread_scope_device_tag>
{
  static constexpr int __value = __NV_THREAD_SCOPE_DEVICE;
};

template <>
struct __cuda_atomic_nvvm_scope<__thread_scope_system_tag>
{
  static constexpr int __value = __NV_THREAD_SCOPE_SYSTEM;
};

template <class _Type>
[[nodiscard]] _CCCL_DEVICE_API __unv<_Type>* __cuda_atomic_nvvm_ptr(_Type* __ptr)
{
  return const_cast<__unv<_Type>*>(__ptr);
}

template <class _Order>
struct __cuda_atomic_nvvm_failure_order
{
  using type = _Order;
};

template <>
struct __cuda_atomic_nvvm_failure_order<__cuda_atomic_order_release>
{
  using type = __cuda_atomic_order_relaxed;
};

template <>
struct __cuda_atomic_nvvm_failure_order<__cuda_atomic_order_acq_rel>
{
  using type = __cuda_atomic_order_acquire;
};

template <class _Order>
struct __cuda_atomic_nvvm_cas_orders
{
  using __success = _Order;
  using __failure = typename __cuda_atomic_nvvm_failure_order<_Order>::type;
};

template <class _Success, class _Failure>
struct __cuda_atomic_nvvm_cas_orders<__cuda_atomic_cas_order<_Success, _Failure>>
{
  using __success = _Success;
  using __failure = _Failure;
};

template <class _Type, class _Order, class _Operand, class _Scope>
_CCCL_DEVICE_API void __cuda_atomic_load(
  __cuda_atomic_nvvm_backend,
  const _Type* __ptr,
  __unv<_Type>& __dst,
  _Order,
  _Operand,
  _Scope __scope,
  __cuda_atomic_mmio_disable)
{
  ::__nv_atomic_load(__cuda_atomic_nvvm_ptr(__ptr),
                     &__dst,
                     +__cuda_atomic_nvvm_order<_Order>::__value,
                     +__cuda_atomic_nvvm_scope<_Scope>::__value);
}

template <class _Type, class _Order, class _Operand, class _Scope>
_CCCL_DEVICE_API void __cuda_atomic_store(
  __cuda_atomic_nvvm_backend,
  _Type* __ptr,
  __unv<_Type> __val,
  _Order,
  _Operand,
  _Scope __scope,
  __cuda_atomic_mmio_disable)
{
  ::__nv_atomic_store(__cuda_atomic_nvvm_ptr(__ptr),
                      &__val,
                      +__cuda_atomic_nvvm_order<_Order>::__value,
                      +__cuda_atomic_nvvm_scope<_Scope>::__value);
}

template <class _Type,
          class _Cas,
          class _Order,
          class _Operand,
          class _Scope,
          enable_if_t<(_Operand::__size != 8) || (_CCCL_PTX_ARCH() >= 1000), bool> = false>
[[nodiscard]] _CCCL_DEVICE_API bool __cuda_atomic_compare_exchange(
  __cuda_atomic_nvvm_backend,
  _Type* __ptr,
  __unv<_Type>& __dst,
  __unv<_Type> __cmp,
  __unv<_Type> __op,
  _Cas,
  _Order,
  _Operand,
  _Scope __scope)
{
  using __orders  = __cuda_atomic_nvvm_cas_orders<_Order>;
  using __success = typename __orders::__success;
  using __failure = typename __orders::__failure;
  __dst           = __cmp;

  if constexpr (_Operand::__size == 128)
  {
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_90,
      (return ::__nv_atomic_compare_exchange(
                __cuda_atomic_nvvm_ptr(__ptr),
                &__dst,
                &__op,
                __cuda_atomic_cas_is_weak(_Cas{}),
                +__cuda_atomic_nvvm_order<__success>::__value,
                +__cuda_atomic_nvvm_order<__failure>::__value,
                +__cuda_atomic_nvvm_scope<_Scope>::__value);),
      (__cuda_atomic_nvvm_cas_128b_unsupported_before_SM_90(); return false;))
  }
  else
  {
    return ::__nv_atomic_compare_exchange(
      __cuda_atomic_nvvm_ptr(__ptr),
      &__dst,
      &__op,
      __cuda_atomic_cas_is_weak(_Cas{}),
      +__cuda_atomic_nvvm_order<__success>::__value,
      +__cuda_atomic_nvvm_order<__failure>::__value,
      +__cuda_atomic_nvvm_scope<_Scope>::__value);
  }
}

template <class _Type, class _Order, class _Operand, class _Scope>
_CCCL_DEVICE_API void __cuda_atomic_exchange(
  __cuda_atomic_nvvm_backend,
  _Type* __ptr,
  __unv<_Type>& __dst,
  __unv<_Type> __op,
  _Order __order,
  _Operand,
  _Scope __scope)
{
  if constexpr (_Operand::__size < 32)
  {
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_100,
      (::__nv_atomic_exchange(
         __cuda_atomic_nvvm_ptr(__ptr),
         &__op,
         &__dst,
         +__cuda_atomic_nvvm_order<_Order>::__value,
         +__cuda_atomic_nvvm_scope<_Scope>::__value);),
      (__dst = __cuda_atomic_fetch_update(
         __cuda_atomic_nvvm_backend{},
         __ptr,
         __cuda_atomic_op_bind<__unv<_Type>, __cuda_atomic_op_store>{__op},
         __order,
         _Operand{},
         __scope);))
  }
  else if constexpr (_Operand::__size == 128)
  {
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_90,
      (::__nv_atomic_exchange(
         __cuda_atomic_nvvm_ptr(__ptr),
         &__op,
         &__dst,
         +__cuda_atomic_nvvm_order<_Order>::__value,
         +__cuda_atomic_nvvm_scope<_Scope>::__value);),
      (__cuda_atomic_nvvm_exchange_128b_unsupported_before_SM_90();))
  }
  else
  {
    ::__nv_atomic_exchange(
      __cuda_atomic_nvvm_ptr(__ptr),
      &__op,
      &__dst,
      +__cuda_atomic_nvvm_order<_Order>::__value,
      +__cuda_atomic_nvvm_scope<_Scope>::__value);
  }
}

#  define _CCCL_DEFINE_NVVM_FETCH_ARITHMETIC(_Name)                                                                   \
    template <class _Type,                                                                                            \
              class _Order,                                                                                           \
              class _Operand,                                                                                         \
              class _Scope,                                                                                           \
              enable_if_t<(_Operand::__size < 128) && ((_Operand::__size >= 32) || (_CCCL_PTX_ARCH() >= 1000))        \
                            && !(is_integral_v<_Type> && is_signed_v<_Type> && sizeof(_Type) == 8),                   \
                          bool> = false>                                                                              \
    _CCCL_DEVICE_API void __cuda_atomic_fetch_##_Name(                                                                \
      __cuda_atomic_nvvm_backend, _Type* __ptr, __unv<_Type>& __dst, __unv<_Type> __op, _Order, _Operand, _Scope)     \
    {                                                                                                                 \
      __dst = ::__nv_atomic_fetch_##_Name(                                                                            \
        __cuda_atomic_nvvm_ptr(__ptr),                                                                                \
        __op,                                                                                                         \
        +__cuda_atomic_nvvm_order<_Order>::__value,                                                                   \
        +__cuda_atomic_nvvm_scope<_Scope>::__value);                                                                  \
    }                                                                                                                 \
                                                                                                                      \
    template <class _Type,                                                                                            \
              class _Order,                                                                                           \
              class _Operand,                                                                                         \
              class _Scope,                                                                                           \
              enable_if_t<is_integral_v<_Type> && is_signed_v<_Type> && sizeof(_Type) == 8 && _Operand::__size == 64, \
                          bool> = false>                                                                              \
    _CCCL_DEVICE_API void __cuda_atomic_fetch_##_Name(                                                                \
      __cuda_atomic_nvvm_backend, _Type* __ptr, __unv<_Type>& __dst, __unv<_Type> __op, _Order, _Operand, _Scope)     \
    {                                                                                                                 \
      const auto __result = ::__nv_atomic_fetch_##_Name(                                                              \
        reinterpret_cast<uint64_t*>(__cuda_atomic_nvvm_ptr(__ptr)),                                                   \
        ::cuda::std::bit_cast<uint64_t>(__op),                                                                        \
        +__cuda_atomic_nvvm_order<_Order>::__value,                                                                   \
        +__cuda_atomic_nvvm_scope<_Scope>::__value);                                                                  \
      __dst = ::cuda::std::bit_cast<__unv<_Type>>(__result);                                                          \
    }

#  define _CCCL_DEFINE_NVVM_FETCH_OP(_Name, _TypeConstraint)                                                      \
    template <class _Type,                                                                                        \
              class _Order,                                                                                       \
              class _Operand,                                                                                     \
              class _Scope,                                                                                       \
              enable_if_t<(_Operand::__size < 128) && ((_Operand::__size >= 32) || (_CCCL_PTX_ARCH() >= 1000))    \
                            && (_TypeConstraint),                                                                 \
                          bool> = false>                                                                          \
    _CCCL_DEVICE_API void __cuda_atomic_fetch_##_Name(                                                            \
      __cuda_atomic_nvvm_backend, _Type* __ptr, __unv<_Type>& __dst, __unv<_Type> __op, _Order, _Operand, _Scope) \
    {                                                                                                             \
      __dst = ::__nv_atomic_fetch_##_Name(                                                                        \
        __cuda_atomic_nvvm_ptr(__ptr),                                                                            \
        __op,                                                                                                     \
        +__cuda_atomic_nvvm_order<_Order>::__value,                                                               \
        +__cuda_atomic_nvvm_scope<_Scope>::__value);                                                              \
    }

_CCCL_DEFINE_NVVM_FETCH_ARITHMETIC(add)
_CCCL_DEFINE_NVVM_FETCH_ARITHMETIC(sub)
_CCCL_DEFINE_NVVM_FETCH_OP(and, true)
_CCCL_DEFINE_NVVM_FETCH_OP(or, true)
_CCCL_DEFINE_NVVM_FETCH_OP(xor, true)
_CCCL_DEFINE_NVVM_FETCH_OP(min, is_integral_v<_Type>)
_CCCL_DEFINE_NVVM_FETCH_OP(max, is_integral_v<_Type>)

#  undef _CCCL_DEFINE_NVVM_FETCH_ARITHMETIC
#  undef _CCCL_DEFINE_NVVM_FETCH_OP

template <class _Scope>
struct __cuda_atomic_nvvm_fence
{
  template <class _Order>
  _CCCL_DEVICE_API void operator()(_Order) const
  {
    ::__nv_atomic_thread_fence(+__cuda_atomic_nvvm_order<_Order>::__value, +__cuda_atomic_nvvm_scope<_Scope>::__value);
  }
};

template <class _Scope>
_CCCL_DEVICE_API void
__cuda_atomic_thread_fence(__cuda_atomic_nvvm_backend __backend, memory_order __order, _Scope __scope)
{
  (void) __backend;
  (void) __scope;
  __cuda_atomic_nvvm_fence<_Scope> __fence;
  switch (__atomic_order_to_int(__order))
  {
    case __ATOMIC_RELAXED:
      return;
    case __ATOMIC_CONSUME:
      [[fallthrough]];
    case __ATOMIC_ACQUIRE:
      return __fence(__cuda_atomic_order_acquire{});
    case __ATOMIC_RELEASE:
      return __fence(__cuda_atomic_order_release{});
    case __ATOMIC_ACQ_REL:
      return __fence(__cuda_atomic_order_acq_rel{});
    case __ATOMIC_SEQ_CST:
      return __fence(__cuda_atomic_order_seq_cst{});
    default:
      _CCCL_ASSERT(false, "invalid fence memory order");
  }
}

_CCCL_DEVICE_API void __cuda_atomic_signal_fence(__cuda_atomic_nvvm_backend, memory_order)
{
  asm volatile("" ::: "memory");
}

#endif // _CCCL_CTK_AT_LEAST(13, 5) && _CCCL_HAS_NV_ATOMIC_BUILTINS()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ATOMIC_FUNCTIONS_CUDA_NVVM_H
