//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA_STD___ATOMIC_BACKENDS_DEVICE_FALLBACKS_H
#define __CUDA_STD___ATOMIC_BACKENDS_DEVICE_FALLBACKS_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/backends/cuda_supported_atomics_helper.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_scalar.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#if _CCCL_CUDA_COMPILATION()

template <class _Type>
struct __atomic_nvvm_dispatch_helper
{
  using __bitwise    = __atomic_cuda_deduce_bitwise<_Type>;
  using __arithmetic = __atomic_cuda_deduce_arithmetic<_Type>;
  using __minmax     = __atomic_cuda_deduce_minmax<_Type>;

  static constexpr __atomic_nvvm_is_native_arithmetic =
    /* fp16 and up */ ((_Operand::__size >= 16) && (_Operand::__op == __atomic_cuda_operand::_f)) ||
    /* 32 bits and up */ ((_Operand::__size >= 32));

  static constexpr __atomic_nvvm_is_native_arithmetic =
    /* fp16 and up */ ((_Operand::__size >= 16) && (_Operand::__op == __atomic_cuda_operand::_f)) ||
    /* 32 bits and up */ ((_Operand::__size >= 32));

  static constexpr __atomic_nvvm_is_native_bitwise =
    /* 32 bits and up */ ((__bitwise::__size >= 32));

  static constexpr __atomic_nvvm_is_native_cas =
    /* 16 bits and up */ ((__bitwise::__size >= 16));

  // Native ld/st differs from PTX due to missing 8 bit constraints in inline PTX
  static constexpr __atomic_nvvm_is_native_ld_st =
    /* 8 bits and up */ ((__bitwise::__size >= 8));

  using __enable_if_native_arithmetic     = enable_if_t<__atomic_nvvm_is_native_arithmetic, bool>;
  using __enable_if_not_native_arithmetic = enable_if_t<!__atomic_nvvm_is_native_arithmetic, bool>;

  using __enable_if_native_minmax     = enable_if_t<__atomic_nvvm_is_native_minmax, bool>;
  using __enable_if_not_native_minmax = enable_if_t<!__atomic_nvvm_is_native_minmax, bool>;

  using __enable_if_native_bitwise     = enable_if_t<__atomic_nvvm_is_native_bitwise, bool>;
  using __enable_if_not_native_bitwise = enable_if_t<!__atomic_nvvm_is_native_bitwise, bool>;

  using __enable_if_native_cas     = enable_if_t<__atomic_nvvm_is_native_cas, bool>;
  using __enable_if_not_native_cas = enable_if_t<!__atomic_nvvm_is_native_cas, bool>;

  using __enable_if_native_ld_st     = enable_if_t<__atomic_nvvm_is_native_ld_st, bool>;
  using __enable_if_not_native_ld_st = enable_if_t<!__atomic_nvvm_is_native_ld_st, bool>;
};

template <class _Type, __atomic_nvvm_dispatch_helper<_Type>::__enable_if_not_native_ld_st = 0>
_CCCL_ARTIFICIAL static inline _CCCL_DEVICE void
__atomic_load_nvvm_dispatch(const _Type* __ptr, _Type* __dst, int __memorder, int __sco)
{}

template <class _Type, __atomic_nvvm_dispatch_helper<_Type>::__enable_if_not_native_ld_st = 0>
_CCCL_ARTIFICIAL static inline _CCCL_DEVICE void
__atomic_store_nvvm_dispatch(_Type* __ptr, _Type* __val, int __memorder, int __sco)
{}

template <class _Type, __atomic_nvvm_dispatch_helper<_Type>::__enable_if_not_native_cas = 0>
_CCCL_ARTIFICIAL static inline _CCCL_DEVICE bool __atomic_compare_exchange_nvvm_dispatch(
  _Type* __ptr, _Type* __exp, _Type* __des, bool __weak, int __success_memorder, int __failure_memorder, int __sco)
{}

template <class _Type, __atomic_nvvm_dispatch_helper<_Type>::__enable_if_not_native_cas = 0>
_CCCL_ARTIFICIAL static inline _CCCL_DEVICE void
__atomic_exchange_nvvm_dispatch(_Type* __atom, _Type* __val, _Type* __ret, int __memorder, int __sco)
{}

template <class _Type>
_CCCL_ARTIFICIAL static inline _CCCL_DEVICE _Type
__atomic_fetch_max_nvvm_dispatch(_Type* __ptr, _Type __op, int __memorder, int __sco)
{}

template <class _Type>
_CCCL_ARTIFICIAL static inline _CCCL_DEVICE _Type
__atomic_fetch_min_nvvm_dispatch(_Type* __ptr, _Type __op, int __memorder, int __sco)
{}

template <class _Type>
_CCCL_ARTIFICIAL static inline _CCCL_DEVICE _Type
__atomic_fetch_and_nvvm_dispatch(_Type* __ptr, _Type __op, int __memorder, int __sco)
{}

template <class _Type>
_CCCL_ARTIFICIAL static inline _CCCL_DEVICE _Type
__atomic_fetch_or_nvvm_dispatch(_Type* __ptr, _Type __op, int __memorder, int __sco)
{}

template <class _Type>
_CCCL_ARTIFICIAL static inline _CCCL_DEVICE _Type
__atomic_fetch_xor_nvvm_dispatch(_Type* __ptr, _Type __op, int __memorder, int __sco)
{}

template <class _Type>
_CCCL_ARTIFICIAL static inline _CCCL_DEVICE _Type
__atomic_fetch_add_nvvm_dispatch(_Type* __ptr, _Type __op, int __memorder, int __sco)
{}

template <class _Type, class _Order, class _Operand, class _Sco, __cuda_atomic_enable_non_native_ld_st<_Operand> = 0>
static inline _CCCL_DEVICE void __atomic_load_nonnative(const _Type* __ptr, _Type& __dst, _Order, _Operand, _Sco)
{
  constexpr uint64_t __alignmask = (sizeof(uint16_t) - 1);
  uint16_t* __aligned            = (uint16_t*) ((intptr_t) __ptr & (~__alignmask));
  const uint8_t __offset         = uint16_t((intptr_t) __ptr & __alignmask) * 8;

  uint16_t __value = 0;
  __cuda_atomic_load(__aligned, __value, _Order{}, __atomic_cuda_operand_b16{}, _Sco{}, __atomic_cuda_mmio_disable{});

  __dst = static_cast<_Type>(__value >> __offset);
}

template <class _Type, class _Order, class _Operand, class _Sco, __cuda_atomic_enable_non_native_bitwise<_Operand> = 0>
static inline _CCCL_DEVICE bool
__atomic_cas_nonnative(_Type* __ptr, _Type& __dst, _Type __cmp, _Type __op, _Order, _Operand, _Sco)
{
  constexpr uint64_t __alignmask = (sizeof(uint32_t) - 1);
  constexpr uint32_t __sizemask  = (1 << (sizeof(_Type) * 8)) - 1;
  uint32_t* __aligned            = (uint32_t*) ((intptr_t) __ptr & (~__alignmask));
  const uint8_t __offset         = uint32_t((intptr_t) __ptr & __alignmask) * 8;
  const uint32_t __valueMask     = __sizemask << __offset;
  const uint32_t __windowMask    = ~__valueMask;
  const uint32_t __cmpOffset     = __cmp << __offset;
  const uint32_t __opOffset      = __op << __offset;

  // Algorithm for 8b CAS with 32b intrinsics
  // __old = __window[0:32] where [__cmp] resides within some offset.
  uint32_t __old;
  // Start by loading __old with the current value, this optimizes for early return when __cmp is wrong
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (__cuda_atomic_load(
       __aligned, __old, __atomic_cuda_relaxed{}, __atomic_cuda_operand_b32{}, _Sco{}, __atomic_cuda_mmio_disable{});),
    (__cuda_atomic_load(
       __aligned, __old, __atomic_cuda_volatile{}, __atomic_cuda_operand_b32{}, _Sco{}, __atomic_cuda_mmio_disable{});))
  // Reemit CAS instructions until we succeed or the old value is a mismatch
  while (__cmpOffset == (__old & __valueMask))
  {
    // Combine the desired value and most recently fetched expected masked portion of the window
    const uint32_t __attempt = (__old & __windowMask) | __opOffset;

    if (__cuda_atomic_compare_exchange(
          __aligned, __old, __old, __attempt, _Order{}, __atomic_cuda_operand_b32{}, _Sco{}))
    {
      // CAS was successful
      return true;
    }
  }
  __dst = static_cast<_Type>(__old >> __offset);
  return false;
}

// Optimized fetch_update CAS loop with op determined after first load reducing waste.
template <class _Type,
          class _Fn,
          class _Order,
          class _Operand,
          class _Sco,
          __cuda_atomic_enable_non_native_bitwise<_Operand> = 0>
_CCCL_DEVICE _Type __atomic_fetch_update_nonnative(_Type* __ptr, const _Fn& __op, _Order, _Operand, _Sco)
{
  constexpr uint64_t __alignmask = (sizeof(uint32_t) - 1);
  constexpr uint32_t __sizemask  = (1 << (sizeof(_Type) * 8)) - 1;
  uint32_t* __aligned            = (uint32_t*) ((intptr_t) __ptr & (~__alignmask));
  const uint8_t __offset         = uint8_t((intptr_t) __ptr & __alignmask) * 8;
  const uint32_t __valueMask     = __sizemask << __offset;
  const uint32_t __windowMask    = ~__valueMask;

  // 8/16b fetch update is similar to CAS implementation, but compresses the logic for recalculating the operand
  // __old = __window[0:32] where [__cmp] resides within some offset.
  uint32_t __old;
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (__cuda_atomic_load(
       __aligned, __old, __atomic_cuda_relaxed{}, __atomic_cuda_operand_b32{}, _Sco{}, __atomic_cuda_mmio_disable{});),
    (__cuda_atomic_load(
       __aligned, __old, __atomic_cuda_volatile{}, __atomic_cuda_operand_b32{}, _Sco{}, __atomic_cuda_mmio_disable{});))

  // Reemit CAS instructions until we succeed
  while (1)
  {
    // Calculate new desired value from last fetched __old
    // Use of the value mask is required due to the possibility of overflow when ops are widened. Possible compiler bug?
    const uint32_t __attempt =
      ((static_cast<uint32_t>(__op(static_cast<_Type>(__old >> __offset))) << __offset) & __valueMask)
      | (__old & __windowMask);

    if (__cuda_atomic_compare_exchange(
          __aligned, __old, __old, __attempt, _Order{}, __atomic_cuda_operand_b32{}, _Sco{}))
    {
      // CAS was successful
      return static_cast<_Type>(__old >> __offset);
    }
  }
}

#endif // ^_CCCL_CUDA_COMPILATION()

_CCCL_END_NAMESPACE_CUDA_STD

#endif // __CUDA_STD___ATOMIC_BACKENDS_DEVICE_FALLBACKS_H
