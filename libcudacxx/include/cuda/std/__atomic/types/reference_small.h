//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ATOMIC_TYPES_REFERENCE_SMALL_H
#define _CUDA_STD___ATOMIC_TYPES_REFERENCE_SMALL_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/functions/host.h>
#include <cuda/std/__atomic/order.h>
#include <cuda/std/__atomic/scopes.h>
#include <cuda/std/__atomic/types/common.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/cstdint>
#include <cuda/std/cstring>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

namespace __atomic_ref_small_detail
{
constexpr unsigned int __lock_count = 1 << 12;

#if _CCCL_CUDA_COMPILATION()
// Device-side lock table used to provide byte-granular atomic_ref semantics.
// This avoids triggering compute-sanitizer memcheck false positives on
// architectures that internally promote sub-32-bit atomics to 32-bit accesses.
// See https://github.com/NVIDIA/cccl/issues/6430 for details.
inline __device__ unsigned int __lock_table[__lock_count];
#endif // _CCCL_CUDA_COMPILATION()

_CCCL_HOST_DEVICE inline unsigned int __hash_ptr(const void* __ptr)
{
  auto __address = reinterpret_cast<::cuda::std::uintptr_t>(__ptr);
  __address >>= 2;
  return static_cast<unsigned int>(__address & (__lock_count - 1));
}

#if _CCCL_CUDA_COMPILATION()
_CCCL_DEVICE inline unsigned int __lock_acquire_device(const void* __ptr)
{
  unsigned int __index = __hash_ptr(__ptr);
  while (atomicCAS(&__lock_table[__index], 0u, 1u) != 0u)
  {
    // spin
  }
  __threadfence();
  return __index;
}

_CCCL_DEVICE inline void __lock_release_device(unsigned int __index)
{
  __threadfence();
  atomicExch(&__lock_table[__index], 0u);
}
#endif // _CCCL_CUDA_COMPILATION()

template <typename _Sco>
_CCCL_HOST_DEVICE inline unsigned int __lock_acquire(const void* __ptr, _Sco)
{
#if defined(__CUDA_ARCH__)
  return __lock_acquire_device(__ptr);
#else
  (void) __ptr;
  (void) sizeof(_Sco);
  return 0;
#endif
}

template <typename _Sco>
_CCCL_HOST_DEVICE inline void __lock_release(unsigned int __index, _Sco)
{
#if defined(__CUDA_ARCH__)
  __lock_release_device(__index);
#else
  (void) __index;
  (void) sizeof(_Sco);
#endif
}
} // namespace __atomic_ref_small_detail

template <typename _Tp>
_CCCL_HOST_DEVICE inline void __atomic_ref_small_assign(_Tp* __ptr, const _Tp& __value)
{
  __atomic_assign_volatile(__ptr, __value);
}

template <typename _Sto, typename _Sco, __atomic_storage_is_ref_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_load_dispatch(const _Sto* __a, memory_order __order, _Sco __scope)
  -> __atomic_underlying_t<_Sto>
{
  using _Tp = __atomic_underlying_t<_Sto>;
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (auto __ptr          = const_cast<_Tp*>(__a->get());
     unsigned int __lock = __atomic_ref_small_detail::__lock_acquire(__ptr, __scope);
     _Tp __value{};
     __atomic_ref_small_assign(&__value, *__ptr);
     __atomic_ref_small_detail::__lock_release(__lock, __scope);
     return __value;),
    (return __atomic_load_host(const_cast<_Tp*>(__a->get()), __order);))
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_ref_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline void __atomic_store_dispatch(_Sto* __a, _Up __val, memory_order __order, _Sco __scope = {})
{
  using _Tp = __atomic_underlying_t<_Sto>;
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (auto __ptr = __a->get(); unsigned int __lock = __atomic_ref_small_detail::__lock_acquire(__ptr, __scope);
     _Tp __converted(__val);
     __atomic_ref_small_assign(__ptr, __converted);
     __atomic_ref_small_detail::__lock_release(__lock, __scope);),
    (__atomic_store_host(__a->get(), __val, __order);))
}

template <typename _Sto, typename _Sco, __atomic_storage_is_ref_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline void __atomic_init_dispatch(_Sto* __a, __atomic_underlying_t<_Sto> __value, _Sco __scope = {})
{
  __atomic_store_dispatch(__a, __value, memory_order_relaxed, __scope);
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_ref_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_exchange_dispatch(_Sto* __a, _Up __value, memory_order __order, _Sco __scope = {})
  -> __atomic_underlying_t<_Sto>
{
  using _Tp = __atomic_underlying_t<_Sto>;
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (auto __ptr = __a->get(); unsigned int __lock = __atomic_ref_small_detail::__lock_acquire(__ptr, __scope);
     _Tp __old{};
     __atomic_ref_small_assign(&__old, *__ptr);
     _Tp __converted(__value);
     __atomic_ref_small_assign(__ptr, __converted);
     __atomic_ref_small_detail::__lock_release(__lock, __scope);
     return __old;),
    (return __atomic_exchange_host(__a->get(), __value, __order);))
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_ref_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline bool __atomic_compare_exchange_strong_dispatch(
  _Sto* __a, _Up* __expected, _Up __desired, memory_order __success, memory_order __failure, _Sco __scope = {})
{
  using _Tp = __atomic_underlying_t<_Sto>;
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (
      auto __ptr = __a->get(); unsigned int __lock = __atomic_ref_small_detail::__lock_acquire(__ptr, __scope);
      _Tp __current{};
      __atomic_ref_small_assign(&__current, *__ptr);
      bool __matched = (__current == *__expected);
      if (__matched) {
        _Tp __converted(__desired);
        __atomic_ref_small_assign(__ptr, __converted);
      } else {
        __atomic_ref_small_assign(__expected, __current);
      } __atomic_ref_small_detail::__lock_release(__lock, __scope);
      (void) __success;
      (void) __failure;
      return __matched;),
    (return __atomic_compare_exchange_strong_host(__a->get(), __expected, __desired, __success, __failure);))
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_ref_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline bool __atomic_compare_exchange_weak_dispatch(
  _Sto* __a, _Up* __expected, _Up __desired, memory_order __success, memory_order __failure, _Sco __scope = {})
{
  return __atomic_compare_exchange_strong_dispatch(__a, __expected, __desired, __success, __failure, __scope);
}

template <typename _Sto, typename _Op, typename _Up, typename _Sco>
_CCCL_HOST_DEVICE inline auto __atomic_ref_small_modify(_Sto* __a, _Up __operand, _Op __operation, _Sco __scope = {})
  -> __atomic_underlying_t<_Sto>
{
  using _Tp           = __atomic_underlying_t<_Sto>;
  auto __ptr          = __a->get();
  unsigned int __lock = __atomic_ref_small_detail::__lock_acquire(__ptr, __scope);
  _Tp __old{};
  __atomic_ref_small_assign(&__old, *__ptr);
  _Tp __updated = __operation(__old, __operand);
  __atomic_ref_small_assign(__ptr, __updated);
  __atomic_ref_small_detail::__lock_release(__lock, __scope);
  return __old;
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_ref_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto
__atomic_fetch_add_dispatch(_Sto* __a, _Up __delta, memory_order __order, _Sco __scope = {})
  -> __atomic_underlying_t<_Sto>
{
  using _Tp = __atomic_underlying_t<_Sto>;
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (
      if constexpr (is_floating_point_v<_Tp>) {
        return __atomic_ref_small_modify(
          __a,
          __delta,
          [] _CCCL_HOST_DEVICE(const _Tp& __value, _Up __op) {
            return static_cast<_Tp>(__value + __op);
          },
          __scope);
      } else {
        return __atomic_ref_small_modify(
          __a,
          __delta,
          [] _CCCL_HOST_DEVICE(const _Tp& __value, _Up __op) {
            return static_cast<_Tp>(__value + __op);
          },
          __scope);
      }),
    (return __atomic_fetch_add_host(__a->get(), __delta, __order);))
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_ref_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto
__atomic_fetch_sub_dispatch(_Sto* __a, _Up __delta, memory_order __order, _Sco __scope = {})
  -> __atomic_underlying_t<_Sto>
{
  using _Tp = __atomic_underlying_t<_Sto>;
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (
      if constexpr (is_floating_point_v<_Tp>) {
        return __atomic_ref_small_modify(
          __a,
          __delta,
          [] _CCCL_HOST_DEVICE(const _Tp& __value, _Up __op) {
            return static_cast<_Tp>(__value - __op);
          },
          __scope);
      } else {
        return __atomic_ref_small_modify(
          __a,
          __delta,
          [] _CCCL_HOST_DEVICE(const _Tp& __value, _Up __op) {
            return static_cast<_Tp>(__value - __op);
          },
          __scope);
      }),
    (return __atomic_fetch_sub_host(__a->get(), __delta, __order);))
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_ref_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto
__atomic_fetch_and_dispatch(_Sto* __a, _Up __pattern, memory_order __order, _Sco __scope = {})
  -> __atomic_underlying_t<_Sto>
{
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (return __atomic_ref_small_modify(
              __a,
              __pattern,
              [] _CCCL_HOST_DEVICE(const __atomic_underlying_t<_Sto>& __value, _Up __op) {
                return static_cast<__atomic_underlying_t<_Sto>>(__value & __op);
              },
              __scope);),
    (return __atomic_fetch_and_host(__a->get(), __pattern, __order);))
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_ref_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto
__atomic_fetch_or_dispatch(_Sto* __a, _Up __pattern, memory_order __order, _Sco __scope = {})
  -> __atomic_underlying_t<_Sto>
{
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (return __atomic_ref_small_modify(
              __a,
              __pattern,
              [] _CCCL_HOST_DEVICE(const __atomic_underlying_t<_Sto>& __value, _Up __op) {
                return static_cast<__atomic_underlying_t<_Sto>>(__value | __op);
              },
              __scope);),
    (return __atomic_fetch_or_host(__a->get(), __pattern, __order);))
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_ref_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto
__atomic_fetch_xor_dispatch(_Sto* __a, _Up __pattern, memory_order __order, _Sco __scope = {})
  -> __atomic_underlying_t<_Sto>
{
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (return __atomic_ref_small_modify(
              __a,
              __pattern,
              [] _CCCL_HOST_DEVICE(const __atomic_underlying_t<_Sto>& __value, _Up __op) {
                return static_cast<__atomic_underlying_t<_Sto>>(__value ^ __op);
              },
              __scope);),
    (return __atomic_fetch_xor_host(__a->get(), __pattern, __order);))
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_ref_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_fetch_max_dispatch(_Sto* __a, _Up __val, memory_order __order, _Sco __scope = {})
  -> __atomic_underlying_t<_Sto>
{
  using _Tp = __atomic_underlying_t<_Sto>;
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (return __atomic_ref_small_modify(
              __a,
              __val,
              [] _CCCL_HOST_DEVICE(const _Tp& __current, _Up __operand) {
                return __current < __operand ? static_cast<_Tp>(__operand) : __current;
              },
              __scope);),
    (return __atomic_fetch_max_host(__a->get(), __val, __order);))
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_ref_small<_Sto> = 0>
_CCCL_HOST_DEVICE inline auto __atomic_fetch_min_dispatch(_Sto* __a, _Up __val, memory_order __order, _Sco __scope = {})
  -> __atomic_underlying_t<_Sto>
{
  using _Tp = __atomic_underlying_t<_Sto>;
  NV_IF_TARGET(
    NV_IS_DEVICE,
    (return __atomic_ref_small_modify(
              __a,
              __val,
              [] _CCCL_HOST_DEVICE(const _Tp& __current, _Up __operand) {
                return __current > __operand ? static_cast<_Tp>(__operand) : __current;
              },
              __scope);),
    (return __atomic_fetch_min_host(__a->get(), __val, __order);))
}

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ATOMIC_TYPES_REFERENCE_SMALL_H
