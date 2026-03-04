//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ATOMIC_BACKENDS_CUDA_NVVM_H
#  define _CUDA_STD___ATOMIC_BACKENDS_CUDA_NVVM_H

#  include <cuda/std/detail/__config>

#  if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#    pragma GCC system_header
#  elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#    pragma clang system_header
#  elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#    pragma system_header
#  endif // no system header

#  include <cuda/std/__atomic/backends/common.h>
#  include <cuda/std/__atomic/backends/cuda_local.h>
#  include <cuda/std/__atomic/backends/cuda_nvvm_wrapped.h>
#  include <cuda/std/__atomic/backends/cuda_supported_atomics_helper.h>
#  include <cuda/std/__atomic/order.h>
#  include <cuda/std/__atomic/scopes.h>
#  include <cuda/std/__type_traits/enable_if.h>
#  include <cuda/std/__type_traits/is_signed.h>
#  include <cuda/std/__type_traits/is_unsigned.h>
#  include <cuda/std/cassert>
#  include <cuda/std/cstdint>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

#  if _CCCL_CUDA_COMPILATION()

template <typename _Sco>
static inline _CCCL_DEVICE void __atomic_thread_fence_cuda(int __memorder, _Sco)
{
  // nv_atomic_thread_fence_nvvm_dispatch(__memorder, __atomic_scope_tag_to_nvvm_scope(_Sco{}));
}

template <>
inline _CCCL_DEVICE void __atomic_thread_fence_cuda(int __memorder, ::cuda::std::__thread_scope_thread_tag)
{}

template <class _Type, class _Sco>
static inline _CCCL_DEVICE void __atomic_load_cuda(const _Type* __ptr, _Type& __dst, int __memorder, _Sco)
{
  using __proxy_t              = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag            = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  const __proxy_t* __ptr_proxy = reinterpret_cast<const __proxy_t*>(__ptr);
  __proxy_t* __dst_proxy       = reinterpret_cast<__proxy_t*>(&__dst);
  if (__cuda_load_weak_if_local(__ptr_proxy, __dst_proxy, sizeof(__proxy_t)))
  {
    return;
  }
  __atomic_load_nvvm_dispatch(__ptr_proxy, __dst_proxy, __memorder, __atomic_scope_tag_to_nvvm_scope(_Sco{}));
}
template <class _Type, class _Sco>
static inline _CCCL_DEVICE void __atomic_load_cuda(const _Type volatile* __ptr, _Type& __dst, int __memorder, _Sco)
{
  using __proxy_t              = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag            = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  const __proxy_t* __ptr_proxy = reinterpret_cast<const __proxy_t*>(const_cast<_Type*>(__ptr));
  __proxy_t* __dst_proxy       = reinterpret_cast<__proxy_t*>(&__dst);
  if (__cuda_load_weak_if_local(__ptr_proxy, __dst_proxy, sizeof(__proxy_t)))
  {
    return;
  }
  __atomic_load_nvvm_dispatch(__ptr_proxy, __dst_proxy, __memorder, __atomic_scope_tag_to_nvvm_scope(_Sco{}));
}

template <class _Type, class _Sco>
static inline _CCCL_DEVICE void __atomic_store_cuda(_Type* __ptr, _Type& __val, int __memorder, _Sco)
{
  using __proxy_t        = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag      = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(__ptr);
  __proxy_t* __val_proxy = reinterpret_cast<__proxy_t*>(&__val);
  if (__cuda_store_weak_if_local(__ptr_proxy, __val_proxy, sizeof(__proxy_t)))
  {
    return;
  }
  __atomic_store_nvvm_dispatch(__ptr_proxy, __val_proxy, __memorder, __atomic_scope_tag_to_nvvm_scope(_Sco{}));
}
template <class _Type, class _Sco>
static inline _CCCL_DEVICE void __atomic_store_cuda(_Type volatile* __ptr, _Type& __val, int __memorder, _Sco)
{
  using __proxy_t        = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag      = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(const_cast<_Type*>(__ptr));
  __proxy_t* __val_proxy = reinterpret_cast<__proxy_t*>(&__val);
  if (__cuda_store_weak_if_local(__ptr_proxy, __val_proxy, sizeof(__proxy_t)))
  {
    return;
  }
  __atomic_store_nvvm_dispatch(__ptr_proxy, __val_proxy, __memorder, __atomic_scope_tag_to_nvvm_scope(_Sco{}));
}

template <class _Type, class _Sco>
static inline _CCCL_DEVICE bool __atomic_compare_exchange_cuda(
  _Type* __ptr, _Type* __exp, _Type __des, bool __weak, int __success_memorder, int __failure_memorder, _Sco)
{
  using __proxy_t        = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag      = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(__ptr);
  __proxy_t* __exp_proxy = reinterpret_cast<__proxy_t*>(__exp);
  __proxy_t* __des_proxy = reinterpret_cast<__proxy_t*>(&__des);
  bool __res             = false;
  if (__cuda_compare_exchange_weak_if_local(__ptr_proxy, __exp_proxy, __des_proxy, &__res))
  {
    return __res;
  }
  return __nv_atomic_compare_exchange_nvvm_dispatch(
    __ptr_proxy,
    __exp_proxy,
    __des_proxy,
    __weak,
    __success_memorder,
    __failure_memorder,
    __atomic_scope_tag_to_nvvm_scope(_Sco{}));
}
template <class _Type, class _Sco>
static inline _CCCL_DEVICE bool __atomic_compare_exchange_cuda(
  _Type volatile* __ptr, _Type* __exp, _Type __des, bool __weak, int __success_memorder, int __failure_memorder, _Sco)
{
  using __proxy_t        = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag      = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(const_cast<_Type*>(__ptr));
  __proxy_t* __exp_proxy = reinterpret_cast<__proxy_t*>(__exp);
  __proxy_t* __des_proxy = reinterpret_cast<__proxy_t*>(&__des);
  bool __res             = false;
  if (__cuda_compare_exchange__weak_if_local(__ptr_proxy, __exp_proxy, __des_proxy, &__res))
  {
    return __res;
  }
  return __nv_atomic_compare_exchange_nvvm_dispatch(
    __ptr_proxy,
    __exp_proxy,
    __des_proxy,
    __weak,
    __success_memorder,
    __failure_memorder,
    __atomic_scope_tag_to_nvvm_scope(_Sco{}));
}

template <class _Type, class _Sco>
static inline _CCCL_DEVICE void __atomic_exchange_cuda(_Type* __ptr, _Type& __old, _Type __new, int __memorder, _Sco)
{
  using __proxy_t        = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag      = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(__ptr);
  __proxy_t* __old_proxy = reinterpret_cast<__proxy_t*>(&__old);
  __proxy_t* __new_proxy = reinterpret_cast<__proxy_t*>(&__new);
  if (__cuda_exchange_weak_if_local(__ptr_proxy, __new_proxy, __old_proxy))
  {
    return;
  }
  __atomic_exchange_nvvm_dispatch(
    __ptr_proxy, __old_proxy, __new_proxy, __memorder, __atomic_scope_tag_to_nvvm_scope(_Sco{}));
}
template <class _Type, class _Sco>
static inline _CCCL_DEVICE void
__atomic_exchange_cuda(_Type volatile* __ptr, _Type& __old, _Type __new, int __memorder, _Sco)
{
  using __proxy_t        = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag      = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(const_cast<_Type*>(__ptr));
  __proxy_t* __old_proxy = reinterpret_cast<__proxy_t*>(&__old);
  __proxy_t* __new_proxy = reinterpret_cast<__proxy_t*>(&__new);
  if (__cuda_exchange_weak_if_local(__ptr_proxy, __new_proxy, __old_proxy))
  {
    return;
  }
  __atomic_exchange_nvvm_dispatch(
    __ptr_proxy, __old_proxy, __new_proxy, __memorder, __atomic_scope_tag_to_nvvm_scope(_Sco{}));
}

template <class _Type, class _Up, class _Sco, __atomic_enable_if_native_bitwise<_Type> = 0>
static inline _CCCL_DEVICE _Type __atomic_fetch_and_cuda(_Type* __ptr, _Up __op, int __memorder, _Sco)
{
  using __proxy_t   = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  _Type __dst{};
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(__ptr);
  __proxy_t* __dst_proxy = reinterpret_cast<__proxy_t*>(&__dst);
  __proxy_t* __op_proxy  = reinterpret_cast<__proxy_t*>(&__op);
  if (__cuda_fetch_and_weak_if_local(__ptr_proxy, *__op_proxy, __dst_proxy))
  {
    return __dst;
  }
  return __nv_atomic_fetch_and_nvvm_dispatch(
    __ptr_proxy, __op_proxy, __memorder, __atomic_scope_tag_to_nvvm_scope(_Sco{}));
}
template <class _Type, class _Up, class _Sco, __atomic_enable_if_native_bitwise<_Type> = 0>
static inline _CCCL_DEVICE _Type __atomic_fetch_and_cuda(_Type volatile* __ptr, _Up __op, int __memorder, _Sco)
{
  using __proxy_t   = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  _Type __dst{};
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(const_cast<_Type*>(__ptr));
  __proxy_t* __dst_proxy = reinterpret_cast<__proxy_t*>(&__dst);
  __proxy_t* __op_proxy  = reinterpret_cast<__proxy_t*>(&__op);
  if (__cuda_fetch_and_weak_if_local(__ptr_proxy, *__op_proxy, __dst_proxy))
  {
    return __dst;
  }
  return __nv_atomic_fetch_and_nvvm_dispatch(
    __ptr_proxy, __op_proxy, __memorder, __atomic_scope_tag_to_nvvm_scope(_Sco{}));
}

template <class _Type, class _Up, class _Sco, __atomic_enable_if_native_minmax<_Type> = 0>
static inline _CCCL_DEVICE _Type __atomic_fetch_max_cuda(_Type* __ptr, _Up __op, int __memorder, _Sco)
{
  using __proxy_t   = typename __atomic_cuda_deduce_minmax<_Type>::__type;
  using __proxy_tag = typename __atomic_cuda_deduce_minmax<_Type>::__tag;
  _Type __dst{};
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(__ptr);
  __proxy_t* __dst_proxy = reinterpret_cast<__proxy_t*>(&__dst);
  __proxy_t* __op_proxy  = reinterpret_cast<__proxy_t*>(&__op);
  if (__cuda_fetch_max_weak_if_local(__ptr_proxy, *__op_proxy, __dst_proxy))
  {
    return __dst;
  }
  return __nv_atomic_fetch_max_nvvm_dispatch(
    __ptr_proxy, __op_proxy, __memorder, __atomic_scope_tag_to_nvvm_scope(_Sco{}));
}
template <class _Type, class _Up, class _Sco, __atomic_enable_if_native_minmax<_Type> = 0>
static inline _CCCL_DEVICE _Type __atomic_fetch_max_cuda(_Type volatile* __ptr, _Up __op, int __memorder, _Sco)
{
  using __proxy_t   = typename __atomic_cuda_deduce_minmax<_Type>::__type;
  using __proxy_tag = typename __atomic_cuda_deduce_minmax<_Type>::__tag;
  _Type __dst{};
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(const_cast<_Type*>(__ptr));
  __proxy_t* __dst_proxy = reinterpret_cast<__proxy_t*>(&__dst);
  __proxy_t* __op_proxy  = reinterpret_cast<__proxy_t*>(&__op);
  if (__cuda_fetch_max_weak_if_local(__ptr_proxy, *__op_proxy, __dst_proxy))
  {
    return __dst;
  }
  return __nv_atomic_fetch_max_nvvm_dispatch(
    __ptr_proxy, __op_proxy, __memorder, __atomic_scope_tag_to_nvvm_scope(_Sco{}));
}

template <class _Type, class _Up, class _Sco, __atomic_enable_if_native_minmax<_Type> = 0>
static inline _CCCL_DEVICE _Type __atomic_fetch_min_cuda(_Type* __ptr, _Up __op, int __memorder, _Sco)
{
  using __proxy_t   = typename __atomic_cuda_deduce_minmax<_Type>::__type;
  using __proxy_tag = typename __atomic_cuda_deduce_minmax<_Type>::__tag;
  _Type __dst{};
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(__ptr);
  __proxy_t* __dst_proxy = reinterpret_cast<__proxy_t*>(&__dst);
  __proxy_t* __op_proxy  = reinterpret_cast<__proxy_t*>(&__op);
  if (__cuda_fetch_min_weak_if_local(__ptr_proxy, *__op_proxy, __dst_proxy))
  {
    return __dst;
  }
  return __nv_atomic_fetch_min_nvvm_dispatch(
    __ptr_proxy, __op_proxy, __memorder, __atomic_scope_tag_to_nvvm_scope(_Sco{}));
}
template <class _Type, class _Up, class _Sco, __atomic_enable_if_native_minmax<_Type> = 0>
static inline _CCCL_DEVICE _Type __atomic_fetch_min_cuda(_Type volatile* __ptr, _Up __op, int __memorder, _Sco)
{
  using __proxy_t   = typename __atomic_cuda_deduce_minmax<_Type>::__type;
  using __proxy_tag = typename __atomic_cuda_deduce_minmax<_Type>::__tag;
  _Type __dst{};
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(const_cast<_Type*>(__ptr));
  __proxy_t* __dst_proxy = reinterpret_cast<__proxy_t*>(&__dst);
  __proxy_t* __op_proxy  = reinterpret_cast<__proxy_t*>(&__op);
  if (__cuda_fetch_min_weak_if_local(__ptr_proxy, *__op_proxy, __dst_proxy))
  {
    return __dst;
  }
  return __nv_atomic_fetch_min_nvvm_dispatch(
    __ptr_proxy, __op_proxy, __memorder, __atomic_scope_tag_to_nvvm_scope(_Sco{}));
}

template <class _Type, class _Up, class _Sco, __atomic_enable_if_native_bitwise<_Type> = 0>
static inline _CCCL_DEVICE _Type __atomic_fetch_or_cuda(_Type* __ptr, _Up __op, int __memorder, _Sco)
{
  using __proxy_t   = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  _Type __dst{};
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(__ptr);
  __proxy_t* __dst_proxy = reinterpret_cast<__proxy_t*>(&__dst);
  __proxy_t* __op_proxy  = reinterpret_cast<__proxy_t*>(&__op);
  if (__cuda_fetch_or_weak_if_local(__ptr_proxy, *__op_proxy, __dst_proxy))
  {
    return __dst;
  }
  return __nv_atomic_fetch_or_nvvm_dispatch(
    __ptr_proxy, __op_proxy, __memorder, __atomic_scope_tag_to_nvvm_scope(_Sco{}));
}
template <class _Type, class _Up, class _Sco, __atomic_enable_if_native_bitwise<_Type> = 0>
static inline _CCCL_DEVICE _Type __atomic_fetch_or_cuda(_Type volatile* __ptr, _Up __op, int __memorder, _Sco)
{
  using __proxy_t   = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  _Type __dst{};
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(const_cast<_Type*>(__ptr));
  __proxy_t* __dst_proxy = reinterpret_cast<__proxy_t*>(&__dst);
  __proxy_t* __op_proxy  = reinterpret_cast<__proxy_t*>(&__op);
  if (__cuda_fetch_or_weak_if_local(__ptr_proxy, *__op_proxy, __dst_proxy))
  {
    return __dst;
  }
  return __nv_atomic_fetch_or_nvvm_dispatch(
    __ptr_proxy, __op_proxy, __memorder, __atomic_scope_tag_to_nvvm_scope(_Sco{}));
}

template <class _Type, class _Up, class _Sco, __atomic_enable_if_native_bitwise<_Type> = 0>
static inline _CCCL_DEVICE _Type __atomic_fetch_xor_cuda(_Type* __ptr, _Up __op, int __memorder, _Sco)
{
  using __proxy_t   = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  _Type __dst{};
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(__ptr);
  __proxy_t* __dst_proxy = reinterpret_cast<__proxy_t*>(&__dst);
  __proxy_t* __op_proxy  = reinterpret_cast<__proxy_t*>(&__op);
  if (__cuda_fetch_xor_weak_if_local(__ptr_proxy, *__op_proxy, __dst_proxy))
  {
    return __dst;
  }
  return __nv_atomic_fetch_xor_nvvm_dispatch(
    __ptr_proxy, __op_proxy, __memorder, __atomic_scope_tag_to_nvvm_scope(_Sco{}));
}
template <class _Type, class _Up, class _Sco, __atomic_enable_if_native_bitwise<_Type> = 0>
static inline _CCCL_DEVICE _Type __atomic_fetch_xor_cuda(_Type volatile* __ptr, _Up __op, int __memorder, _Sco)
{
  using __proxy_t   = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  _Type __dst{};
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(const_cast<_Type*>(__ptr));
  __proxy_t* __dst_proxy = reinterpret_cast<__proxy_t*>(&__dst);
  __proxy_t* __op_proxy  = reinterpret_cast<__proxy_t*>(&__op);
  if (__cuda_fetch_xor_weak_if_local(__ptr_proxy, *__op_proxy, __dst_proxy))
  {
    return __dst;
  }
  return __nv_atomic_fetch_xor_nvvm_dispatch(
    __ptr_proxy, __op_proxy, __memorder, __atomic_scope_tag_to_nvvm_scope(_Sco{}));
}

template <class _Type, class _Up, class _Sco>
static inline _CCCL_DEVICE _Type __atomic_fetch_add_cuda(_Type* __ptr, _Up __op, int __memorder, _Sco)
{
  constexpr auto __skip_v = 1;
  __op                    = __op * __skip_v;
  using __proxy_t         = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag       = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  _Type __dst{};
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(const_cast<_Type*>(__ptr));
  __proxy_t* __dst_proxy = reinterpret_cast<__proxy_t*>(&__dst);
  __proxy_t* __op_proxy  = reinterpret_cast<__proxy_t*>(&__op);
  if (__cuda_fetch_add_weak_if_local(__ptr_proxy, *__op_proxy, __dst_proxy))
  {
    return __dst;
  }
  return __nv_atomic_fetch_add_nvvm_dispatch(
    __ptr_proxy, __op_proxy, __memorder, __atomic_scope_tag_to_nvvm_scope(_Sco{}));
}
template <class _Type, class _Up, class _Sco>
static inline _CCCL_DEVICE _Type __atomic_fetch_add_cuda(volatile _Type* __ptr, _Up __op, int __memorder, _Sco)
{
  constexpr auto __skip_v = 1;
  __op                    = __op * __skip_v;
  using __proxy_t         = typename __atomic_cuda_deduce_bitwise<_Type>::__type;
  using __proxy_tag       = typename __atomic_cuda_deduce_bitwise<_Type>::__tag;
  _Type __dst{};
  __proxy_t* __ptr_proxy = reinterpret_cast<__proxy_t*>(const_cast<_Type*>(__ptr));
  __proxy_t* __dst_proxy = reinterpret_cast<__proxy_t*>(&__dst);
  __proxy_t* __op_proxy  = reinterpret_cast<__proxy_t*>(&__op);
  if (__cuda_fetch_add_weak_if_local(__ptr_proxy, *__op_proxy, __dst_proxy))
  {
    return __dst;
  }
  return __nv_atomic_fetch_add_nvvm_dispatch(
    __ptr_proxy, __op_proxy, __memorder, __atomic_scope_tag_to_nvvm_scope(_Sco{}));
}

template <class _Type, class _Up, class _Sco>
static inline _CCCL_DEVICE _Type __atomic_fetch_sub_cuda(_Type* __ptr, _Up __op, int __memorder, _Sco)
{
  return __atomic_fetch_add_cuda(__ptr, -__op, __memorder, _Sco{});
}
template <class _Type, class _Up, class _Sco>
static inline _CCCL_DEVICE _Type __atomic_fetch_sub_cuda(_Type volatile* __ptr, _Up __op, int __memorder, _Sco)
{
  return __atomic_fetch_add_cuda(__ptr, -__op, __memorder, _Sco{});
}

#  endif // _CCCL_HAS_CUDA_COMPILER()

_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ATOMIC_BACKENDS_CUDA_NVVM_H

// clang-format on
