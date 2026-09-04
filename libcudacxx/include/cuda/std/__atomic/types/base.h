//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ATOMIC_TYPES_BASE_H
#define _CUDA_STD___ATOMIC_TYPES_BASE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/functions.h>
#include <cuda/std/__atomic/types/common.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__type_traits/remove_cvref.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <typename _Tp>
struct __atomic_storage
{
  using __underlying_t                = _Tp;
  static constexpr __atomic_tag __tag = __atomic_tag::__atomic_base_tag;

#if _CCCL_COMPILER(GCC, <, 10) // older gcc fails to handle volatile in is_trivially_copyable
  static_assert(is_trivially_copyable_v<remove_cvref_t<_Tp>>,
                "std::atomic<Tp> requires that 'Tp' be a trivially copyable type");
#else // ^^^ _CCCL_COMPILER(GCC, <, 10) ^^^ / vvv !_CCCL_COMPILER(GCC, <, 10) vvv
  static_assert(is_trivially_copyable_v<_Tp>, "std::atomic<Tp> requires that 'Tp' be a trivially copyable type");
#endif // !_CCCL_COMPILER(GCC, <, 10)

  _CCCL_ALIGNAS(sizeof(_Tp)) _Tp __a_value;

  _CCCL_HIDE_FROM_ABI explicit constexpr __atomic_storage() noexcept = default;

  _CCCL_HOST_DEVICE_API constexpr explicit __atomic_storage(_Tp value) noexcept
      : __a_value(value)
  {}

  _CCCL_HOST_DEVICE_API auto get() noexcept -> __underlying_t*
  {
    return &__a_value;
  }
  _CCCL_HOST_DEVICE_API auto get() const noexcept -> const __underlying_t*
  {
    return &__a_value;
  }
  _CCCL_HOST_DEVICE_API auto get() volatile noexcept -> volatile __underlying_t*
  {
    return &__a_value;
  }
  _CCCL_HOST_DEVICE_API auto get() const volatile noexcept -> const volatile __underlying_t*
  {
    return &__a_value;
  }
};

#define _CCCL_DISPATCH_ATOMIC_BACKEND(_Fn, ...)                                  \
  NV_DISPATCH_TARGET(NV_IS_DEVICE,                                               \
                     (return _Fn(__cuda_atomic_device_backend{}, __VA_ARGS__);), \
                     NV_IS_HOST,                                                 \
                     (return _Fn(__cuda_atomic_host_backend{}, __VA_ARGS__);))

#define _CCCL_DISPATCH_SCOPED_ATOMIC_BACKEND(_Fn, _Scope, ...)                           \
  NV_DISPATCH_TARGET(NV_IS_DEVICE,                                                       \
                     (return _Fn(__cuda_atomic_device_backend{}, __VA_ARGS__, _Scope);), \
                     NV_IS_HOST,                                                         \
                     (return _Fn(__cuda_atomic_host_backend{}, __VA_ARGS__, __thread_scope_tag{});))

_CCCL_HOST_DEVICE_API inline void __atomic_thread_fence_dispatch(memory_order __order)
{
  _CCCL_DISPATCH_SCOPED_ATOMIC_BACKEND(__cuda_atomic_thread_fence, __thread_scope_system_tag{}, __order);
}

_CCCL_HOST_DEVICE_API inline void __atomic_signal_fence_dispatch(memory_order __order)
{
  _CCCL_DISPATCH_ATOMIC_BACKEND(__cuda_atomic_signal_fence, __order);
}

template <typename _Sto, typename _Up, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE_API void __atomic_init_dispatch(_Sto* __a, _Up __val)
{
  __atomic_assign_volatile(__a->get(), __val);
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE_API void __atomic_store_dispatch(_Sto* __a, _Up __val, memory_order __order, _Sco = {})
{
  _CCCL_DISPATCH_SCOPED_ATOMIC_BACKEND(__cuda_atomic_store_dispatch, _Sco{}, __a->get(), __val, __order);
}

template <typename _Sto, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE_API auto __atomic_load_dispatch(const _Sto* __a, memory_order __order, _Sco = {})
  -> __atomic_underlying_remove_cv_t<_Sto>
{
  _CCCL_DISPATCH_SCOPED_ATOMIC_BACKEND(__cuda_atomic_load_dispatch, _Sco{}, __a->get(), __order);
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE_API auto __atomic_exchange_dispatch(_Sto* __a, _Up __value, memory_order __order, _Sco = {})
  -> __atomic_underlying_remove_cv_t<_Sto>
{
  _CCCL_DISPATCH_SCOPED_ATOMIC_BACKEND(__cuda_atomic_exchange_dispatch, _Sco{}, __a->get(), __value, __order);
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE_API bool __atomic_compare_exchange_strong_dispatch(
  _Sto* __a, _Up* __expected, _Up __val, memory_order __success, memory_order __failure, _Sco = {})
{
  _CCCL_DISPATCH_SCOPED_ATOMIC_BACKEND(
    __cuda_atomic_compare_exchange_dispatch,
    _Sco{},
    __a->get(),
    __expected,
    __val,
    __cuda_atomic_cas_strong{},
    __success,
    __failure);
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE_API bool __atomic_compare_exchange_weak_dispatch(
  _Sto* __a, _Up* __expected, _Up __val, memory_order __success, memory_order __failure, _Sco = {})
{
  _CCCL_DISPATCH_SCOPED_ATOMIC_BACKEND(
    __cuda_atomic_compare_exchange_dispatch,
    _Sco{},
    __a->get(),
    __expected,
    __val,
    __cuda_atomic_cas_weak{},
    __success,
    __failure);
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE_API auto __atomic_fetch_add_dispatch(_Sto* __a, _Up __delta, memory_order __order, _Sco = {})
  -> __atomic_underlying_remove_cv_t<_Sto>
{
  _CCCL_DISPATCH_SCOPED_ATOMIC_BACKEND(__cuda_atomic_fetch_add_dispatch, _Sco{}, __a->get(), __delta, __order);
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE_API auto __atomic_fetch_sub_dispatch(_Sto* __a, _Up __delta, memory_order __order, _Sco = {})
  -> __atomic_underlying_remove_cv_t<_Sto>
{
  _CCCL_DISPATCH_SCOPED_ATOMIC_BACKEND(__cuda_atomic_fetch_sub_dispatch, _Sco{}, __a->get(), __delta, __order);
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE_API auto __atomic_fetch_and_dispatch(_Sto* __a, _Up __pattern, memory_order __order, _Sco = {})
  -> __atomic_underlying_remove_cv_t<_Sto>
{
  _CCCL_DISPATCH_SCOPED_ATOMIC_BACKEND(__cuda_atomic_fetch_and_dispatch, _Sco{}, __a->get(), __pattern, __order);
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE_API auto __atomic_fetch_or_dispatch(_Sto* __a, _Up __pattern, memory_order __order, _Sco = {})
  -> __atomic_underlying_remove_cv_t<_Sto>
{
  _CCCL_DISPATCH_SCOPED_ATOMIC_BACKEND(__cuda_atomic_fetch_or_dispatch, _Sco{}, __a->get(), __pattern, __order);
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE_API auto __atomic_fetch_xor_dispatch(_Sto* __a, _Up __pattern, memory_order __order, _Sco = {})
  -> __atomic_underlying_remove_cv_t<_Sto>
{
  _CCCL_DISPATCH_SCOPED_ATOMIC_BACKEND(__cuda_atomic_fetch_xor_dispatch, _Sco{}, __a->get(), __pattern, __order);
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE_API auto __atomic_fetch_max_dispatch(_Sto* __a, _Up __val, memory_order __order, _Sco = {})
  -> __atomic_underlying_remove_cv_t<_Sto>
{
  _CCCL_DISPATCH_SCOPED_ATOMIC_BACKEND(__cuda_atomic_fetch_max_dispatch, _Sco{}, __a->get(), __val, __order);
}

template <typename _Sto, typename _Up, typename _Sco, __atomic_storage_is_base<_Sto> = 0>
_CCCL_HOST_DEVICE_API auto __atomic_fetch_min_dispatch(_Sto* __a, _Up __val, memory_order __order, _Sco = {})
  -> __atomic_underlying_remove_cv_t<_Sto>
{
  _CCCL_DISPATCH_SCOPED_ATOMIC_BACKEND(__cuda_atomic_fetch_min_dispatch, _Sco{}, __a->get(), __val, __order);
}

#undef _CCCL_DISPATCH_SCOPED_ATOMIC_BACKEND
#undef _CCCL_DISPATCH_ATOMIC_BACKEND

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ATOMIC_TYPES_BASE_H
