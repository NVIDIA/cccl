// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ATOMIC_ATOMIC_H
#define _CUDA___ATOMIC_ATOMIC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/copy_cv.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_volatile.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/atomic>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// atomic<T>

template <class _Tp, thread_scope _Sco = thread_scope::thread_scope_system>
struct atomic : public ::cuda::std::__atomic_impl<_Tp, _Sco>
{
  static_assert(::cuda::std::is_same_v<_Tp, ::cuda::std::remove_cv_t<_Tp>>,
                "cuda::atomic<T> requires T to be cv-unqualified (P3323R1)");

  using value_type = _Tp;

  _CCCL_HIDE_FROM_ABI constexpr atomic() noexcept = default;

  _CCCL_HOST_DEVICE_API constexpr atomic(_Tp __d) noexcept
      : ::cuda::std::__atomic_impl<_Tp, _Sco>(__d)
  {}

  atomic(const atomic&)                     = delete;
  atomic& operator=(const atomic&)          = delete;
  atomic& operator=(const atomic&) volatile = delete;

  _CCCL_HOST_DEVICE_API inline _Tp operator=(_Tp __d) volatile noexcept
  {
    this->store(__d);
    return __d;
  }
  _CCCL_HOST_DEVICE_API inline _Tp operator=(_Tp __d) noexcept
  {
    this->store(__d);
    return __d;
  }

  _CCCL_HOST_DEVICE_API inline _Tp fetch_max(const _Tp& __op, memory_order __m = memory_order_seq_cst) noexcept
  {
    return ::cuda::std::__atomic_fetch_max_dispatch(&this->__a, __op, __m, ::cuda::std::__scope_to_tag<_Sco>{});
  }
  _CCCL_HOST_DEVICE_API inline _Tp fetch_max(const _Tp& __op, memory_order __m = memory_order_seq_cst) volatile noexcept
  {
    return ::cuda::std::__atomic_fetch_max_dispatch(&this->__a, __op, __m, ::cuda::std::__scope_to_tag<_Sco>{});
  }

  _CCCL_HOST_DEVICE_API inline _Tp fetch_min(const _Tp& __op, memory_order __m = memory_order_seq_cst) noexcept
  {
    return ::cuda::std::__atomic_fetch_min_dispatch(&this->__a, __op, __m, ::cuda::std::__scope_to_tag<_Sco>{});
  }
  _CCCL_HOST_DEVICE_API inline _Tp fetch_min(const _Tp& __op, memory_order __m = memory_order_seq_cst) volatile noexcept
  {
    return ::cuda::std::__atomic_fetch_min_dispatch(&this->__a, __op, __m, ::cuda::std::__scope_to_tag<_Sco>{});
  }
};

// atomic_ref<T>

template <class _Tp, thread_scope _Sco = thread_scope::thread_scope_system>
struct atomic_ref : public ::cuda::std::__atomic_ref_impl<_Tp, _Sco>
{
  using value_type = ::cuda::std::remove_cv_t<_Tp>;

  static constexpr size_t required_alignment = sizeof(_Tp);

  static constexpr bool is_always_lock_free = sizeof(_Tp) <= 8;

  // P3323R1 requires is_always_lock_free for volatile T, but we intentionally keep
  // atomic_ref<volatile __int128> well-formed as an extension.
  static_assert(!::cuda::std::is_volatile_v<_Tp> || is_always_lock_free
                  || ::cuda::std::__atomic_ref_is_volatile_int128_v<_Tp>,
                "cuda::atomic_ref<volatile T> requires T to be always lock-free (P3323R1)");

  _CCCL_HOST_DEVICE_API explicit constexpr atomic_ref(_Tp& __ref)
      : ::cuda::std::__atomic_ref_impl<_Tp, _Sco>(__ref)
  {}

  _CCCL_TEMPLATE(class _T2 = _Tp)
  _CCCL_REQUIRES((!::cuda::std::is_const_v<_T2>) )
  _CCCL_HOST_DEVICE_API inline value_type operator=(value_type __v) const noexcept
  {
    this->store(__v);
    return __v;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr ::cuda::std::__copy_cv_t<_Tp, void>* address() const noexcept
  {
    return this->__a.get();
  }

  _CCCL_HIDE_FROM_ABI atomic_ref(const atomic_ref&) noexcept = default;
  atomic_ref& operator=(const atomic_ref&)                   = delete;
  atomic_ref& operator=(const atomic_ref&) const             = delete;

  _CCCL_TEMPLATE(class _T2 = _Tp)
  _CCCL_REQUIRES((!::cuda::std::is_const_v<_T2>) )
  _CCCL_HOST_DEVICE_API inline value_type
  fetch_max(const value_type& __op, memory_order __m = memory_order_seq_cst) const noexcept
  {
    return ::cuda::std::__atomic_fetch_max_dispatch(&this->__a, __op, __m, ::cuda::std::__scope_to_tag<_Sco>{});
  }

  _CCCL_TEMPLATE(class _T2 = _Tp)
  _CCCL_REQUIRES((!::cuda::std::is_const_v<_T2>) )
  _CCCL_HOST_DEVICE_API inline value_type
  fetch_min(const value_type& __op, memory_order __m = memory_order_seq_cst) const noexcept
  {
    return ::cuda::std::__atomic_fetch_min_dispatch(&this->__a, __op, __m, ::cuda::std::__scope_to_tag<_Sco>{});
  }
};

_CCCL_HOST_DEVICE_API inline void
atomic_thread_fence(memory_order __m, [[maybe_unused]] thread_scope _Scope = thread_scope::thread_scope_system)
{
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE,
    (switch (_Scope) {
      case thread_scope::thread_scope_system:
        ::cuda::std::__cuda_atomic_thread_fence(
          ::cuda::std::__cuda_atomic_device_backend{}, __m, __thread_scope_system_tag{});
        break;
      case thread_scope::thread_scope_device:
        ::cuda::std::__cuda_atomic_thread_fence(
          ::cuda::std::__cuda_atomic_device_backend{}, __m, __thread_scope_device_tag{});
        break;
      case thread_scope::thread_scope_block:
        ::cuda::std::__cuda_atomic_thread_fence(
          ::cuda::std::__cuda_atomic_device_backend{}, __m, __thread_scope_block_tag{});
        break;
      // Atomics scoped to themselves do not require fencing
      case thread_scope::thread_scope_thread:
        break;
    }),
    NV_IS_HOST,
    (::cuda::std::atomic_thread_fence(__m);))
}

_CCCL_HOST_DEVICE_API inline void atomic_signal_fence(memory_order __m)
{
  ::cuda::std::atomic_signal_fence(__m);
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ATOMIC_ATOMIC_H
