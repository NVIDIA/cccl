// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___CUDA_ATOMIC_H
#define _LIBCUDACXX___CUDA_ATOMIC_H

#include <cuda/std/detail/__config>
#include <cuda/std/atomic>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

// atomic<T>

template <class _Tp, thread_scope _Sco = thread_scope::thread_scope_system>
struct atomic
  : public std::__atomic_impl<_Tp, _Sco>
{
  using value_type = _Tp;

  _CCCL_HOST_DEVICE
  constexpr atomic() noexcept
      : std::__atomic_impl<_Tp, _Sco>() {}
  _CCCL_HOST_DEVICE
  constexpr atomic(_Tp __d) noexcept
      : std::__atomic_impl<_Tp, _Sco>(__d) {}

  _CCCL_HOST_DEVICE _Tp operator=(_Tp __d) volatile noexcept
  {
    this->store(__d);
    return __d;
  }
  _CCCL_HOST_DEVICE _Tp operator=(_Tp __d) noexcept
  {
    this->store(__d);
    return __d;
  }

  _CCCL_HOST_DEVICE _Tp fetch_max(const _Tp& __op, memory_order __m = memory_order_seq_cst) volatile noexcept
  {
    return std::__atomic_fetch_max_dispatch(this->__this_atom(), __op, __m, std::__scope_to_tag<_Sco>{});
  }

  _CCCL_HOST_DEVICE _Tp fetch_min(const _Tp& __op, memory_order __m = memory_order_seq_cst) volatile noexcept
  {
    return std::__atomic_fetch_min_dispatch(this->__this_atom(), __op, __m, std::__scope_to_tag<_Sco>{});
  }
};

// atomic<T*>

template <class _Tp, thread_scope _Sco>
struct atomic<_Tp*, _Sco>
  : public std::__atomic_impl<_Tp*, _Sco>
{
  using value_type = _Tp*;

  _CCCL_HOST_DEVICE
  constexpr atomic() noexcept
      : std::__atomic_impl<_Tp*, _Sco>() {}

  _CCCL_HOST_DEVICE
  constexpr atomic(_Tp* __d) noexcept
      : std::__atomic_impl<_Tp*, _Sco>(__d) {}

  _CCCL_HOST_DEVICE _Tp* operator=(_Tp* __d) volatile noexcept
  {
    this->store(__d);
    return __d;
  }
  _CCCL_HOST_DEVICE _Tp* operator=(_Tp* __d) noexcept
  {
    this->store(__d);
    return __d;
  }

  _CCCL_HOST_DEVICE _Tp* fetch_add(ptrdiff_t __op, memory_order __m = memory_order_seq_cst) volatile noexcept
  {
    return std::__atomic_fetch_add_dispatch(this->__this_atom(), __op, __m, std::__scope_to_tag<_Sco>{});
  }
  _CCCL_HOST_DEVICE _Tp* fetch_add(ptrdiff_t __op, memory_order __m = memory_order_seq_cst) noexcept
  {
    return std::__atomic_fetch_add_dispatch(this->__this_atom(), __op, __m, std::__scope_to_tag<_Sco>{});
  }
  _CCCL_HOST_DEVICE _Tp* fetch_sub(ptrdiff_t __op, memory_order __m = memory_order_seq_cst) volatile noexcept
  {
    return std::__atomic_fetch_sub_dispatch(this->__this_atom(), __op, __m, std::__scope_to_tag<_Sco>{});
  }
  _CCCL_HOST_DEVICE _Tp* fetch_sub(ptrdiff_t __op, memory_order __m = memory_order_seq_cst) noexcept
  {
    return std::__atomic_fetch_sub_dispatch(this->__this_atom(), __op, __m, std::__scope_to_tag<_Sco>{});
  }

  _CCCL_HOST_DEVICE _Tp* operator++(int) volatile noexcept
  {
    return fetch_add(1);
  }
  _CCCL_HOST_DEVICE _Tp* operator++(int) noexcept
  {
    return fetch_add(1);
  }
  _CCCL_HOST_DEVICE _Tp* operator--(int) volatile noexcept
  {
    return fetch_sub(1);
  }
  _CCCL_HOST_DEVICE _Tp* operator--(int) noexcept
  {
    return fetch_sub(1);
  }
  _CCCL_HOST_DEVICE _Tp* operator++() volatile noexcept
  {
    return fetch_add(1) + 1;
  }
  _CCCL_HOST_DEVICE _Tp* operator++() noexcept
  {
    return fetch_add(1) + 1;
  }
  _CCCL_HOST_DEVICE _Tp* operator--() volatile noexcept
  {
    return fetch_sub(1) - 1;
  }
  _CCCL_HOST_DEVICE _Tp* operator--() noexcept
  {
    return fetch_sub(1) - 1;
  }
  _CCCL_HOST_DEVICE _Tp* operator+=(ptrdiff_t __op) volatile noexcept
  {
    return fetch_add(__op) + __op;
  }
  _CCCL_HOST_DEVICE _Tp* operator+=(ptrdiff_t __op) noexcept
  {
    return fetch_add(__op) + __op;
  }
  _CCCL_HOST_DEVICE _Tp* operator-=(ptrdiff_t __op) volatile noexcept
  {
    return fetch_sub(__op) - __op;
  }
  _CCCL_HOST_DEVICE _Tp* operator-=(ptrdiff_t __op) noexcept
  {
    return fetch_sub(__op) - __op;
  }
};

// atomic_ref<T>

template <class _Tp, thread_scope _Sco = thread_scope::thread_scope_system>
struct atomic_ref : public std::__atomic_ref_impl<_Tp, _Sco>
{
  typedef std::__atomic_ref_impl<_Tp, _Sco> __base;

  _CCCL_HOST_DEVICE constexpr atomic_ref(_Tp& __d) noexcept
      : __base(__d)
  {}

  _CCCL_HOST_DEVICE _Tp operator=(_Tp __d) const noexcept
  {
    this->store(__d);
    return __d;
  }

  _CCCL_HOST_DEVICE _Tp fetch_max(const _Tp& __op, memory_order __m = memory_order_seq_cst) const noexcept
  {
    return std::__atomic_fetch_max_dispatch(this->__this_atom(), __op, __m, std::__scope_to_tag<_Sco>{});
  }

  _CCCL_HOST_DEVICE _Tp fetch_min(const _Tp& __op, memory_order __m = memory_order_seq_cst) const noexcept
  {
    return std::__atomic_fetch_min_dispatch(this->__this_atom(), __op, __m, std::__scope_to_tag<_Sco>{});
  }
};

// atomic_ref<T*>

template <class _Tp, thread_scope _Sco>
struct atomic_ref<_Tp*, _Sco> : public std::__atomic_ref_impl<_Tp*, _Sco>
{
  typedef std::__atomic_ref_impl<_Tp*, _Sco> __base;

  _CCCL_HOST_DEVICE constexpr atomic_ref(_Tp*& __d) noexcept
      : __base(__d)
  {}

  _CCCL_HOST_DEVICE _Tp* operator=(_Tp* __d) const noexcept
  {
    this->store(__d);
    return __d;
  }

  _CCCL_HOST_DEVICE _Tp* fetch_add(ptrdiff_t __op, memory_order __m = memory_order_seq_cst) const noexcept
  {
    return __atomic_fetch_add_dispatch(this->__this_atom(), __op, __m, std::__scope_to_tag<_Sco>{});
  }
  _CCCL_HOST_DEVICE _Tp* fetch_sub(ptrdiff_t __op, memory_order __m = memory_order_seq_cst) const noexcept
  {
    return __atomic_fetch_sub_dispatch(this->__this_atom(), __op, __m, std::__scope_to_tag<_Sco>{});
  }

  _CCCL_HOST_DEVICE _Tp* operator++(int) const noexcept
  {
    return fetch_add(1);
  }
  _CCCL_HOST_DEVICE _Tp* operator--(int) const noexcept
  {
    return fetch_sub(1);
  }
  _CCCL_HOST_DEVICE _Tp* operator++() const noexcept
  {
    return fetch_add(1) + 1;
  }
  _CCCL_HOST_DEVICE _Tp* operator--() const noexcept
  {
    return fetch_sub(1) - 1;
  }
  _CCCL_HOST_DEVICE _Tp* operator+=(ptrdiff_t __op) const noexcept
  {
    return fetch_add(__op) + __op;
  }
  _CCCL_HOST_DEVICE _Tp* operator-=(ptrdiff_t __op) const noexcept
  {
    return fetch_sub(__op) - __op;
  }
};

inline _CCCL_HOST_DEVICE void
atomic_thread_fence(memory_order __m, thread_scope _Scope = thread_scope::thread_scope_system)
{
  NV_DISPATCH_TARGET(
    NV_IS_DEVICE,
    (switch (_Scope) {
      case thread_scope::thread_scope_system:
        std::__atomic_thread_fence_cuda((int) __m, __thread_scope_system_tag{});
        break;
      case thread_scope::thread_scope_device:
        std::__atomic_thread_fence_cuda((int) __m, __thread_scope_device_tag{});
        break;
      case thread_scope::thread_scope_block:
        std::__atomic_thread_fence_cuda((int) __m, __thread_scope_block_tag{});
        break;
      // Atomics scoped to themselves do not require fencing
      case thread_scope::thread_scope_thread:
        break;
    }),
    NV_IS_HOST,
    ((void) _Scope; std::atomic_thread_fence(__m);))
}

inline _CCCL_HOST_DEVICE void atomic_signal_fence(memory_order __m)
{
  std::atomic_signal_fence(__m);
}

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _LIBCUDACXX___CUDA_ATOMIC_H
