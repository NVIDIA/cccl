// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ATOMIC_STORAGE_LOCKED_H
#define _LIBCUDACXX___ATOMIC_STORAGE_LOCKED_H

#include <cuda/std/detail/__config>

#include <cuda/std/type_traits>

#include <cuda/std/__atomic/storage/base.h>
#include <cuda/std/__atomic/order.h>
#include <cuda/std/__atomic/scopes.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

// Locked atomics must override the dispatch to be able to implement RMW primitives around the embedded lock.
struct __atomic_locked_tag {};

template<typename _Tp>
struct __atomic_locked_storage {
  using __underlying_t = typename remove_cv<_Tp>::type;
  using __tag_t = typename __atomic_locked_tag;

  _LIBCUDACXX_HOST_DEVICE
  __atomic_locked_storage() noexcept
    : __a_value(), __a_lock(0) {}
  _LIBCUDACXX_HOST_DEVICE constexpr explicit
  __atomic_locked_storage(_Tp value) noexcept
    : __a_value(value), __a_lock(0) {}

  _Tp __a_value;
  mutable __atomic_storage<_LIBCUDACXX_ATOMIC_FLAG_TYPE> __a_lock;

  template <typename _Sco>
  _LIBCUDACXX_HOST_DEVICE void __lock(_Sco) const volatile {
    while(1 == __atomic_exchange_dispatch(__a_lock, _LIBCUDACXX_ATOMIC_FLAG_TYPE(true), memory_order_acquire, _Sco{}))
        /*spin*/;
  }
  template <typename _Sco>
  _LIBCUDACXX_HOST_DEVICE void __lock(_Sco) const {
    while(1 == __atomic_exchange_dispatch(__a_lock, _LIBCUDACXX_ATOMIC_FLAG_TYPE(true), memory_order_acquire, _Sco{}))
        /*spin*/;
  }
  template <typename _Sco>
  _LIBCUDACXX_HOST_DEVICE void __unlock(_Sco) const volatile {
    __atomic_store_dispatch(__a_lock, _LIBCUDACXX_ATOMIC_FLAG_TYPE(false), memory_order_release, _Sco{});
  }
  template <typename _Sco>
  _LIBCUDACXX_HOST_DEVICE void __unlock(_Sco) const {
    __atomic_store_dispatch(__a_lock, _LIBCUDACXX_ATOMIC_FLAG_TYPE(false), memory_order_release, _Sco{});
  }
};

template <typename _Tp, typename _Sco>
_LIBCUDACXX_HOST_DEVICE
void __atomic_init_dispatch(_Tp& __a,  __atomic_underlying_t<_Tp> __val, _Sco, __atomic_locked_tag) {
  __atomic_assign_volatile(__a.__a_value, __val);
}

template <typename _Tp, typename _Sco>
_LIBCUDACXX_HOST_DEVICE
void __atomic_store_dispatch(_Tp& __a,  __atomic_underlying_t<_Tp> __val, memory_order, _Sco, __atomic_locked_tag) {
  __a.__lock(_Sco{});
  __atomic_assign_volatile(__a.__a_value, __val);
  __a.__unlock(_Sco{});
}

template <typename _Tp, typename _Sco>
_LIBCUDACXX_HOST_DEVICE
__atomic_underlying_t<_Tp> __atomic_load_dispatch(const _Tp& __a, memory_order, _Sco, __atomic_locked_tag) {
  __atomic_underlying_t<_Tp> __old;
  __a.__lock(_Sco{});
  __atomic_assign_volatile(__old, __a.__a_value);
  __a.__unlock(_Sco{});
  return __old;
}

template <typename _Tp, typename _Sco>
_LIBCUDACXX_HOST_DEVICE
__atomic_underlying_t<_Tp> __atomic_exchange_dispatch(_Tp& __a, __atomic_underlying_t<_Tp> __value, memory_order, _Sco, __atomic_locked_tag) {
  __atomic_underlying_t<_Tp> __old;
  __a.__lock(_Sco{});
  __atomic_assign_volatile(__old, __a.__a_value);
  __atomic_assign_volatile(__a.__a_value, __value);
  __a.__unlock(_Sco{});
  return __old;
}

template <typename _Tp, typename _Sco>
_LIBCUDACXX_HOST_DEVICE
bool __atomic_compare_exchange_strong_dispatch(_Tp& __a,
                                          __atomic_underlying_t<_Tp>* __expected, __atomic_underlying_t<_Tp> __value, memory_order, memory_order, _Sco, __atomic_locked_tag) {
  __atomic_underlying_t<_Tp> __temp;
  __a.__lock(_Sco{});
  __atomic_assign_volatile(__temp, __a.__a_value);
  bool __ret = __temp == *__expected;
  if(__ret)
    __atomic_assign_volatile(__a.__a_value, __value);
  else
    __atomic_assign_volatile(*__expected, __a.__a_value);
  __a.__unlock(_Sco{});
  return __ret;
}

template <typename _Tp, typename _Sco>
_LIBCUDACXX_HOST_DEVICE
bool __atomic_compare_exchange_weak_dispatch(_Tp& __a,
                                        __atomic_underlying_t<_Tp>* __expected, __atomic_underlying_t<_Tp> __value, memory_order, memory_order, _Sco, __atomic_locked_tag) {
  __atomic_underlying_t<_Tp> __temp;
  __a.__lock(_Sco{});
  __atomic_assign_volatile(__temp, __a.__a_value);
  bool __ret = __temp == *__expected;
  if(__ret)
    __atomic_assign_volatile(__a.__a_value, __value);
  else
    __atomic_assign_volatile(*__expected, __a.__a_value);
  __a.__unlock(_Sco{});
  return __ret;
}

template <typename _Tp, typename _Td, typename _Sco>
_LIBCUDACXX_HOST_DEVICE
__atomic_underlying_t<_Tp> __atomic_fetch_add_dispatch(_Tp& __a,
                           _Td __delta, memory_order, _Sco, __atomic_locked_tag) {
  __atomic_underlying_t<_Tp> __old;
  __a.__lock(_Sco{});
  __atomic_assign_volatile(__old, __a.__a_value);
  __atomic_assign_volatile(__a.__a_value, __atomic_underlying_t<_Tp>(__old + __delta));
  __a.__unlock(_Sco{});
  return __old;
}

template <typename _Tp, typename _Sco>
_LIBCUDACXX_HOST_DEVICE
__atomic_underlying_t<_Tp> __atomic_fetch_add_dispatch(_Tp& __a,
                           ptrdiff_t __delta, memory_order, _Sco, __atomic_locked_tag) {
  __atomic_underlying_t<_Tp> __old;
  __a.__lock(_Sco{});
  __atomic_assign_volatile(__old, __a.__a_value);
  __atomic_assign_volatile(__a.__a_value, __old + __delta);
  __a.__unlock(_Sco{});
  return __old;
}

template <typename _Tp, typename _Td, typename _Sco>
_LIBCUDACXX_HOST_DEVICE
__atomic_underlying_t<_Tp> __atomic_fetch_sub_dispatch(_Tp& __a,
                           __atomic_underlying_t<_Tp> __delta, memory_order, _Sco, __atomic_locked_tag) {
  __atomic_underlying_t<_Tp> __old;
  __a.__lock(_Sco{});
  __atomic_assign_volatile(__old, __a.__a_value);
  __atomic_assign_volatile(__a.__a_value, __atomic_underlying_t<_Tp>(__old - __delta));
  __a.__unlock(_Sco{});
  return __old;
}

template <typename _Tp, typename _Sco>
_LIBCUDACXX_HOST_DEVICE
__atomic_underlying_t<_Tp> __atomic_fetch_and_dispatch(_Tp& __a,
                           __atomic_underlying_t<_Tp> __pattern, memory_order, _Sco, __atomic_locked_tag) {
  __atomic_underlying_t<_Tp> __old;
  __a.__lock(_Sco{});
  __atomic_assign_volatile(__old, __a.__a_value);
  __atomic_assign_volatile(__a.__a_value, __atomic_underlying_t<_Tp>(__old & __pattern));
  __a.__unlock(_Sco{});
  return __old;
}

template <typename _Tp, typename _Sco>
_LIBCUDACXX_HOST_DEVICE
__atomic_underlying_t<_Tp> __atomic_fetch_or_dispatch(_Tp& __a,
                          __atomic_underlying_t<_Tp> __pattern, memory_order, _Sco, __atomic_locked_tag) {
  __atomic_underlying_t<_Tp> __old;
  __a.__lock(_Sco{});
  __atomic_assign_volatile(__old, __a.__a_value);
  __atomic_assign_volatile(__a.__a_value, __atomic_underlying_t<_Tp>(__old | __pattern));
  __a.__unlock(_Sco{});
  return __old;
}

template <typename _Tp, typename _Sco>
_LIBCUDACXX_HOST_DEVICE
__atomic_underlying_t<_Tp> __atomic_fetch_xor_dispatch(_Tp& __a,
                           __atomic_underlying_t<_Tp> __pattern, memory_order, _Sco, __atomic_locked_tag) {
  __atomic_underlying_t<_Tp> __old;
  __a.__lock(_Sco{});
  __atomic_assign_volatile(__old, __a.__a_value);
  __atomic_assign_volatile(__a.__a_value, _Tp(__old ^ __pattern));
  __a.__unlock(_Sco{});
  return __old;
}

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ATOMIC_STORAGE_LOCKED_H
