//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ATOMICS_FUNCTIONS_HOST_H
#define _LIBCUDACXX___ATOMICS_FUNCTIONS_HOST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/order.h>
#include <cuda/std/__atomic/platform.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/remove_cvref.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Watomic-alignment")

#if !defined(_CCCL_COMPILER_NVRTC)

template <typename _Tp>
struct __atomic_alignment_wrapper
{
  _CCCL_ALIGNAS(sizeof(_Tp)) _Tp __atom;
};

template <typename _Tp>
__atomic_alignment_wrapper<__remove_cv_t<_Tp>>& __atomic_auto_align(_Tp* __a)
{
  using __aligned_t = __atomic_alignment_wrapper<__remove_cv_t<_Tp>>;
  return *reinterpret_cast<__aligned_t*>(__a);
};
template <typename _Tp>
const __atomic_alignment_wrapper<__remove_cv_t<_Tp>>& __atomic_auto_align(const _Tp* __a)
{
  using __aligned_t = const __atomic_alignment_wrapper<__remove_cv_t<_Tp>>;
  return *reinterpret_cast<__aligned_t*>(__a);
};
template <typename _Tp>
volatile __atomic_alignment_wrapper<__remove_cv_t<_Tp>>& __atomic_auto_align(volatile _Tp* __a)
{
  using __aligned_t = volatile __atomic_alignment_wrapper<__remove_cv_t<_Tp>>;
  return *reinterpret_cast<__aligned_t*>(__a);
};
template <typename _Tp>
const volatile __atomic_alignment_wrapper<__remove_cv_t<_Tp>>& __atomic_auto_align(const volatile _Tp* __a)
{
  using __aligned_t = const volatile __atomic_alignment_wrapper<__remove_cv_t<_Tp>>;
  return *reinterpret_cast<__aligned_t*>(__a);
};

// Guard ifdef for lock free query in case it is assigned elsewhere (MSVC/CUDA)
inline void __atomic_thread_fence_host(memory_order __order)
{
  __atomic_thread_fence(__atomic_order_to_int(__order));
}

inline void __atomic_signal_fence_host(memory_order __order)
{
  __atomic_signal_fence(__atomic_order_to_int(__order));
}

template <typename _Tp, typename _Up>
inline void __atomic_store_host(_Tp* __a, _Up __val, memory_order __order)
{
  __atomic_store(
    &__atomic_auto_align<_Tp>(__a), &__atomic_auto_align<__remove_cv_t<_Tp>>(&__val), __atomic_order_to_int(__order));
}

template <typename _Tp>
inline auto __atomic_load_host(_Tp* __a, memory_order __order) -> __remove_cv_t<_Tp>
{
  __remove_cv_t<_Tp> __ret;
  __atomic_load(
    &__atomic_auto_align<_Tp>(__a), &__atomic_auto_align<__remove_cv_t<_Tp>>(&__ret), __atomic_order_to_int(__order));
  return __ret;
}

template <typename _Tp, typename _Up>
inline auto __atomic_exchange_host(_Tp* __a, _Up __val, memory_order __order) -> __remove_cv_t<_Tp>
{
  __remove_cv_t<_Tp> __ret;
  __atomic_exchange(&__atomic_auto_align<_Tp>(__a),
                    &__atomic_auto_align<__remove_cv_t<_Tp>>(&__val),
                    &__atomic_auto_align<__remove_cv_t<_Tp>>(&__ret),
                    __atomic_order_to_int(__order));
  return __ret;
}

template <typename _Tp, typename _Up>
inline bool __atomic_compare_exchange_strong_host(
  _Tp* __a, _Up* __expected, _Up __value, memory_order __success, memory_order __failure)
{
  return __atomic_compare_exchange(
    &__atomic_auto_align<_Tp>(__a),
    &__atomic_auto_align<__remove_cv_t<_Tp>>(__expected),
    &__atomic_auto_align<__remove_cv_t<_Tp>>(&__value),
    false,
    __atomic_order_to_int(__success),
    __atomic_failure_order_to_int(__failure));
}

template <typename _Tp, typename _Up>
inline bool __atomic_compare_exchange_weak_host(
  _Tp* __a, _Up* __expected, _Up __value, memory_order __success, memory_order __failure)
{
  return __atomic_compare_exchange(
    &__atomic_auto_align<_Tp>(__a),
    &__atomic_auto_align<__remove_cv_t<_Tp>>(__expected),
    &__atomic_auto_align<__remove_cv_t<_Tp>>(&__value),
    true,
    __atomic_order_to_int(__success),
    __atomic_failure_order_to_int(__failure));
}

template <typename _Tp>
struct __atomic_ptr_skip
{
  static constexpr auto __skip = 1;
};

template <typename _Tp>
struct __atomic_ptr_skip<_Tp*>
{
  static constexpr auto __skip = sizeof(_Tp);
};

// FIXME: Haven't figured out what the spec says about using arrays with
// atomic_fetch_add. Force a failure rather than creating bad behavior.
template <typename _Tp>
struct __atomic_ptr_skip<_Tp[]>
{};
template <typename _Tp, int n>
struct __atomic_ptr_skip<_Tp[n]>
{};

template <typename _Tp>
using __atomic_ptr_skip_t = __atomic_ptr_skip<__remove_cvref_t<_Tp>>;

template <typename _Tp, typename _Td, __enable_if_t<!is_floating_point<_Tp>::value, int> = 0>
inline __remove_cv_t<_Tp> __atomic_fetch_add_host(_Tp* __a, _Td __delta, memory_order __order)
{
  constexpr auto __skip_v = __atomic_ptr_skip_t<_Tp>::__skip;
  return __atomic_fetch_add(__a, __delta * __skip_v, __atomic_order_to_int(__order));
}

template <typename _Tp, typename _Td, __enable_if_t<is_floating_point<_Tp>::value, int> = 0>
inline __remove_cv_t<_Tp> __atomic_fetch_add_host(_Tp* __a, _Td __delta, memory_order __order)
{
  auto __expected = __atomic_load_host(__a, memory_order_relaxed);
  auto __desired  = __expected + __delta;

  while (!__atomic_compare_exchange_strong_host(__a, &__expected, __desired, __order, __order))
  {
    __desired = __expected + __delta;
  }

  return __expected;
}

template <typename _Tp, typename _Td, __enable_if_t<!is_floating_point<_Tp>::value, int> = 0>
inline __remove_cv_t<_Tp> __atomic_fetch_sub_host(_Tp* __a, _Td __delta, memory_order __order)
{
  constexpr auto __skip_v = __atomic_ptr_skip_t<_Tp>::__skip;
  return __atomic_fetch_sub(__a, __delta * __skip_v, __atomic_order_to_int(__order));
}

template <typename _Tp, typename _Td, __enable_if_t<is_floating_point<_Tp>::value, int> = 0>
inline __remove_cv_t<_Tp> __atomic_fetch_sub_host(_Tp* __a, _Td __delta, memory_order __order)
{
  auto __expected = __atomic_load_host(__a, memory_order_relaxed);
  auto __desired  = __expected - __delta;

  while (!__atomic_compare_exchange_strong_host(__a, &__expected, __desired, __order, __order))
  {
    __desired = __expected - __delta;
  }

  return __expected;
}

template <typename _Tp, typename _Td>
inline __remove_cv_t<_Tp> __atomic_fetch_and_host(_Tp* __a, _Td __pattern, memory_order __order)
{
  return __atomic_fetch_and(__a, __pattern, __atomic_order_to_int(__order));
}

template <typename _Tp, typename _Td>
inline __remove_cv_t<_Tp> __atomic_fetch_or_host(_Tp* __a, _Td __pattern, memory_order __order)
{
  return __atomic_fetch_or(__a, __pattern, __atomic_order_to_int(__order));
}

template <typename _Tp, typename _Td>
inline __remove_cv_t<_Tp> __atomic_fetch_xor_host(_Tp* __a, _Td __pattern, memory_order __order)
{
  return __atomic_fetch_xor(__a, __pattern, __atomic_order_to_int(__order));
}

template <typename _Tp, typename _Td>
inline __remove_cv_t<_Tp> __atomic_fetch_max_host(_Tp* __a, _Td __val, memory_order __order)
{
  auto __expected = __atomic_load_host(__a, memory_order_relaxed);
  auto __desired  = __expected > __val ? __expected : __val;

  while (__desired == __val && !__atomic_compare_exchange_strong_host(__a, &__expected, __desired, __order, __order))
  {
    __desired = __expected > __val ? __expected : __val;
  }

  return __expected;
}

template <typename _Tp, typename _Td>
inline __remove_cv_t<_Tp> __atomic_fetch_min_host(_Tp* __a, _Td __val, memory_order __order)
{
  auto __expected = __atomic_load_host(__a, memory_order_relaxed);
  auto __desired  = __expected < __val ? __expected : __val;

  while (__desired == __val && !__atomic_compare_exchange_strong_host(__a, &__expected, __desired, __order, __order))
  {
    __desired = __expected < __val ? __expected : __val;
  }

  return __expected;
}

#endif // !defined(_CCCL_COMPILER_NVRTC)

_CCCL_DIAG_POP

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ATOMICS_FUNCTIONS_HOST_H
