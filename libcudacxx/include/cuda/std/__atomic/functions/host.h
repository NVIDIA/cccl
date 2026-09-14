//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ATOMICS_FUNCTIONS_HOST_H
#define _CUDA_STD___ATOMICS_FUNCTIONS_HOST_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/functions/backend.h>
#include <cuda/std/__atomic/functions/common.h>
#include <cuda/std/__atomic/functions/host_backend.h>
#include <cuda/std/__atomic/order.h>
#include <cuda/std/__atomic/platform.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_floating_point.h>
#include <cuda/std/__type_traits/remove_cv.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_CLANG("-Watomic-alignment")

#if !_CCCL_COMPILER(NVRTC)

// The compiler can provide 128b atomic support. Some onus on user to guarantee support.
#  if _CCCL_HOST_128_ATOMICS_ENABLED()
#    define _LIBCUDACXX_INT128_WARN(TYPE)
// The compiler supports 128b via libatomic or another API.
#  elif _CCCL_HOST_128_ATOMICS_MAYBE()
#    define _LIBCUDACXX_INT128_WARN(TYPE)                                                                        \
      static_assert(                                                                                             \
        sizeof(TYPE) < 16,                                                                                       \
        "CCCL has detected possible support for 128 bit atomics. However this feature is experimental. You can " \
        "define CCCL_ENABLE_EXPERIMENTAL_HOST_ATOMICS_128B to ignore and acknowledge that runtime corruption "   \
        "may occur if you link with libatomic and use locked atomics.");
// The compiler does not provide support or proof of support. eg. msvc
#  else
#    define _LIBCUDACXX_INT128_WARN(TYPE) \
      static_assert(sizeof(TYPE) < 16, "atomic_ref<T> where sizeof(T) > 8 is not supported on this system.");
#  endif

template <typename _Tp>
struct _CCCL_ALIGNAS(sizeof(_Tp)) __atomic_alignment_wrapper
{
  _Tp __atom;
};

template <typename _Tp>
_CCCL_HOST_API __atomic_alignment_wrapper<_Tp>* __atomic_force_align_host(_Tp* __a)
{
  __atomic_alignment_wrapper<_Tp>* __w =
    reinterpret_cast<__atomic_alignment_wrapper<_Tp>*>(const_cast<remove_cv_t<_Tp>*>(__a));
  return __w;
}

// Guard ifdef for lock free query in case it is assigned elsewhere (MSVC/CUDA)
_CCCL_HOST_API inline void
__cuda_atomic_thread_fence(__cuda_atomic_host_backend, memory_order __order, __thread_scope_tag)
{
  __atomic_thread_fence(__atomic_order_to_int(__order));
}

_CCCL_HOST_API inline void __cuda_atomic_signal_fence(__cuda_atomic_host_backend, memory_order __order)
{
  __atomic_signal_fence(__atomic_order_to_int(__order));
}

[[nodiscard]] _CCCL_HOST_API constexpr memory_order __cuda_atomic_failure_order(memory_order __order)
{
  return __order == memory_order_release
         ? memory_order_relaxed
         : (__order == memory_order_acq_rel ? memory_order_acquire : __order);
}

template <class _Type, class _Operand, class _Mmio>
_CCCL_HOST_API void __cuda_atomic_load(
  __cuda_atomic_host_backend,
  const _Type* __ptr,
  __unv<_Type>& __dst,
  memory_order __order,
  _Operand,
  __thread_scope_tag,
  _Mmio)
{
  _LIBCUDACXX_INT128_WARN(_Type)
  __atomic_load(&__atomic_force_align_host(__ptr)->__atom, &__dst, __atomic_order_to_int(__order));
}

template <class _Type, class _Operand, class _Mmio>
_CCCL_HOST_API void __cuda_atomic_store(
  __cuda_atomic_host_backend,
  _Type* __ptr,
  __unv<_Type>& __val,
  memory_order __order,
  _Operand,
  __thread_scope_tag,
  _Mmio)
{
  _LIBCUDACXX_INT128_WARN(_Type)
  __atomic_store(&__atomic_force_align_host(__ptr)->__atom, &__val, __atomic_order_to_int(__order));
}

template <class _Type, class _Cas, class _Operand>
_CCCL_HOST_API bool __cuda_atomic_compare_exchange(
  __cuda_atomic_host_backend,
  _Type* __ptr,
  __unv<_Type>& __dst,
  __unv<_Type> __cmp,
  __unv<_Type> __op,
  _Cas __cas,
  __cuda_atomic_runtime_cas_order __order,
  _Operand,
  __thread_scope_tag)
{
  _LIBCUDACXX_INT128_WARN(_Type)
  __dst = __cmp;
  return __atomic_compare_exchange(
    &__atomic_force_align_host(__ptr)->__atom,
    // This is only alignment wrapped in order to prevent GCC-6 from triggering an unused warning.
    &__atomic_force_align_host(&__dst)->__atom,
    &__op,
    ::cuda::std::__cuda_atomic_cas_is_weak(__cas),
    __atomic_order_to_int(__order.__success),
    __atomic_failure_order_to_int(__order.__failure));
}

template <class _Type, class _Cas, class _Operand>
_CCCL_HOST_API bool __cuda_atomic_compare_exchange(
  __cuda_atomic_host_backend __backend,
  _Type* __ptr,
  __unv<_Type>& __dst,
  __unv<_Type> __cmp,
  __unv<_Type> __op,
  _Cas __cas,
  memory_order __order,
  _Operand __operand,
  __thread_scope_tag __scope)
{
  return ::cuda::std::__cuda_atomic_compare_exchange(
    __backend,
    __ptr,
    __dst,
    __cmp,
    __op,
    __cas,
    __cuda_atomic_runtime_cas_order{__order, ::cuda::std::__cuda_atomic_failure_order(__order)},
    __operand,
    __scope);
}

template <class _Type, class _Operand>
_CCCL_HOST_API void __cuda_atomic_exchange(
  __cuda_atomic_host_backend,
  _Type* __ptr,
  __unv<_Type>& __dst,
  __unv<_Type> __op,
  memory_order __order,
  _Operand,
  __thread_scope_tag)
{
  _LIBCUDACXX_INT128_WARN(_Type)
  __atomic_exchange(&__atomic_force_align_host(__ptr)->__atom, &__op, &__dst, __atomic_order_to_int(__order));
}

template <class _Type,
          class _Operand,
          enable_if_t<!is_floating_point_v<__unv<_Type>> && (_Operand::__op != __cuda_atomic_operand::_f)
                        && (_Operand::__size <= 64),
                      bool> = false>
_CCCL_HOST_API void __cuda_atomic_fetch_add(
  __cuda_atomic_host_backend,
  _Type* __ptr,
  __unv<_Type>& __dst,
  __unv<_Type> __op,
  memory_order __order,
  _Operand,
  __thread_scope_tag)
{
  _LIBCUDACXX_INT128_WARN(_Type)
  __dst = __atomic_fetch_add(__ptr, __op, __atomic_order_to_int(__order));
}

template <class _Type,
          class _Operand,
          enable_if_t<!is_floating_point_v<__unv<_Type>> && (_Operand::__op != __cuda_atomic_operand::_f)
                        && (_Operand::__size <= 64),
                      bool> = false>
_CCCL_HOST_API void __cuda_atomic_fetch_sub(
  __cuda_atomic_host_backend,
  _Type* __ptr,
  __unv<_Type>& __dst,
  __unv<_Type> __op,
  memory_order __order,
  _Operand,
  __thread_scope_tag)
{
  _LIBCUDACXX_INT128_WARN(_Type)
  __dst = __atomic_fetch_sub(__ptr, __op, __atomic_order_to_int(__order));
}

template <class _Type,
          class _Operand,
          enable_if_t<(_Operand::__op == __cuda_atomic_operand::_b) && (_Operand::__size <= 64), int> = 0>
_CCCL_HOST_API void __cuda_atomic_fetch_and(
  __cuda_atomic_host_backend,
  _Type* __ptr,
  __unv<_Type>& __dst,
  __unv<_Type> __op,
  memory_order __order,
  _Operand,
  __thread_scope_tag)
{
  _LIBCUDACXX_INT128_WARN(_Type)
  __dst = __atomic_fetch_and(__ptr, __op, __atomic_order_to_int(__order));
}

template <class _Type,
          class _Operand,
          enable_if_t<(_Operand::__op == __cuda_atomic_operand::_b) && (_Operand::__size <= 64), int> = 0>
_CCCL_HOST_API void __cuda_atomic_fetch_or(
  __cuda_atomic_host_backend,
  _Type* __ptr,
  __unv<_Type>& __dst,
  __unv<_Type> __op,
  memory_order __order,
  _Operand,
  __thread_scope_tag)
{
  _LIBCUDACXX_INT128_WARN(_Type)
  __dst = __atomic_fetch_or(__ptr, __op, __atomic_order_to_int(__order));
}

template <class _Type,
          class _Operand,
          enable_if_t<(_Operand::__op == __cuda_atomic_operand::_b) && (_Operand::__size <= 64), int> = 0>
_CCCL_HOST_API void __cuda_atomic_fetch_xor(
  __cuda_atomic_host_backend,
  _Type* __ptr,
  __unv<_Type>& __dst,
  __unv<_Type> __op,
  memory_order __order,
  _Operand,
  __thread_scope_tag)
{
  _LIBCUDACXX_INT128_WARN(_Type)
  __dst = __atomic_fetch_xor(__ptr, __op, __atomic_order_to_int(__order));
}

#endif // !_CCCL_COMPILER(NVRTC)

_CCCL_DIAG_POP

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ATOMICS_FUNCTIONS_HOST_H
