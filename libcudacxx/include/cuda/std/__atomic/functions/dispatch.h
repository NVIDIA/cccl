//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ATOMIC_FUNCTIONS_DISPATCH_H
#define _CUDA_STD___ATOMIC_FUNCTIONS_DISPATCH_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/functions/common.h>
#include <cuda/std/__atomic/functions/cuda_local.h>
#include <cuda/std/__atomic/functions/cuda_ptx.h>
#include <cuda/std/__atomic/functions/generic.h>
#include <cuda/std/__atomic/functions/host.h>
#include <cuda/std/__type_traits/copy_cv.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/cassert>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Backend, class _Fn, class _Sco, class... _Args>
_CCCL_HOST_DEVICE_API void __cuda_atomic_load_order_dispatch(
  [[maybe_unused]] _Backend __backend, _Fn& __fn, memory_order __order, _Sco __scope, _Args... __args)
{
  if constexpr (!_Backend::__needs_constant_order)
  {
    __fn(__order, __args..., __scope);
  }
  else
  {
    switch (__atomic_order_to_int(__order))
    {
      case __ATOMIC_RELAXED:
        return __backend.__with_transformed_order(
          __cuda_atomic_operation_load{}, __fn, __cuda_atomic_order_relaxed{}, __scope, __args...);
      case __ATOMIC_CONSUME:
        [[fallthrough]];
      case __ATOMIC_ACQUIRE:
        return __backend.__with_transformed_order(
          __cuda_atomic_operation_load{}, __fn, __cuda_atomic_order_acquire{}, __scope, __args...);
      case __ATOMIC_SEQ_CST:
        return __backend.__with_transformed_order(
          __cuda_atomic_operation_load{}, __fn, __cuda_atomic_order_seq_cst{}, __scope, __args...);
      default:
        _CCCL_ASSERT(false, "invalid load memory order");
    }
  }
}

template <class _Backend, class _Fn, class _Sco, class... _Args>
_CCCL_HOST_DEVICE_API void __cuda_atomic_store_order_dispatch(
  [[maybe_unused]] _Backend __backend, _Fn& __fn, memory_order __order, _Sco __scope, _Args... __args)
{
  if constexpr (!_Backend::__needs_constant_order)
  {
    __fn(__order, __args..., __scope);
  }
  else
  {
    switch (__atomic_order_to_int(__order))
    {
      case __ATOMIC_RELAXED:
        return __backend.__with_transformed_order(
          __cuda_atomic_operation_store{}, __fn, __cuda_atomic_order_relaxed{}, __scope, __args...);
      case __ATOMIC_RELEASE:
        return __backend.__with_transformed_order(
          __cuda_atomic_operation_store{}, __fn, __cuda_atomic_order_release{}, __scope, __args...);
      case __ATOMIC_SEQ_CST:
        return __backend.__with_transformed_order(
          __cuda_atomic_operation_store{}, __fn, __cuda_atomic_order_seq_cst{}, __scope, __args...);
      default:
        _CCCL_ASSERT(false, "invalid store memory order");
    }
  }
}

template <class _Backend, class _Fn, class _Sco, class... _Args>
_CCCL_HOST_DEVICE_API void __cuda_atomic_rmw_order_dispatch(
  [[maybe_unused]] _Backend __backend, _Fn& __fn, memory_order __order, _Sco __scope, _Args... __args)
{
  if constexpr (!_Backend::__needs_constant_order)
  {
    __fn(__order, __args..., __scope);
  }
  else
  {
    switch (__atomic_order_to_int(__order))
    {
      case __ATOMIC_RELAXED:
        return __backend.__with_transformed_order(
          __cuda_atomic_operation_rmw{}, __fn, __cuda_atomic_order_relaxed{}, __scope, __args...);
      case __ATOMIC_CONSUME:
        [[fallthrough]];
      case __ATOMIC_ACQUIRE:
        return __backend.__with_transformed_order(
          __cuda_atomic_operation_rmw{}, __fn, __cuda_atomic_order_acquire{}, __scope, __args...);
      case __ATOMIC_RELEASE:
        return __backend.__with_transformed_order(
          __cuda_atomic_operation_rmw{}, __fn, __cuda_atomic_order_release{}, __scope, __args...);
      case __ATOMIC_ACQ_REL:
        return __backend.__with_transformed_order(
          __cuda_atomic_operation_rmw{}, __fn, __cuda_atomic_order_acq_rel{}, __scope, __args...);
      case __ATOMIC_SEQ_CST:
        return __backend.__with_transformed_order(
          __cuda_atomic_operation_rmw{}, __fn, __cuda_atomic_order_seq_cst{}, __scope, __args...);
      default:
        _CCCL_ASSERT(false, "invalid read-modify-write memory order");
    }
  }
}

template <class _Backend, class _Fn, class _Sco, class... _Args>
_CCCL_HOST_DEVICE_API void __cuda_atomic_exchange_order_dispatch(
  _Backend __backend, _Fn& __fn, memory_order __order, _Sco __scope, _Args... __args)
{
  ::cuda::std::__cuda_atomic_rmw_order_dispatch(__backend, __fn, __order, __scope, __args...);
}

template <class _Backend, class _Fn, class _Sco, class... _Args>
_CCCL_HOST_DEVICE_API void
__cuda_atomic_fetch_order_dispatch(_Backend __backend, _Fn& __fn, memory_order __order, _Sco __scope, _Args... __args)
{
  ::cuda::std::__cuda_atomic_rmw_order_dispatch(__backend, __fn, __order, __scope, __args...);
}

template <class _Success, class _Backend, class _Fn, class _Sco, class... _Args>
[[nodiscard]] _CCCL_HOST_DEVICE_API bool __cuda_atomic_compare_exchange_failure_order_dispatch(
  _Backend __backend, _Fn& __fn, int __failure, _Sco __scope, _Args... __args)
{
  switch (__failure)
  {
    case __ATOMIC_RELAXED:
      return __backend.__with_transformed_order(
        __cuda_atomic_operation_rmw{},
        __fn,
        __backend.__collapse_cas_order(__cuda_atomic_cas_order<_Success, __cuda_atomic_order_relaxed>{}),
        __scope,
        __args...);
    case __ATOMIC_CONSUME:
      [[fallthrough]];
    case __ATOMIC_ACQUIRE:
      return __backend.__with_transformed_order(
        __cuda_atomic_operation_rmw{},
        __fn,
        __backend.__collapse_cas_order(__cuda_atomic_cas_order<_Success, __cuda_atomic_order_acquire>{}),
        __scope,
        __args...);
    case __ATOMIC_SEQ_CST:
      return __backend.__with_transformed_order(
        __cuda_atomic_operation_rmw{},
        __fn,
        __backend.__collapse_cas_order(__cuda_atomic_cas_order<_Success, __cuda_atomic_order_seq_cst>{}),
        __scope,
        __args...);
    default:
      _CCCL_ASSERT(false, "invalid compare-exchange failure memory order");
      _CCCL_UNREACHABLE();
  }
}

template <class _Backend, class _Fn, class _Sco, class... _Args>
[[nodiscard]] _CCCL_HOST_DEVICE_API bool __cuda_atomic_compare_exchange_order_dispatch(
  [[maybe_unused]] _Backend __backend,
  _Fn& __fn,
  memory_order __success,
  memory_order __failure,
  _Sco __scope,
  _Args... __args)
{
  if constexpr (!_Backend::__needs_constant_order)
  {
    return __fn(__cuda_atomic_runtime_cas_order{__success, __failure}, __args..., __scope);
  }
  else
  {
    const int __failure_order = __atomic_failure_order_to_int(__failure);
    switch (__atomic_order_to_int(__success))
    {
      case __ATOMIC_RELAXED:
        return __cuda_atomic_compare_exchange_failure_order_dispatch<__cuda_atomic_order_relaxed>(
          __backend, __fn, __failure_order, __scope, __args...);
      case __ATOMIC_CONSUME:
        [[fallthrough]];
      case __ATOMIC_ACQUIRE:
        return __cuda_atomic_compare_exchange_failure_order_dispatch<__cuda_atomic_order_acquire>(
          __backend, __fn, __failure_order, __scope, __args...);
      case __ATOMIC_RELEASE:
        return __cuda_atomic_compare_exchange_failure_order_dispatch<__cuda_atomic_order_release>(
          __backend, __fn, __failure_order, __scope, __args...);
      case __ATOMIC_ACQ_REL:
        return __cuda_atomic_compare_exchange_failure_order_dispatch<__cuda_atomic_order_acq_rel>(
          __backend, __fn, __failure_order, __scope, __args...);
      case __ATOMIC_SEQ_CST:
        return __cuda_atomic_compare_exchange_failure_order_dispatch<__cuda_atomic_order_seq_cst>(
          __backend, __fn, __failure_order, __scope, __args...);
      default:
        _CCCL_ASSERT(false, "invalid compare-exchange success memory order");
        _CCCL_UNREACHABLE();
    }
  }
}

template <class _Backend, class _Type>
struct __cuda_atomic_bind_fetch_sub
{
  _Backend __backend;
  _Type* __ptr;
  __unv<_Type>* __dst;
  __unv<_Type>* __op;

  template <class _Order, class _Operand, class _Sco>
  _CCCL_HOST_DEVICE_API void operator()(_Order __order, _Operand, _Sco)
  {
    ::cuda::std::__cuda_atomic_fetch_sub(__backend, __ptr, *__dst, *__op, __order, _Operand{}, _Sco{});
  }
};

template <class _Backend, class _Type, class _Up, class _Sco>
[[nodiscard]] _CCCL_HOST_DEVICE_API __unv<_Type>
__cuda_atomic_fetch_sub_dispatch(_Backend __backend, _Type* __ptr, _Up __op, memory_order __order, _Sco __scope)
{
  using __value_type    = __unv<_Type>;
  constexpr auto __skip = __atomic_ptr_skip_t<__value_type>::__skip;
  __op                  = __op * __skip;
  using __proxy_type    = __cuda_atomic_deduce_arithmetic_t<__value_type>;
  using __proxy_pointee = __copy_cv_t<_Type, __proxy_type>;
  using __proxy_operand = __cuda_atomic_deduce_arithmetic_tag_t<__value_type>;
  __value_type __dst{};
  auto* __ptr_proxy = reinterpret_cast<__proxy_pointee*>(__ptr);
  auto* __dst_proxy = reinterpret_cast<__proxy_type*>(&__dst);
  auto* __op_proxy  = reinterpret_cast<__proxy_type*>(&__op);
#if _CCCL_CUDA_COMPILATION()
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    if (::cuda::std::__cuda_atomic_fetch_sub_weak_if_local(__ptr_proxy, *__op_proxy, __dst_proxy))
    {
      return __dst;
    }
  }
#endif // _CCCL_CUDA_COMPILATION()
  __cuda_atomic_bind_fetch_sub<_Backend, __proxy_pointee> __bound_fetch_sub{
    __backend, __ptr_proxy, __dst_proxy, __op_proxy};
  ::cuda::std::__cuda_atomic_fetch_order_dispatch(__backend, __bound_fetch_sub, __order, __scope, __proxy_operand{});
  return __dst;
}

#if _CCCL_CUDA_COMPILATION()
_CCCL_DEVICE_API inline void __cuda_atomic_signal_fence(__cuda_atomic_ptx_backend, memory_order)
{
  asm volatile("" ::: "memory");
}
#endif // _CCCL_CUDA_COMPILATION()

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ATOMIC_FUNCTIONS_DISPATCH_H
