//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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
#include <cuda/std/__atomic/functions/device_backend.h>
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

template <typename _Backend, typename _Type>
struct __cuda_atomic_bind_load
{
  _Backend __backend;
  const _Type* __ptr;
  __unv<_Type>* __dst;

  template <typename _Atomic_Memorder, typename _Tag, typename _Sco, typename _Mmio>
  _CCCL_HOST_DEVICE_API void operator()(_Atomic_Memorder __order, _Tag, _Mmio, _Sco)
  {
    ::cuda::std::__cuda_atomic_load(__backend, __ptr, *__dst, __order, _Tag{}, _Sco{}, _Mmio{});
  }
};
template <class _Backend, class _Type, class _Sco>
_CCCL_HOST_DEVICE_API void __cuda_atomic_load_dispatch(
  _Backend __backend, const _Type* __ptr, __unv<_Type>& __dst, memory_order __order, _Sco __scope)
{
  using __value_type                 = __unv<_Type>;
  using __proxy_t                    = __cuda_atomic_deduce_bitwise_t<__value_type>;
  using __proxy_pointee              = __copy_cv_t<_Type, __proxy_t>;
  using __proxy_tag                  = __cuda_atomic_deduce_bitwise_tag_t<__value_type>;
  const __proxy_pointee* __ptr_proxy = reinterpret_cast<const __proxy_pointee*>(__ptr);
  __proxy_t* __dst_proxy             = reinterpret_cast<__proxy_t*>(&__dst);
#if _CCCL_CUDA_COMPILATION()
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    if (::cuda::std::__cuda_atomic_load_weak_if_local(__ptr_proxy, __dst_proxy, sizeof(__proxy_t)))
    {
      return;
    }
  }
#endif // _CCCL_CUDA_COMPILATION()
  __cuda_atomic_bind_load<_Backend, __proxy_pointee> __bound_load{__backend, __ptr_proxy, __dst_proxy};
  __cuda_atomic_load_order_dispatch(
    __backend, __bound_load, __order, __scope, __proxy_tag{}, __cuda_atomic_mmio_disable{});
}

template <class _Backend, class _Type, class _Sco>
[[nodiscard]] _CCCL_HOST_DEVICE_API __unv<_Type>
__cuda_atomic_load_dispatch(_Backend __backend, const _Type* __ptr, memory_order __order, _Sco __scope)
{
  __unv<_Type> __dst;
  ::cuda::std::__cuda_atomic_load_dispatch(__backend, __ptr, __dst, __order, __scope);
  return __dst;
}

template <typename _Backend, typename _Type>
struct __cuda_atomic_bind_store
{
  _Backend __backend;
  _Type* __ptr;
  __unv<_Type> __val;

  template <typename _Atomic_Memorder, typename _Tag, typename _Sco, typename _Mmio>
  _CCCL_HOST_DEVICE_API void operator()(_Atomic_Memorder __order, _Tag, _Mmio, _Sco)
  {
    ::cuda::std::__cuda_atomic_store(__backend, __ptr, __val, __order, _Tag{}, _Sco{}, _Mmio{});
  }
};
template <class _Backend, class _Type, class _Up, class _Sco>
_CCCL_HOST_DEVICE_API void
__cuda_atomic_store_dispatch(_Backend __backend, _Type* __ptr, _Up __val, memory_order __order, _Sco __scope)
{
  using __value_type           = __unv<_Type>;
  using __proxy_t              = __cuda_atomic_deduce_bitwise_t<__value_type>;
  using __proxy_pointee        = __copy_cv_t<_Type, __proxy_t>;
  using __proxy_tag            = __cuda_atomic_deduce_bitwise_tag_t<__value_type>;
  __proxy_pointee* __ptr_proxy = reinterpret_cast<__proxy_pointee*>(__ptr);
  __value_type __store         = __val;
  __proxy_t* __val_proxy       = reinterpret_cast<__proxy_t*>(&__store);
#if _CCCL_CUDA_COMPILATION()
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    if (::cuda::std::__cuda_atomic_store_weak_if_local(__ptr_proxy, __val_proxy, sizeof(__proxy_t)))
    {
      return;
    }
  }
#endif // _CCCL_CUDA_COMPILATION()
  __cuda_atomic_bind_store<_Backend, __proxy_pointee> __bound_store{__backend, __ptr_proxy, *__val_proxy};
  __cuda_atomic_store_order_dispatch(
    __backend, __bound_store, __order, __scope, __proxy_tag{}, __cuda_atomic_mmio_disable{});
}

template <typename _Backend, typename _Type>
struct __cuda_atomic_bind_compare_exchange
{
  _Backend __backend;
  _Type* __ptr;
  __unv<_Type>* __exp;
  __unv<_Type> __cmp;
  __unv<_Type> __des;

  template <typename _Atomic_Memorder, typename _Cas, typename _Tag, typename _Sco>
  [[nodiscard]] _CCCL_HOST_DEVICE_API bool operator()(_Atomic_Memorder __order, _Cas, _Tag, _Sco)
  {
    return ::cuda::std::__cuda_atomic_compare_exchange(
      __backend, __ptr, *__exp, __cmp, __des, _Cas{}, __order, _Tag{}, _Sco{});
  }
};
template <class _Backend, class _Type, class _Cas, class _Sco>
[[nodiscard]] _CCCL_HOST_DEVICE_API bool __cuda_atomic_compare_exchange_dispatch(
  _Backend __backend,
  _Type* __ptr,
  __unv<_Type>* __exp,
  __unv<_Type> __des,
  _Cas,
  memory_order __success,
  memory_order __failure,
  _Sco __scope)
{
  using __value_type           = __unv<_Type>;
  using __proxy_t              = __cuda_atomic_deduce_bitwise_t<__value_type>;
  using __proxy_pointee        = __copy_cv_t<_Type, __proxy_t>;
  using __proxy_tag            = __cuda_atomic_deduce_bitwise_tag_t<__value_type>;
  __proxy_pointee* __ptr_proxy = reinterpret_cast<__proxy_pointee*>(__ptr);
  __proxy_t* __exp_proxy       = reinterpret_cast<__proxy_t*>(__exp);
  __proxy_t* __des_proxy       = reinterpret_cast<__proxy_t*>(&__des);
#if _CCCL_CUDA_COMPILATION()
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    bool __res = false;
    if (::cuda::std::__cuda_atomic_compare_exchange_weak_if_local(__ptr_proxy, __exp_proxy, __des_proxy, &__res))
    {
      return __res;
    }
  }
#endif // _CCCL_CUDA_COMPILATION()
  __cuda_atomic_bind_compare_exchange<_Backend, __proxy_pointee> __bound_compare_swap{
    __backend, __ptr_proxy, __exp_proxy, *__exp_proxy, *__des_proxy};
  return __cuda_atomic_compare_exchange_order_dispatch(
    __backend, __bound_compare_swap, __success, __failure, __scope, _Cas{}, __proxy_tag{});
}

template <typename _Backend, typename _Type>
struct __cuda_atomic_bind_exchange
{
  _Backend __backend;
  _Type* __ptr;
  __unv<_Type>* __old;
  __unv<_Type> __new;

  template <typename _Atomic_Memorder, typename _Tag, typename _Sco>
  _CCCL_HOST_DEVICE_API void operator()(_Atomic_Memorder __order, _Tag, _Sco)
  {
    ::cuda::std::__cuda_atomic_exchange(__backend, __ptr, *__old, __new, __order, _Tag{}, _Sco{});
  }
};
template <class _Backend, class _Type, class _Sco>
_CCCL_HOST_DEVICE_API void __cuda_atomic_exchange_dispatch(
  _Backend __backend, _Type* __ptr, __unv<_Type>& __old, __unv<_Type> __new, memory_order __order, _Sco __scope)
{
  using __value_type _CCCL_NODEBUG    = __unv<_Type>;
  using __proxy_t _CCCL_NODEBUG       = __cuda_atomic_deduce_bitwise_t<__value_type>;
  using __proxy_pointee _CCCL_NODEBUG = __copy_cv_t<_Type, __proxy_t>;
  using __proxy_tag _CCCL_NODEBUG     = __cuda_atomic_deduce_bitwise_tag_t<__value_type>;
  __proxy_pointee* __ptr_proxy        = reinterpret_cast<__proxy_pointee*>(__ptr);
  __proxy_t* __old_proxy              = reinterpret_cast<__proxy_t*>(&__old);
  __proxy_t* __new_proxy              = reinterpret_cast<__proxy_t*>(&__new);
#if _CCCL_CUDA_COMPILATION()
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    if (::cuda::std::__cuda_atomic_exchange_weak_if_local(__ptr_proxy, __new_proxy, __old_proxy))
    {
      return;
    }
  }
#endif // _CCCL_CUDA_COMPILATION()
  __cuda_atomic_bind_exchange<_Backend, __proxy_pointee> __bound_swap{__backend, __ptr_proxy, __old_proxy, *__new_proxy};
  __cuda_atomic_exchange_order_dispatch(__backend, __bound_swap, __order, __scope, __proxy_tag{});
}

template <class _Backend, class _Type, class _Up, class _Sco>
[[nodiscard]] _CCCL_HOST_DEVICE_API __unv<_Type>
__cuda_atomic_exchange_dispatch(_Backend __backend, _Type* __ptr, _Up __new, memory_order __order, _Sco __scope)
{
  using __value_type _CCCL_NODEBUG = __unv<_Type>;
  __value_type __old;
  ::cuda::std::__cuda_atomic_exchange_dispatch(
    __backend, __ptr, __old, static_cast<__value_type>(__new), __order, __scope);
  return __old;
}

template <typename _Backend, typename _Type>
struct __cuda_atomic_bind_fetch_add
{
  _Backend __backend;
  _Type* __ptr;
  __unv<_Type>* __dst;
  __unv<_Type> __op;

  template <typename _Atomic_Memorder, typename _Tag, typename _Sco>
  _CCCL_HOST_DEVICE_API void operator()(_Atomic_Memorder __order, _Tag, _Sco)
  {
    ::cuda::std::__cuda_atomic_fetch_add(__backend, __ptr, *__dst, __op, __order, _Tag{}, _Sco{});
  }
};
template <class _Backend, class _Type, class _Up, class _Sco>
[[nodiscard]] _CCCL_HOST_DEVICE_API __unv<_Type>
__cuda_atomic_fetch_add_dispatch(_Backend __backend, _Type* __ptr, _Up __op, memory_order __order, _Sco __scope)
{
  __op                  = __op * __atomic_ptr_skip_t<_Type>::__skip;
  using __value_type    = __unv<_Type>;
  using __proxy_t       = __cuda_atomic_deduce_arithmetic_t<__value_type>;
  using __proxy_pointee = __copy_cv_t<_Type, __proxy_t>;
  using __proxy_tag     = __cuda_atomic_deduce_arithmetic_tag_t<__value_type>;
  __value_type __dst{};
  __proxy_pointee* __ptr_proxy = reinterpret_cast<__proxy_pointee*>(__ptr);
  __proxy_t* __dst_proxy       = reinterpret_cast<__proxy_t*>(&__dst);
  __proxy_t* __op_proxy        = reinterpret_cast<__proxy_t*>(&__op);
#if _CCCL_CUDA_COMPILATION()
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    if (::cuda::std::__cuda_atomic_fetch_add_weak_if_local(__ptr_proxy, *__op_proxy, __dst_proxy))
    {
      return __dst;
    }
  }
#endif // _CCCL_CUDA_COMPILATION()
  __cuda_atomic_bind_fetch_add<_Backend, __proxy_pointee> __bound_add{__backend, __ptr_proxy, __dst_proxy, *__op_proxy};
  __cuda_atomic_fetch_order_dispatch(__backend, __bound_add, __order, __scope, __proxy_tag{});
  return __dst;
}

template <typename _Backend, typename _Type>
struct __cuda_atomic_bind_fetch_and
{
  _Backend __backend;
  _Type* __ptr;
  __unv<_Type>* __dst;
  __unv<_Type> __op;

  template <typename _Atomic_Memorder, typename _Tag, typename _Sco>
  _CCCL_HOST_DEVICE_API void operator()(_Atomic_Memorder __order, _Tag, _Sco)
  {
    ::cuda::std::__cuda_atomic_fetch_and(__backend, __ptr, *__dst, __op, __order, _Tag{}, _Sco{});
  }
};
template <class _Backend, class _Type, class _Up, class _Sco>
[[nodiscard]] _CCCL_HOST_DEVICE_API __unv<_Type>
__cuda_atomic_fetch_and_dispatch(_Backend __backend, _Type* __ptr, _Up __op, memory_order __order, _Sco __scope)
{
  using __value_type    = __unv<_Type>;
  using __proxy_t       = __cuda_atomic_deduce_bitwise_t<__value_type>;
  using __proxy_pointee = __copy_cv_t<_Type, __proxy_t>;
  using __proxy_tag     = __cuda_atomic_deduce_bitwise_tag_t<__value_type>;
  __value_type __dst{};
  __proxy_pointee* __ptr_proxy = reinterpret_cast<__proxy_pointee*>(__ptr);
  __proxy_t* __dst_proxy       = reinterpret_cast<__proxy_t*>(&__dst);
  __proxy_t* __op_proxy        = reinterpret_cast<__proxy_t*>(&__op);
#if _CCCL_CUDA_COMPILATION()
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    if (::cuda::std::__cuda_atomic_fetch_and_weak_if_local(__ptr_proxy, *__op_proxy, __dst_proxy))
    {
      return __dst;
    }
  }
#endif // _CCCL_CUDA_COMPILATION()
  __cuda_atomic_bind_fetch_and<_Backend, __proxy_pointee> __bound_and{__backend, __ptr_proxy, __dst_proxy, *__op_proxy};
  __cuda_atomic_fetch_order_dispatch(__backend, __bound_and, __order, __scope, __proxy_tag{});
  return __dst;
}

template <typename _Backend, typename _Type>
struct __cuda_atomic_bind_fetch_max
{
  _Backend __backend;
  _Type* __ptr;
  __unv<_Type>* __dst;
  __unv<_Type> __op;

  template <typename _Atomic_Memorder, typename _Tag, typename _Sco>
  _CCCL_HOST_DEVICE_API void operator()(_Atomic_Memorder __order, _Tag, _Sco)
  {
    ::cuda::std::__cuda_atomic_fetch_max(__backend, __ptr, *__dst, __op, __order, _Tag{}, _Sco{});
  }
};
template <class _Backend, class _Type, class _Up, class _Sco>
[[nodiscard]] _CCCL_HOST_DEVICE_API __unv<_Type>
__cuda_atomic_fetch_max_dispatch(_Backend __backend, _Type* __ptr, _Up __op, memory_order __order, _Sco __scope)
{
  using __value_type    = __unv<_Type>;
  using __proxy_t       = __cuda_atomic_deduce_minmax_t<__value_type>;
  using __proxy_pointee = __copy_cv_t<_Type, __proxy_t>;
  using __proxy_tag     = __cuda_atomic_deduce_minmax_tag_t<__value_type>;
  __value_type __dst{};
  __proxy_pointee* __ptr_proxy = reinterpret_cast<__proxy_pointee*>(__ptr);
  __proxy_t* __dst_proxy       = reinterpret_cast<__proxy_t*>(&__dst);
  __proxy_t* __op_proxy        = reinterpret_cast<__proxy_t*>(&__op);
#if _CCCL_CUDA_COMPILATION()
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    if (::cuda::std::__cuda_atomic_fetch_max_weak_if_local(__ptr_proxy, *__op_proxy, __dst_proxy))
    {
      return __dst;
    }
  }
#endif // _CCCL_CUDA_COMPILATION()
  __cuda_atomic_bind_fetch_max<_Backend, __proxy_pointee> __bound_max{__backend, __ptr_proxy, __dst_proxy, *__op_proxy};
  __cuda_atomic_fetch_order_dispatch(__backend, __bound_max, __order, __scope, __proxy_tag{});
  return __dst;
}

template <typename _Backend, typename _Type>
struct __cuda_atomic_bind_fetch_min
{
  _Backend __backend;
  _Type* __ptr;
  __unv<_Type>* __dst;
  __unv<_Type> __op;

  template <typename _Atomic_Memorder, typename _Tag, typename _Sco>
  _CCCL_HOST_DEVICE_API void operator()(_Atomic_Memorder __order, _Tag, _Sco)
  {
    ::cuda::std::__cuda_atomic_fetch_min(__backend, __ptr, *__dst, __op, __order, _Tag{}, _Sco{});
  }
};
template <class _Backend, class _Type, class _Up, class _Sco>
[[nodiscard]] _CCCL_HOST_DEVICE_API __unv<_Type>
__cuda_atomic_fetch_min_dispatch(_Backend __backend, _Type* __ptr, _Up __op, memory_order __order, _Sco __scope)
{
  using __value_type    = __unv<_Type>;
  using __proxy_t       = __cuda_atomic_deduce_minmax_t<__value_type>;
  using __proxy_pointee = __copy_cv_t<_Type, __proxy_t>;
  using __proxy_tag     = __cuda_atomic_deduce_minmax_tag_t<__value_type>;
  __value_type __dst{};
  __proxy_pointee* __ptr_proxy = reinterpret_cast<__proxy_pointee*>(__ptr);
  __proxy_t* __dst_proxy       = reinterpret_cast<__proxy_t*>(&__dst);
  __proxy_t* __op_proxy        = reinterpret_cast<__proxy_t*>(&__op);
#if _CCCL_CUDA_COMPILATION()
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    if (::cuda::std::__cuda_atomic_fetch_min_weak_if_local(__ptr_proxy, *__op_proxy, __dst_proxy))
    {
      return __dst;
    }
  }
#endif // _CCCL_CUDA_COMPILATION()
  __cuda_atomic_bind_fetch_min<_Backend, __proxy_pointee> __bound_min{__backend, __ptr_proxy, __dst_proxy, *__op_proxy};
  __cuda_atomic_fetch_order_dispatch(__backend, __bound_min, __order, __scope, __proxy_tag{});
  return __dst;
}

template <typename _Backend, typename _Type>
struct __cuda_atomic_bind_fetch_or
{
  _Backend __backend;
  _Type* __ptr;
  __unv<_Type>* __dst;
  __unv<_Type> __op;

  template <typename _Atomic_Memorder, typename _Tag, typename _Sco>
  _CCCL_HOST_DEVICE_API void operator()(_Atomic_Memorder __order, _Tag, _Sco)
  {
    ::cuda::std::__cuda_atomic_fetch_or(__backend, __ptr, *__dst, __op, __order, _Tag{}, _Sco{});
  }
};
template <class _Backend, class _Type, class _Up, class _Sco>
[[nodiscard]] _CCCL_HOST_DEVICE_API __unv<_Type>
__cuda_atomic_fetch_or_dispatch(_Backend __backend, _Type* __ptr, _Up __op, memory_order __order, _Sco __scope)
{
  using __value_type    = __unv<_Type>;
  using __proxy_t       = __cuda_atomic_deduce_bitwise_t<__value_type>;
  using __proxy_pointee = __copy_cv_t<_Type, __proxy_t>;
  using __proxy_tag     = __cuda_atomic_deduce_bitwise_tag_t<__value_type>;
  __value_type __dst{};
  __proxy_pointee* __ptr_proxy = reinterpret_cast<__proxy_pointee*>(__ptr);
  __proxy_t* __dst_proxy       = reinterpret_cast<__proxy_t*>(&__dst);
  __proxy_t* __op_proxy        = reinterpret_cast<__proxy_t*>(&__op);
#if _CCCL_CUDA_COMPILATION()
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    if (::cuda::std::__cuda_atomic_fetch_or_weak_if_local(__ptr_proxy, *__op_proxy, __dst_proxy))
    {
      return __dst;
    }
  }
#endif // _CCCL_CUDA_COMPILATION()
  __cuda_atomic_bind_fetch_or<_Backend, __proxy_pointee> __bound_or{__backend, __ptr_proxy, __dst_proxy, *__op_proxy};
  __cuda_atomic_fetch_order_dispatch(__backend, __bound_or, __order, __scope, __proxy_tag{});
  return __dst;
}

template <typename _Backend, typename _Type>
struct __cuda_atomic_bind_fetch_xor
{
  _Backend __backend;
  _Type* __ptr;
  __unv<_Type>* __dst;
  __unv<_Type> __op;

  template <typename _Atomic_Memorder, typename _Tag, typename _Sco>
  _CCCL_HOST_DEVICE_API void operator()(_Atomic_Memorder __order, _Tag, _Sco)
  {
    ::cuda::std::__cuda_atomic_fetch_xor(__backend, __ptr, *__dst, __op, __order, _Tag{}, _Sco{});
  }
};
template <class _Backend, class _Type, class _Up, class _Sco>
[[nodiscard]] _CCCL_HOST_DEVICE_API __unv<_Type>
__cuda_atomic_fetch_xor_dispatch(_Backend __backend, _Type* __ptr, _Up __op, memory_order __order, _Sco __scope)
{
  using __value_type    = __unv<_Type>;
  using __proxy_t       = __cuda_atomic_deduce_bitwise_t<__value_type>;
  using __proxy_pointee = __copy_cv_t<_Type, __proxy_t>;
  using __proxy_tag     = __cuda_atomic_deduce_bitwise_tag_t<__value_type>;
  __value_type __dst{};
  __proxy_pointee* __ptr_proxy = reinterpret_cast<__proxy_pointee*>(__ptr);
  __proxy_t* __dst_proxy       = reinterpret_cast<__proxy_t*>(&__dst);
  __proxy_t* __op_proxy        = reinterpret_cast<__proxy_t*>(&__op);
#if _CCCL_CUDA_COMPILATION()
  if constexpr (_Backend::__requires_local_memory_workaround)
  {
    if (::cuda::std::__cuda_atomic_fetch_xor_weak_if_local(__ptr_proxy, *__op_proxy, __dst_proxy))
    {
      return __dst;
    }
  }
#endif // _CCCL_CUDA_COMPILATION()
  __cuda_atomic_bind_fetch_xor<_Backend, __proxy_pointee> __bound_xor{__backend, __ptr_proxy, __dst_proxy, *__op_proxy};
  __cuda_atomic_fetch_order_dispatch(__backend, __bound_xor, __order, __scope, __proxy_tag{});
  return __dst;
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

_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ATOMIC_FUNCTIONS_DISPATCH_H
