//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___ATOMIC_FUNCTIONS_GENERIC_H
#define _CUDA_STD___ATOMIC_FUNCTIONS_GENERIC_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/functions/common.h>
#include <cuda/std/__atomic/functions/generic_rmw.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/cstdint>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

template <class _Backend,
          class _Pointee,
          class _Cas,
          class _Order,
          class _Operand,
          class _Sco,
          enable_if_t<(_Operand::__op == __cuda_atomic_operand::_b) && (_Operand::__size < _Backend::__smallest_cas),
                      bool> = false>
_CCCL_HOST_DEVICE_API bool __cuda_atomic_compare_exchange(
  _Backend __backend,
  _Pointee* __ptr,
  __unv<_Pointee>& __dst,
  __unv<_Pointee> __cmp,
  __unv<_Pointee> __op,
  _Cas,
  _Order __order,
  _Operand,
  _Sco __scope)
{
  using _Type                 = __unv<_Pointee>;
  constexpr size_t __rmw_size = _Backend::__smallest_cas;
  static_assert(__rmw_size <= _Backend::__widest_cas, "atomic CAS cannot be widened beyond the backend's widest CAS");

  using __rmw_operand = __cuda_atomic_operand_tag<__cuda_atomic_operand::_b, __rmw_size>;
  const auto __result = ::cuda::std::__cuda_atomic_rmw(
    __backend,
    __ptr,
    __cuda_atomic_compare_exchange_op<_Type>{__cmp, __op},
    __order,
    ::cuda::std::__cuda_atomic_compare_exchange_initial_load_order(__order),
    _Operand{},
    __rmw_operand{},
    __scope);
  __dst = __result.__old;
  return __result.__applied;
}

template <class _Backend,
          class _Pointee,
          class _Cas,
          class _Order,
          class _Operand,
          class _Sco,
          enable_if_t<(_Operand::__op == __cuda_atomic_operand::_b) && (_Operand::__size > _Backend::__widest_cas),
                      bool> = false>
_CCCL_HOST_DEVICE_API bool __cuda_atomic_compare_exchange(
  _Backend, _Pointee*, __unv<_Pointee>&, __unv<_Pointee>, __unv<_Pointee>, _Cas, _Order, _Operand, _Sco)
{
  static_assert(_Operand::__size < _Backend::__widest_cas, "the backend must provide its widest CAS operation");
  return false;
}

template <class _Backend,
          class _Type,
          class _Order,
          class _Operand,
          class _Sco,
          __cuda_atomic_enable_generic_rmw<_Backend, _Operand> = false>
_CCCL_HOST_DEVICE_API void __cuda_atomic_fetch_add(
  _Backend __backend, _Type* __ptr, __unv<_Type>& __dst, __unv<_Type> __op, _Order __order, _Operand, _Sco __scope)
{
  using _ValueType = __unv<_Type>;
  __dst            = ::cuda::std::__cuda_atomic_fetch_update(
    __backend, __ptr, __cuda_atomic_op_bind<_ValueType, __cuda_atomic_op_fetch_add>{__op}, __order, _Operand{}, __scope);
}

template <class _Backend,
          class _Type,
          class _Order,
          class _Operand,
          class _Sco,
          __cuda_atomic_enable_generic_rmw<_Backend, _Operand> = false>
_CCCL_HOST_DEVICE_API void __cuda_atomic_fetch_sub(
  _Backend __backend, _Type* __ptr, __unv<_Type>& __dst, __unv<_Type> __op, _Order __order, _Operand, _Sco __scope)
{
  using _ValueType = __unv<_Type>;
  __dst            = ::cuda::std::__cuda_atomic_fetch_update(
    __backend, __ptr, __cuda_atomic_op_bind<_ValueType, __cuda_atomic_op_fetch_sub>{__op}, __order, _Operand{}, __scope);
}

template <class _Backend,
          class _Type,
          class _Order,
          class _Operand,
          class _Sco,
          __cuda_atomic_enable_generic_rmw<_Backend, _Operand> = false>
_CCCL_HOST_DEVICE_API void __cuda_atomic_fetch_and(
  _Backend __backend, _Type* __ptr, __unv<_Type>& __dst, __unv<_Type> __op, _Order __order, _Operand, _Sco __scope)
{
  using _ValueType = __unv<_Type>;
  __dst            = ::cuda::std::__cuda_atomic_fetch_update(
    __backend,
    __ptr,
    __cuda_atomic_op_bind<_ValueType, ::cuda::std::bit_and<_ValueType>>{__op},
    __order,
    _Operand{},
    __scope);
}

template <class _Backend,
          class _Type,
          class _Order,
          class _Operand,
          class _Sco,
          __cuda_atomic_enable_generic_rmw<_Backend, _Operand> = false>
_CCCL_HOST_DEVICE_API void __cuda_atomic_fetch_or(
  _Backend __backend, _Type* __ptr, __unv<_Type>& __dst, __unv<_Type> __op, _Order __order, _Operand, _Sco __scope)
{
  using _ValueType = __unv<_Type>;
  __dst            = ::cuda::std::__cuda_atomic_fetch_update(
    __backend,
    __ptr,
    __cuda_atomic_op_bind<_ValueType, ::cuda::std::bit_or<_ValueType>>{__op},
    __order,
    _Operand{},
    __scope);
}

template <class _Backend,
          class _Type,
          class _Order,
          class _Operand,
          class _Sco,
          __cuda_atomic_enable_generic_rmw<_Backend, _Operand> = false>
_CCCL_HOST_DEVICE_API void __cuda_atomic_fetch_xor(
  _Backend __backend, _Type* __ptr, __unv<_Type>& __dst, __unv<_Type> __op, _Order __order, _Operand, _Sco __scope)
{
  using _ValueType = __unv<_Type>;
  __dst            = ::cuda::std::__cuda_atomic_fetch_update(
    __backend,
    __ptr,
    __cuda_atomic_op_bind<_ValueType, ::cuda::std::bit_xor<_ValueType>>{__op},
    __order,
    _Operand{},
    __scope);
}

template <class _Backend,
          class _Type,
          class _Order,
          class _Operand,
          class _Sco,
          __cuda_atomic_enable_generic_rmw<_Backend, _Operand> = false>
_CCCL_HOST_DEVICE_API void __cuda_atomic_fetch_min(
  _Backend __backend, _Type* __ptr, __unv<_Type>& __dst, __unv<_Type> __op, _Order __order, _Operand, _Sco __scope)
{
  using _ValueType = __unv<_Type>;
  __dst            = ::cuda::std::__cuda_atomic_fetch_update(
    __backend, __ptr, __cuda_atomic_op_bind<_ValueType, __cuda_atomic_op_fetch_min>{__op}, __order, _Operand{}, __scope);
}

template <class _Backend,
          class _Type,
          class _Order,
          class _Operand,
          class _Sco,
          __cuda_atomic_enable_generic_rmw<_Backend, _Operand> = false>
_CCCL_HOST_DEVICE_API void __cuda_atomic_fetch_max(
  _Backend __backend, _Type* __ptr, __unv<_Type>& __dst, __unv<_Type> __op, _Order __order, _Operand, _Sco __scope)
{
  using _ValueType = __unv<_Type>;
  __dst            = ::cuda::std::__cuda_atomic_fetch_update(
    __backend, __ptr, __cuda_atomic_op_bind<_ValueType, __cuda_atomic_op_fetch_max>{__op}, __order, _Operand{}, __scope);
}

template <class _Backend,
          class _Type,
          class _Order,
          class _Operand,
          class _Sco,
          __cuda_atomic_enable_generic_rmw<_Backend, _Operand> = false>
_CCCL_HOST_DEVICE_API void __cuda_atomic_exchange(
  _Backend __backend, _Type* __ptr, __unv<_Type>& __dst, __unv<_Type> __op, _Order __order, _Operand, _Sco __scope)
{
  using _ValueType = __unv<_Type>;
  __dst            = ::cuda::std::__cuda_atomic_fetch_update(
    __backend, __ptr, __cuda_atomic_op_bind<_ValueType, __cuda_atomic_op_store>{__op}, __order, _Operand{}, __scope);
}
_CCCL_END_NAMESPACE_CUDA_STD

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___ATOMIC_FUNCTIONS_GENERIC_H
