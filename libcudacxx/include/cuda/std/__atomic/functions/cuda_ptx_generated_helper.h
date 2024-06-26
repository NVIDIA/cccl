//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCUDACXX___ATOMIC_FUNCTIONS_CUDA_PTX_GENERATED_HELPER_H
#define _LIBCUDACXX___ATOMIC_FUNCTIONS_CUDA_PTX_GENERATED_HELPER_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__atomic/scopes.h>

_LIBCUDACXX_BEGIN_NAMESPACE_STD

enum class __atomic_cuda_memory_orders
{
  _relaxed,
  _release,
  _acquire,
  _acq_rel,
  _seq_cst,
  _volatile,
};

template <__atomic_cuda_memory_orders _Order>
struct __atomic_memorder_constraint;

template <>
struct __atomic_memorder_constraint<__atomic_cuda_memory_orders::_relaxed>
{
  static constexpr char value[] = ".relaxed";
};
template <>
struct __atomic_memorder_constraint<__atomic_cuda_memory_orders::_release>
{
  static constexpr char value[] = ".release";
};
template <>
struct __atomic_memorder_constraint<__atomic_cuda_memory_orders::_acquire>
{
  static constexpr char value[] = ".acquire";
};
template <>
struct __atomic_memorder_constraint<__atomic_cuda_memory_orders::_acq_rel>
{
  static constexpr char value[] = ".acq_rel";
};
template <>
struct __atomic_memorder_constraint<__atomic_cuda_memory_orders::_seq_cst>
{
  static constexpr char value[] = ".sc";
};
template <>
struct __atomic_memorder_constraint<__atomic_cuda_memory_orders::_volatile>
{
  static constexpr char value[] = "";
};

using __atomic_memorder_constraint_relaxed  = __atomic_memorder_constraint<__atomic_cuda_memory_orders::_relaxed>;
using __atomic_memorder_constraint_release  = __atomic_memorder_constraint<__atomic_cuda_memory_orders::_release>;
using __atomic_memorder_constraint_acquire  = __atomic_memorder_constraint<__atomic_cuda_memory_orders::_acquire>;
using __atomic_memorder_constraint_acq_rel  = __atomic_memorder_constraint<__atomic_cuda_memory_orders::_acq_rel>;
using __atomic_memorder_constraint_seq_cst  = __atomic_memorder_constraint<__atomic_cuda_memory_orders::_seq_cst>;
using __atomic_memorder_constraint_volatile = __atomic_memorder_constraint<__atomic_cuda_memory_orders::_volatile>;

template <typename T>
struct __atomic_scope_constraint;

template <>
struct __atomic_scope_constraint<__thread_scope_device_tag>
{
  static constexpr char value[] = ".gpu";
};
template <>
struct __atomic_scope_constraint<__thread_scope_system_tag>
{
  static constexpr char value[] = ".sys";
};
template <>
struct __atomic_scope_constraint<__thread_scope_block_tag>
{
  static constexpr char value[] = ".cta";
};

template <typename T>
struct __atomic_membar_scope_constraint;

template <>
struct __atomic_membar_scope_constraint<__thread_scope_device_tag>
{
  static constexpr char value[] = ".gl";
};
template <>
struct __atomic_membar_scope_constraint<__thread_scope_system_tag>
{
  static constexpr char value[] = ".sys";
};
template <>
struct __atomic_membar_scope_constraint<__thread_scope_block_tag>
{
  static constexpr char value[] = ".cta";
};

enum class __atomic_cuda_operation
{
  _add,
  _or,
  _xor,
  _and,
  _min,
  _max,
};

template <__atomic_cuda_operation T>
struct __atomic_operation_constraint;

template <>
struct __atomic_operation_constraint<__atomic_cuda_operation::_add>
{
  static constexpr char value[] = ".add";
};
template <>
struct __atomic_operation_constraint<__atomic_cuda_operation::_or>
{
  static constexpr char value[] = ".or";
};
template <>
struct __atomic_operation_constraint<__atomic_cuda_operation::_xor>
{
  static constexpr char value[] = ".xor";
};
template <>
struct __atomic_operation_constraint<__atomic_cuda_operation::_and>
{
  static constexpr char value[] = ".and";
};
template <>
struct __atomic_operation_constraint<__atomic_cuda_operation::_min>
{
  static constexpr char value[] = ".min";
};
template <>
struct __atomic_operation_constraint<__atomic_cuda_operation::_max>
{
  static constexpr char value[] = ".max";
};

template <bool>
struct __atomic_mmio_constraint;

template <>
struct __atomic_mmio_constraint<true>
{
  static constexpr char value[] = ".mmio";
};
template <>
struct __atomic_mmio_constraint<false>
{
  static constexpr char value[] = "";
};

_LIBCUDACXX_END_NAMESPACE_STD

#endif // _LIBCUDACXX___ATOMIC_FUNCTIONS_CUDA_PTX_GENERATED_H
