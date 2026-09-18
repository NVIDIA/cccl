//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDA_STD___ATOMIC_SCOPES_H
#define __CUDA_STD___ATOMIC_SCOPES_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD

// REMEMBER CHANGES TO THESE ARE ABI BREAKING
// TODO: Space values out for potential new scopes at an ABI break.
enum thread_scope // NOLINT(cppcoreguidelines-use-enum-class) - Preserve public names and implicit conversions.
{
  thread_scope_system  = 0,
  thread_scope_device  = 1,
  thread_scope_cluster = 3,
  thread_scope_block   = 2,
  thread_scope_thread  = 10
};

struct __thread_scope_tag
{};
struct __thread_scope_thread_tag : __thread_scope_tag
{};
struct __thread_scope_block_tag : __thread_scope_tag
{};
struct __thread_scope_cluster_tag : __thread_scope_tag
{};
struct __thread_scope_device_tag : __thread_scope_tag
{};
struct __thread_scope_system_tag : __thread_scope_tag
{};

template <int _Scope>
struct __scope_enum_to_tag
{};
/* This would be the implementation once an actual thread-scope backend exists.
template<> struct __scope_enum_to_tag<(int)thread_scope_thread> {
    using type = __thread_scope_thread_tag; };
Until then: */
template <>
struct __scope_enum_to_tag<(int) thread_scope_thread>
{
  using __tag = __thread_scope_block_tag;
};
template <>
struct __scope_enum_to_tag<(int) thread_scope_block>
{
  using __tag = __thread_scope_block_tag;
};
template <>
struct __scope_enum_to_tag<(int) thread_scope_cluster>
{
  using __tag = __thread_scope_cluster_tag;
};
template <>
struct __scope_enum_to_tag<(int) thread_scope_device>
{
  using __tag = __thread_scope_device_tag;
};
template <>
struct __scope_enum_to_tag<(int) thread_scope_system>
{
  using __tag = __thread_scope_system_tag;
};

template <int _Scope>
using __scope_to_tag = typename __scope_enum_to_tag<_Scope>::__tag;

_CCCL_END_NAMESPACE_CUDA_STD

_CCCL_BEGIN_NAMESPACE_CUDA

using ::cuda::std::thread_scope;
using ::cuda::std::thread_scope_block;
using ::cuda::std::thread_scope_cluster;
using ::cuda::std::thread_scope_device;
using ::cuda::std::thread_scope_system;
using ::cuda::std::thread_scope_thread;

using ::cuda::std::__thread_scope_block_tag;
using ::cuda::std::__thread_scope_cluster_tag;
using ::cuda::std::__thread_scope_device_tag;
using ::cuda::std::__thread_scope_system_tag;

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // __CUDA_STD___ATOMIC_SCOPES_H
