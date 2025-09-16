//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___ANNOTATED_PTR_APPLY_ACCESS_PROPERTY_H
#define _CUDA___ANNOTATED_PTR_APPLY_ACCESS_PROPERTY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__annotated_ptr/access_property.h>
#include <cuda/__memory/address_space.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _Shape>
_CCCL_API inline void apply_access_property(
  [[maybe_unused]] const volatile void* __ptr,
  [[maybe_unused]] _Shape __shape,
  [[maybe_unused]] access_property::persisting __prop) noexcept
{
  // clang-format off
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (_CCCL_ASSERT(__ptr != nullptr, "null pointer");
     if (!::cuda::device::is_address_from(__ptr, ::cuda::device::address_space::global))
     {
       return;
     }
     constexpr size_t __line_size = 128;
     auto __p                     = reinterpret_cast<uint8_t*>(const_cast<void*>(__ptr));
     auto __nbytes                = static_cast<size_t>(__shape);
     // Apply to all 128 bytes aligned cache lines inclusive of __p
     for (size_t __i = 0; __i < __nbytes; __i += __line_size) {
       asm volatile("prefetch.global.L2::evict_last [%0];" ::"l"(__p + __i) :);
     }))
  // clang-format on
}

template <typename _Shape>
_CCCL_API inline void apply_access_property(
  [[maybe_unused]] const volatile void* __ptr,
  [[maybe_unused]] _Shape __shape,
  [[maybe_unused]] access_property::normal __prop) noexcept
{
  // clang-format off
  NV_IF_TARGET(
    NV_PROVIDES_SM_80,
    (_CCCL_ASSERT(__ptr != nullptr, "null pointer");
     if (!::cuda::device::is_address_from(__ptr, ::cuda::device::address_space::global))
     {
       return;
     }
     constexpr size_t __line_size = 128;
     auto __p                     = reinterpret_cast<uint8_t*>(const_cast<void*>(__ptr));
     auto __nbytes                = static_cast<size_t>(__shape);
     // Apply to all 128 bytes aligned cache lines inclusive of __p
     for (size_t __i = 0; __i < __nbytes; __i += __line_size) {
       asm volatile("prefetch.global.L2::evict_normal [%0];" ::"l"(__p + __i) :);
     }))
  // clang-format on
}

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA___ANNOTATED_PTR_APPLY_ACCESS_PROPERTY_H
