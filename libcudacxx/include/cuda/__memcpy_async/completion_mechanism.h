//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___BARRIER_COMPLETION_MECHANISM_H
#define _CUDA___BARRIER_COMPLETION_MECHANISM_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

_LIBCUDACXX_BEGIN_NAMESPACE_CUDA

//! @brief __completion_mechanism allows memcpy_async to report back what completion
//! mechanism it used. This is necessary to determine in which way to synchronize
//! the memcpy_async with a sync object (barrier or pipeline).
//
//! In addition, we use this enum to create bit flags so that calling functions
//! can specify which completion mechanisms can be used (__sync is always
//! allowed).
enum class __completion_mechanism
{
  __sync                 = 0,
  __mbarrier_complete_tx = 1 << 0, // Use powers of two here to support the
  __async_group          = 1 << 1, // bit flag use case
  __async_bulk_group     = 1 << 2,
};

_LIBCUDACXX_END_NAMESPACE_CUDA

#endif // _CUDA___BARRIER_COMPLETION_MECHANISM_H
