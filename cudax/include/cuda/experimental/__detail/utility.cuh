//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_DETAIL_UTILITY_H
#define __CUDAX_DETAIL_UTILITY_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

namespace cuda::experimental
{
namespace detail
{
// This is a helper type that can be used to ignore function arguments.
struct [[maybe_unused]] __ignore
{
  __ignore() = default;

  template <typename _Arg>
  _CCCL_HOST_DEVICE constexpr __ignore(_Arg&&) noexcept
  {}
};

// Classes can inherit from this type to become immovable.
struct __immovable
{
  __immovable()                         = default;
  __immovable(__immovable&&)            = delete;
  __immovable& operator=(__immovable&&) = delete;
};
} // namespace detail

struct uninit_t
{
  explicit uninit_t() = default;
};

_CCCL_GLOBAL_CONSTANT uninit_t uninit{};
} // namespace cuda::experimental

#endif // __CUDAX_DETAIL_UTILITY_H
