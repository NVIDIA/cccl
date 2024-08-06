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
struct __ignore
{
  template <typename... Args>
  _CCCL_HOST_DEVICE constexpr __ignore(Args&&...) noexcept
  {}
};
} // namespace detail

struct uninit_t
{
  explicit uninit_t() = default;
};

inline constexpr uninit_t uninit{};
} // namespace cuda::experimental

#endif // __CUDAX_DETAIL_UTILITY_H
