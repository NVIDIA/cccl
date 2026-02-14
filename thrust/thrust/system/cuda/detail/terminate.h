// SPDX-FileCopyrightText: Copyright (c) 2016, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__exception/terminate.h>

#include <cstdio>

THRUST_NAMESPACE_BEGIN
namespace system::cuda::detail
{
inline _CCCL_HOST_DEVICE void terminate_with_message(const char* message)
{
  printf("%s\n", message);
  ::cuda::std::terminate();
}
} // namespace system::cuda::detail
THRUST_NAMESPACE_END
