// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

template <typename T>
static [[nodiscard]] _CCCL_DEVICE_API T* optimizeSmemPtr(const T* smemGeneric)
{
  // See https://nvbugspro.nvidia.com/bug/4907996

  // 1. Convert to 32-bit shared memory pointer
  uint32_t smem32 = __cvta_generic_to_shared(smemGeneric);
  // 2. Pretend to NVVM that the 32-bit pointer is modified. This is required to avoid NVVM constant
  // propagation from pulling the smem32 definition into loops and branches in subsequent code.
  asm("" : "+r"(smem32));
  // 3. Make a generic pointer to smem that is constructed using `__cvta_shared_to_generic`. This
  // benefits from an
  //    optimization pass in NVVM that performs the following simplification:
  //    __cvta_generic_to_shared(__cvta_shared_to_generic(x))    => x.
  //    In our case, `x` is smem32, which is exactly what we want.
  return reinterpret_cast<T*>(__cvta_shared_to_generic(smem32));
}
