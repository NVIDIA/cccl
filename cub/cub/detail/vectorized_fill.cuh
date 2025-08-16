// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/type_traits>
#include <cuda/std/utility>

CUB_NAMESPACE_BEGIN

namespace detail
{

template <typename T, int ItemsPerThread>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE void vectorized_fill(T (&data)[ItemsPerThread], T value)
{
  // Attempt vectorization only for small types, if the type is trivially copyable and, to avoid potential complication
  // from involved alignment requirements, also require the type to be arithmetic.
  if constexpr (sizeof(T) < sizeof(::cuda::std::uint32_t)
                && ::cuda::std::is_trivially_copyable_v<T> && ::cuda::std::is_arithmetic_v<T>)
  {
    constexpr int items_per_dword = sizeof(::cuda::std::uint32_t) / sizeof(T);
    constexpr int dword_count     = ItemsPerThread / items_per_dword;

    ::cuda::std::uint32_t vectorized_default = 0;
    for (int i = 0; i < items_per_dword; ++i)
    {
      ::cuda::std::memcpy(reinterpret_cast<T*>(&vectorized_default) + i, &value, sizeof(T));
    }

    // Assertion to check if the address of 'data' is correctly aligned.
    _CCCL_ASSERT(reinterpret_cast<uintptr_t>(&data) % 4 == 0,
                 "Target array must be four-byte aligned to avoid misaligned writes below.");

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < dword_count; ++i)
    {
      reinterpret_cast<::cuda::std::uint32_t*>(data)[i] = vectorized_default;
    }

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = dword_count * items_per_dword; i < ItemsPerThread; ++i)
    {
      data[i] = value;
    }
  }
  else
  {
    _CCCL_PRAGMA_UNROLL_FULL()
    for (int i = 0; i < ItemsPerThread; ++i)
    {
      data[i] = value;
    }
  }
}

} // namespace detail

CUB_NAMESPACE_END
