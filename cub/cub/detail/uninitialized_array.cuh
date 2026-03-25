// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_namespace.cuh>

#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <typename T, ::cuda::std::size_t N, ::cuda::std::size_t Alignment = alignof(T)>
struct uninitialized_array
{
  using value_type           = T;
  static constexpr auto size = ::cuda::std::integral_constant<::cuda::std::size_t, N>{};
  alignas(Alignment) char data_[N * sizeof(T)];
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T* data()
  {
    return reinterpret_cast<T*>(data_);
  }
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE const T* data() const
  {
    return reinterpret_cast<const T*>(data_);
  }
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T& operator[](unsigned int idx)
  {
    return data()[idx];
  }
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE const T& operator[](unsigned int idx) const
  {
    return data()[idx];
  }
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE T (&as_array())[N]
  {
    return *reinterpret_cast<T(*)[N]>(data_);
  }
};
} // namespace detail

CUB_NAMESPACE_END
