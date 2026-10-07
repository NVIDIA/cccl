//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___FABRIC_FENCE_H
#define _CUDA___FABRIC_FENCE_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_CUDACC_AT_LEAST(13, 4) && !_CCCL_COMPILER(NVRTC)

#  include <cuda/__ptx/instructions/fence.h>
#  include <cuda/std/__atomic/order.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/integral_constant.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

//! @brief State-space selection for the PTX proxy-async fence family.
enum class proxy_async_space
{
  all,
  global,
  shared_cluster,
  shared_cta
};

//! @brief Execute the PTX fence.proxy.async form selected by @p _Space.
//! @tparam _Space Select all spaces, global, shared::cluster, or shared::cta.
template <proxy_async_space _Space = proxy_async_space::all>
_CCCL_DEVICE_API void fence_proxy_async() noexcept
{
  if constexpr (_Space == proxy_async_space::all)
  {
    ::cuda::ptx::fence_proxy_async();
  }
  else if constexpr (_Space == proxy_async_space::global)
  {
    ::cuda::ptx::fence_proxy_async(::cuda::ptx::space_global);
  }
  else if constexpr (_Space == proxy_async_space::shared_cluster)
  {
    ::cuda::ptx::fence_proxy_async(::cuda::ptx::space_cluster);
  }
  else if constexpr (_Space == proxy_async_space::shared_cta)
  {
    ::cuda::ptx::fence_proxy_async(::cuda::ptx::space_shared);
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<::cuda::std::integral_constant<proxy_async_space, _Space>>,
                  "unsupported proxy-async address space");
  }
}

//! @brief Execute PTX fence.proxy.generic::fabric.alias with system scope.
//! @tparam _Order Either memory_order_acquire or memory_order_release.
template <::cuda::memory_order _Order>
_CCCL_DEVICE_API void fence_proxy_generic_fabric_alias() noexcept
{
  if constexpr (_Order == ::cuda::memory_order_acquire)
  {
    ::cuda::ptx::fence_proxy_generic_fabric_alias(::cuda::ptx::sem_acquire);
  }
  else if constexpr (_Order == ::cuda::memory_order_release)
  {
    ::cuda::ptx::fence_proxy_generic_fabric_alias(::cuda::ptx::sem_release);
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<::cuda::std::integral_constant<::cuda::memory_order, _Order>>,
                  "fabric alias fences require acquire or release ordering");
  }
}

//! @brief Execute PTX fence.proxy.fabric::generic.alias with system scope.
//! @tparam _Order Either memory_order_acquire or memory_order_release.
template <::cuda::memory_order _Order>
_CCCL_DEVICE_API void fence_proxy_fabric_generic_alias() noexcept
{
  if constexpr (_Order == ::cuda::memory_order_acquire)
  {
    ::cuda::ptx::fence_proxy_fabric_generic_alias(::cuda::ptx::sem_acquire);
  }
  else if constexpr (_Order == ::cuda::memory_order_release)
  {
    ::cuda::ptx::fence_proxy_fabric_generic_alias(::cuda::ptx::sem_release);
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<::cuda::std::integral_constant<::cuda::memory_order, _Order>>,
                  "fabric alias fences require acquire or release ordering");
  }
}

//! @brief Execute PTX fence.proxy.fabric::fabric.alias with system scope.
//! @tparam _Order Either memory_order_acquire or memory_order_release.
template <::cuda::memory_order _Order>
_CCCL_DEVICE_API void fence_proxy_fabric_fabric_alias() noexcept
{
  if constexpr (_Order == ::cuda::memory_order_acquire)
  {
    ::cuda::ptx::fence_proxy_fabric_fabric_alias(::cuda::ptx::sem_acquire);
  }
  else if constexpr (_Order == ::cuda::memory_order_release)
  {
    ::cuda::ptx::fence_proxy_fabric_fabric_alias(::cuda::ptx::sem_release);
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<::cuda::std::integral_constant<::cuda::memory_order, _Order>>,
                  "fabric alias fences require acquire or release ordering");
  }
}

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_CUDACC_AT_LEAST(13, 4) && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___FABRIC_FENCE_H
