//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___MULTI_GPU_COMMUNICATOR_H
#define _CUDA_EXPERIMENTAL___MULTI_GPU_COMMUNICATOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__utility/move.h>

#include <cuda/experimental/__device/logical_device.cuh>
#include <cuda/experimental/__nccl/nccl_api.h>

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

namespace cuda::experimental
{
class communicator
{
  struct __private_tag
  {};

public:
  _CCCL_HOST_API communicator(__nccl::__ncclComm_t __comm, logical_device __device)
      : communicator{__private_tag{}, __comm, ::cuda::std::move(__device)}
  {}

  [[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::uint32_t rank() const
  {
    return static_cast<::cuda::std::uint32_t>(__nccl::__ncclCommUserRank(comm()));
  }

  [[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::uint32_t size() const
  {
    return static_cast<::cuda::std::uint32_t>(__nccl::__ncclCommCount(comm()));
  }

  [[nodiscard]] _CCCL_HOST_API constexpr __nccl::__ncclComm_t comm() const noexcept
  {
    return __comm_;
  }

  [[nodiscard]] _CCCL_HOST_API constexpr const logical_device& device() const noexcept
  {
    return __device_;
  }

private:
  // Grand central constructor, all other constructors must come through this one
  _CCCL_HOST_API communicator(const __private_tag, __nccl::__ncclComm_t __comm, logical_device __device)
      : __comm_{[&] {
        if (const auto __nccl_device = __nccl::__ncclCommCuDevice(__comm);
            __nccl_device != __device.underlying_device())
        {
          _CCCL_THROW(::std::runtime_error,
                      "Inconsistent devices, NCCL communicator device and provided logical device do not match");
        }
        return __comm;
      }()}
      , __device_{::cuda::std::move(__device)}
  {}

  __nccl::__ncclComm_t __comm_{};
  logical_device __device_;
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

// NOLINTEND(bugprone-reserved-identifier)

#endif // _CUDA_EXPERIMENTAL___MULTI_GPU_COMMUNICATOR_H
