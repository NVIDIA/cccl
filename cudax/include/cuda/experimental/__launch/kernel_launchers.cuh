
//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#ifndef _CUDAX__LAUNCH_KERNEL_LAUNCHERS
#define _CUDAX__LAUNCH_KERNEL_LAUNCHERS

#include <cuda/std/detail/__config>

#include <cuda/std/__functional/invoke.h>

#if _CCCL_STD_VER >= 2017
namespace cuda::experimental::detail
{
template <typename Config, typename Kernel, typename... Args>
__global__ void kernel_launcher(const Config conf, Kernel kernel_fn, Args... args)
{
  kernel_fn(conf, args...);
}

template <typename Kernel, typename... Args>
__global__ void kernel_launcher_no_config(Kernel kernel_fn, Args... args)
{
  kernel_fn(args...);
}

template <typename Kernel, typename ConfOrDims, typename... Args>
_CCCL_NODISCARD auto get_kernel_launcher()
{
  if constexpr (::cuda::std::is_invocable_v<Kernel, ConfOrDims, Args...>
                || __nv_is_extended_device_lambda_closure_type(Kernel))
  {
    return detail::kernel_launcher<ConfOrDims, Kernel, Args...>;
  }
  else
  {
    static_assert(::cuda::std::is_invocable_v<Kernel, Args...>);
    auto launcher = detail::kernel_launcher_no_config<Kernel, Args...>;
  }
}
} // namespace cuda::experimental::detail

#endif
#endif
