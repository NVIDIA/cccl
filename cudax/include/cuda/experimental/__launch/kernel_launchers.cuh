
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

#if _CCCL_STD_VER >= 2017
namespace cuda::experimental::detail
{
template <typename Config, typename Kernel, class... Args>
__global__ void kernel_launcher(const Config conf, Kernel kernel_fn, Args... args)
{
  kernel_fn(conf, args...);
}

template <typename Kernel, class... Args>
__global__ void kernel_launcher_no_config(Kernel kernel_fn, Args... args)
{
  kernel_fn(args...);
}
} // namespace cuda::experimental::detail

#endif
#endif
