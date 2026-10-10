//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: --extended-lambda
// UNSUPPORTED: nvrtc

#include <cuda/launch>

template <typename DefaultConfig>
struct kernel_with_default_config
{
  DefaultConfig config;

  kernel_with_default_config(DefaultConfig c)
      : config(c)
  {}

  DefaultConfig default_config() const
  {
    return config;
  }
};

void test_default_config()
{
  auto grid  = cuda::grid_dims(4);
  auto block = cuda::block_dims<256>;

  [[maybe_unused]] auto verify_lambda = [] __device__(auto config) {
    static_assert(cuda::gpu_thread.count(cuda::block, config) == 256);
    static_assert(cuda::block.count(cuda::grid, config) == 4);
  };

  kernel_with_default_config kernel{cuda::make_config(block, grid, cuda::cooperative_launch())};
  static_assert(cuda::__is_kernel_config<decltype(kernel.default_config())>);
  static_assert(cuda::__kernel_has_default_config<decltype(kernel)>);
  (void) verify_lambda;
  (void) kernel;
}

int main(int, char**)
{
  return 0;
}
