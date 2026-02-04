//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: --extended-lambda
// UNSUPPORTED: nvrtc

#include <cuda/devices>
#include <cuda/launch>
#include <cuda/stream>

#include "../common/utility.cuh"

__host__ void test_extended_lambda()
{
  cuda::stream stream{cuda::devices[0]};
  test::pinned<int> i(0);
  auto config           = cuda::block_dims<32>() & cuda::grid_dims<1>();
  auto assign_42_lambda = [] __device__(int* pi) {
    *pi = 42;
  };
  cuda::launch(stream, config, assign_42_lambda, i.get());
  stream.sync();
  assert(*i == 42);

  auto assign_1337_lambda = [] __device__(auto config, int* pi) {
    static_assert(cuda::gpu_thread.count(cuda::block, config) == 32);
    static_assert(cuda::block.count(cuda::grid, config) == 1);
    *pi = 1337;
  };
  cuda::launch(stream, config, assign_1337_lambda, config, i.get());
  stream.sync();
  assert(*i == 1337);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, test_extended_lambda();)
  return 0;
}
