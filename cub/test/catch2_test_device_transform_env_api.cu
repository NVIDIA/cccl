// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_transform.cuh>

#include <thrust/device_vector.h>

#include <cuda/__execution/tune.h>

#include <iostream>

#include <c2h/catch2_test_helper.h>

#if _CCCL_STD_VER >= 2020

// nvcc turns the `.member = value,` C++ syntax into GNU's `member: value,` when clang (14 - 21) is used
_CCCL_DIAG_PUSH
#  if _CCCL_COMPILER(CLANG)
_CCCL_DIAG_SUPPRESS_CLANG("-Wgnu-designator")
#  endif // _CCCL_COMPILER(CLANG)

// example-begin transform-policy-selector
struct TransformPolicySelector
{
  __host__ __device__ constexpr auto operator()(cuda::compute_capability /*cc*/) const -> cub::TransformPolicy
  {
    return {.min_bytes_in_flight = 64 * 1024,
            .algorithm           = cub::TransformAlgorithm::prefetch,
            .prefetch            = {.threads_per_block = 256},
            .vectorized          = {}, // unused because algorithm is prefetch
            .async_copy          = {}}; // unused because algorithm is prefetch
  }
};
// example-end transform-policy-selector

_CCCL_DIAG_POP

C2H_TEST("cub::DeviceTransform::Transform accepts a custom policy selector", "[transform][env]")
{
  // example-begin transform-tuning
  auto d_input  = thrust::device_vector<int>{1, 2, 3, 4, 5, 6, 7};
  auto d_output = thrust::device_vector<int>(7, thrust::no_init);

  const auto error = cub::DeviceTransform::Transform(
    d_input.data(),
    d_output.data(),
    d_input.size(),
    cuda::std::negate{},
    cuda::execution::tune(TransformPolicySelector{}));
  if (error != cudaSuccess)
  {
    std::cerr << "cub::DeviceTransform::Transform failed with status: " << error << '\n';
  }

  thrust::device_vector<int> expected{-1, -2, -3, -4, -5, -6, -7};
  // example-end transform-tuning

  REQUIRE(error == cudaSuccess);
  REQUIRE(d_output == expected);
}

#else // _CCCL_STD_VER >= 2020

// we need a dummy test for C++17, otherwise the return code of the test executable is 2 (not 0)
C2H_TEST("cub::DeviceTransform::Transform dummy test", "[transform][env]")
{
  SUCCEED();
}

#endif // _CCCL_STD_VER >= 2020
