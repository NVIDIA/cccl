// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Regression test for https://github.com/NVIDIA/cccl/issues/11403. Compiled for a higher GPU architecture than other.cu

#include <cub/util_device.cuh>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <cuda/__runtime/api_wrapper.h>
#include <cuda/std/__cccl/assert.h>

#include <cstdio>

#define STRINGIFY_(...) #__VA_ARGS__
#define STRINGIFY(...)  STRINGIFY_(__VA_ARGS__)

int main()
{
#ifdef __CUDA_ARCH_LIST__
  std::printf("__CUDA_ARCH_LIST__=%s\n", STRINGIFY(__CUDA_ARCH_LIST__));
#endif // __CUDA_ARCH_LIST__

  std::printf("target_compute_capabilities=");
  for (const auto& target_cc : ::cuda::__target_compute_capabilities())
  {
    std::printf("%d.%d ", target_cc.major_cap(), target_cc.minor_cap());
  }
  std::printf("\n");

  ::cuda::compute_capability cc{};
  _CCCL_TRY_RUNTIME_API(cub::detail::ptx_compute_cap, "ptx_compute_cap failed", cc);
  std::printf("ptx_compute_cap=%d.%d\n", cc.major_cap(), cc.minor_cap());
  _CCCL_VERIFY(cc == (::cuda::compute_capability{8, 0}), "ptx_compute_cap did not return 8.0");

  thrust::device_vector<int> v(1000, 1);
  _CCCL_TRY_RUNTIME_API(cudaGetLastError, "device_vector construction failed");

  const int sum = thrust::reduce(v.begin(), v.end());
  _CCCL_TRY_RUNTIME_API(cudaGetLastError, "thrust::reduce failed");
  _CCCL_VERIFY(sum == 1000, "thrust::reduce did not return 1000");

  return 0;
}
