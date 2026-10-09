// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Regression test for https://github.com/NVIDIA/cccl/issues/11403 and
// https://github.com/NVIDIA/cccl/pull/11440#issuecomment-5699615518.
//
// cub::detail::ptx_compute_cap() must report the architecture that the device code, which is actually executed on the
// current GPU, was compiled for (its __CUDA_ARCH__). This is not necessarily the GPU's own architecture, nor the
// highest architecture in __CUDA_ARCH_LIST__: CUDA may run native SASS built from a lower virtual architecture, and
// device-linking against a translation unit compiled for a lower architecture (other.cu) clamps the virtual
// architecture metadata of the whole module (cudaFuncAttributes::ptxVersion).

#include <cub/util_device.cuh>

#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include <cuda/__runtime/api_wrapper.h>
#include <cuda/std/__cccl/assert.h>

#include <cstdio>

#define STRINGIFY_(...) #__VA_ARGS__
#define STRINGIFY(...)  STRINGIFY_(__VA_ARGS__)

__managed__ int actual_arch = 0;

__global__ void write_arch()
{
#ifdef __CUDA_ARCH__
  actual_arch = __CUDA_ARCH__;
#endif // __CUDA_ARCH__
}

int main()
{
#ifdef __CUDA_ARCH_LIST__
  std::printf("__CUDA_ARCH_LIST__=%s\n", STRINGIFY(__CUDA_ARCH_LIST__));
#endif // __CUDA_ARCH_LIST__

  std::printf("target_compute_capabilities=");
  for (const auto& target_cc : ::cuda::__target_compute_capabilities())
  {
    std::printf("%d.%d, ", target_cc.major_cap(), target_cc.minor_cap());
  }
  std::printf("\n");

  ::cuda::compute_capability cc{};
  _CCCL_TRY_RUNTIME_API(cub::detail::ptx_compute_cap, "ptx_compute_cap failed", cc);

  write_arch<<<1, 1>>>();
  _CCCL_TRY_RUNTIME_API(cudaDeviceSynchronize, "kernel launch failed");

  std::printf("ptx_compute_cap=%d.%d kernel's __CUDA_ARCH__=%d\n", cc.major_cap(), cc.minor_cap(), actual_arch);
  _CCCL_VERIFY(cc.get() * 10 == actual_arch, "ptx_compute_cap did not match the architecture actually executed");

  // Also run a CUB algorithm, which dispatches on ptx_compute_cap()
  thrust::device_vector<int> v(1000, 1);
  _CCCL_TRY_RUNTIME_API(cudaGetLastError, "device_vector construction failed");

  const int sum = thrust::reduce(v.begin(), v.end());
  _CCCL_TRY_RUNTIME_API(cudaGetLastError, "thrust::reduce failed");
  _CCCL_VERIFY(sum == 1000, "thrust::reduce did not return 1000");

  return 0;
}
