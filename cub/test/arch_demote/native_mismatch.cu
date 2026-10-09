// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Regression test for https://github.com/NVIDIA/cccl/pull/11440#issuecomment-5699615518, a counterexample to the fix
// for https://github.com/NVIDIA/cccl/issues/11403. __CUDA_ARCH_LIST__ contains PTX targets that CUDA may not
// actually load: when a native SASS binary compiled from a lower virtual architecture matches the real device
// exactly, CUDA runs that native binary rather than JIT-compiling a higher virtual-arch PTX also present in the
// fatbin, even though the higher virtual arch is <= the device's real compute capability.

#include <cub/util_device.cuh>

#include <cuda/__runtime/api_wrapper.h>
#include <cuda/std/__cccl/assert.h>

#include <cstdio>

// Do not use a __managed__ variable: it forces the module to be loaded when the CUDA context is created, which fails on
// GPUs this binary cannot run on, before main() can skip the test.
__global__ void kernel(int* actual_arch)
{
#ifdef __CUDA_ARCH__
  *actual_arch = __CUDA_ARCH__;
#endif
}

int main()
{
  // This binary contains sm_86 SASS, which runs natively on GPUs with compute capability 8.6 or higher within the 8.x
  // family. On any other GPU it would only run by JIT-compiling PTX, which fails if the driver is older than the CUDA
  // Toolkit.
  int device_cc_major = 0;
  int device_cc_minor = 0;
  const int device    = cub::CurrentDevice();
  _CCCL_TRY_RUNTIME_API(
    cudaDeviceGetAttribute,
    "cudaDeviceGetAttribute failed",
    &device_cc_major,
    cudaDevAttrComputeCapabilityMajor,
    device);
  _CCCL_TRY_RUNTIME_API(
    cudaDeviceGetAttribute,
    "cudaDeviceGetAttribute failed",
    &device_cc_minor,
    cudaDevAttrComputeCapabilityMinor,
    device);
  if (device_cc_major != 8 || device_cc_minor < 6)
  {
    std::printf("SKIPPED: this test requires a GPU with compute capability 8.6 or higher within 8.x\n");
    return 0;
  }

  ::cuda::compute_capability cc{};
  _CCCL_TRY_RUNTIME_API(cub::detail::ptx_compute_cap, "ptx_compute_cap failed", cc);

  int* actual_arch = nullptr;
  _CCCL_TRY_RUNTIME_API(cudaMallocManaged, "cudaMallocManaged failed", &actual_arch, sizeof(int), cudaMemAttachGlobal);
  *actual_arch = 0;

  kernel<<<1, 1>>>(actual_arch);
  _CCCL_TRY_RUNTIME_API(cudaDeviceSynchronize, "kernel launch failed");

  _CCCL_VERIFY(cc.get() * 10 == *actual_arch, "ptx_compute_cap did not match the architecture actually executed");

  _CCCL_TRY_RUNTIME_API(cudaFree, "cudaFree failed", actual_arch);
  return 0;
}
