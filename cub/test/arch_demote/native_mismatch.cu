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

__managed__ int actual_arch = 0;

__global__ void kernel()
{
#ifdef __CUDA_ARCH__
  actual_arch = __CUDA_ARCH__;
#endif
}

int main()
{
  ::cuda::compute_capability cc{};
  _CCCL_TRY_RUNTIME_API(cub::detail::ptx_compute_cap, "ptx_compute_cap failed", cc);

  kernel<<<1, 1>>>();
  _CCCL_TRY_RUNTIME_API(cudaDeviceSynchronize, "kernel launch failed");

  _CCCL_VERIFY(cc.get() * 10 == actual_arch, "ptx_compute_cap did not match the architecture actually executed");

  return 0;
}
