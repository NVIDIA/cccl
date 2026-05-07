// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// This is a compile-only test that checks whether cub can handle a future (unknown) architecture. Currently, works only
// with nvcc.

#if defined(__NVCC__) && __CUDACC_VER_MAJOR__ >= 13
#  define TEST_CUDA_ARCH 9900

// Replace __CUDA_ARCH__ only in device code.
#  if defined(__CUDA_ARCH__)
#    undef __CUDA_ARCH__
#    define __CUDA_ARCH__ TEST_CUDA_ARCH
#  endif // __CUDA_ARCH__

// __CUDA_ARCH_LIST__ is defined in both host and device code.
#  undef __CUDA_ARCH_LIST__
#  define __CUDA_ARCH_LIST__ TEST_CUDA_ARCH

// We need to undefine the include guard before including <cuda_runtime.h>. It was defined by us in CMake.
#  undef __CUDA_RUNTIME_H__
#  include <cuda_runtime.h>
#endif // __NVCC__ && is at least 13.0

#include <cub/device/device_reduce.cuh>

#include <cuda/std/cstddef>

cudaError_t compile_only_fn(void* tmp_storage, cuda::std::size_t tmp_storage_size, int* in, int* out, int nitems)
{
  return cub::DeviceReduce::Sum(tmp_storage, tmp_storage_size, in, out, nitems);
}

int main() {}
