//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__mdspan/host_device_mdspan.h>

#include "test_macros.h"

__global__ void kernel(int* host_ptr)
{
#if defined(__CUDA_ARCH__)
  using ext_t = cuda::std::extents<int, 4>;
  cuda::device_mdspan<int, ext_t> d_md{host_ptr, ext_t{}};
  unused(d_md[0]);
#endif
}

int array[] = {1, 2, 3, 4};

int main(int, char**)
{
#if !defined(__CUDA_ARCH__)
  // int array[] = {1, 2, 3, 4};
  // int* host_ptr = (int*) malloc(4 * 4);
  //  assert(cudaMalloc(&device_ptr, 4) == cudaSuccess);
  //  using ext_t = cuda::std::extents<int, 4>;
  //  cuda::host_mdspan<int, ext_t> h_md{device_ptr, ext_t{}};
  //  unused(h_md[0]);
  kernel<<<1, 1>>>(array);
#endif
  return 0;
}
