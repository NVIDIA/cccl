//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/mdspan>

#include "test_macros.h"

__device__ int device_array[]              = {1, 2, 3, 4};
__device__ __managed__ int managed_array[] = {1, 2, 3, 4};

using ext_t = cuda::std::extents<int, 4>;

__host__ __device__ void basic_mdspan_access_test()
{
  int array[] = {1, 2, 3, 4};
  cuda::host_mdspan<int, ext_t> h_md{array, ext_t{}};
  cuda::device_mdspan<int, ext_t> d_md{device_array, ext_t{}};
  cuda::managed_mdspan<int, ext_t> m_md{managed_array, ext_t{}};
  NV_IF_ELSE_TARGET(NV_IS_HOST, (unused(h_md[0]);), unused(d_md[0]);)
  unused(m_md[0]);
}

__global__ void test_kernel(cuda::host_mdspan<int, ext_t> md)
{
  cuda::host_mdspan<int, ext_t> h_md2{md};
  unused(h_md2);
}

#if !_CCCL_COMPILER(NVRTC)

void host_mdspan_to_kernel_test()
{
  int array[] = {1, 2, 3, 4};
  cuda::host_mdspan<int, ext_t> h_md{array, ext_t{}};
  test_kernel<<<1, 1>>>(h_md);
  assert(cudaDeviceSynchronize() == cudaSuccess);
}

#endif // !_CCCL_COMPILER(NVRTC)

int main(int, char**)
{
  basic_mdspan_access_test();
#if !_CCCL_COMPILER(NVRTC)
  NV_IF_TARGET(NV_IS_HOST, (host_mdspan_to_kernel_test();))
#endif // !_CCCL_COMPILER(NVRTC)
  return 0;
}
