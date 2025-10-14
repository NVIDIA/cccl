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

__device__ void basic_mdspan_access_test()
{
  [[maybe_unused]] cuda::device_mdspan<int, ext_t> d_md{device_array, ext_t{}};
  [[maybe_unused]] cuda::device_mdspan<int, ext_t> m_md{managed_array, ext_t{}};
  NV_IF_TARGET(NV_IS_DEVICE, unused(d_md[0]);)
  unused(m_md[0]);
}

__global__ void test_kernel(cuda::device_mdspan<int, ext_t> md)
{
  [[maybe_unused]] cuda::device_mdspan<int, ext_t> h_md2{md};
  unused(h_md2);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_DEVICE, (basic_mdspan_access_test();))
  return 0;
}
