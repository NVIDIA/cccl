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

int main(int, char**)
{
  int array[] = {1, 2, 3, 4};
  using ext_t = cuda::std::extents<int, 4>;
  cuda::host_mdspan<int, ext_t> h_md{array, ext_t{}};
  cuda::device_mdspan<int, ext_t> d_md{device_array, ext_t{}};
  cuda::managed_mdspan<int, ext_t> m_md{managed_array, ext_t{}};
#if !defined(__CUDA_ARCH__)
  unused(h_md[0]);
#else
  unused(d_md[0]);
#endif
  unused(m_md[0]);
  return 0;
}
