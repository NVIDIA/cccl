//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: nvrtc

#include <cuda/mdspan>

#include "test_macros.h"

using ext_t = cuda::std::extents<int, 4>;

bool device_accessor_test()
{
  int* device_ptr;
  int* managed_ptr;
  assert(cudaMalloc(&device_ptr, 4) == cudaSuccess);
  assert(cudaMallocManaged(&managed_ptr, 4) == cudaSuccess);
  cuda::device_mdspan<int, ext_t> d_md{device_ptr, ext_t{}};
  cuda::device_mdspan<int, ext_t> m_md{managed_ptr, ext_t{}};
  unused(d_md);
  unused(m_md);
  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (assert(device_accessor_test());))
  return 0;
}
