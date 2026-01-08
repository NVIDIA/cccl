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

bool managed_accessor_test()
{
  int* device_ptr;
  assert(cudaMalloc(&device_ptr, 4) == cudaSuccess);
  using ext_t = cuda::std::extents<int, 4>;
  cuda::managed_mdspan<int, ext_t> d_md{device_ptr, ext_t{}};
  unused(d_md);
  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (assert(managed_accessor_test());))
  return 0;
}
