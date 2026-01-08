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

bool host_accessor_runtime_fail()
{
  int* device_ptr = nullptr;
  assert(cudaMalloc(&device_ptr, 4) == cudaSuccess);
  using ext_t = cuda::std::extents<int, 4>;
  cuda::host_mdspan<int, ext_t> h_md{device_ptr, ext_t{}};
  unused(h_md);
  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (assert(host_accessor_runtime_fail());))
  return 0;
}
