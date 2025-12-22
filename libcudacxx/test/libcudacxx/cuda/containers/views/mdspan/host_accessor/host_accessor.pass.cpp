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

bool host_accessor_test()
{
  int array[] = {1, 2, 3, 4};
  int* h_ptr;
  assert(cudaMallocHost(&h_ptr, 4) == cudaSuccess);
  cuda::host_mdspan<int, ext_t> h_md{array, ext_t{}};
  cuda::host_mdspan<int, ext_t> h_md2{h_ptr, ext_t{}};
  unused(h_md);
  unused(h_md2);
  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (assert(host_accessor_test());))
  return 0;
}
