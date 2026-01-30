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

__device__ bool host_accessor_runtime_fail()
{
  int array[] = {1, 2, 3, 4};
  using ext_t = cuda::std::extents<int, 4>;
  cuda::host_mdspan<int, ext_t> h_md{array, ext_t{}};
  unused(h_md);
  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_DEVICE, (assert(host_accessor_runtime_fail());))
  return 0;
}
