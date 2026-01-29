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

__device__ void basic_mdspan_access_test()
{
  using ext_t = cuda::std::extents<int, 4>;
  __shared__ int smem[4];
  [[maybe_unused]] cuda::shared_memory_mdspan<int, ext_t> md{smem, ext_t{}};
  unused(md[0]);
  asm volatile("" : : "l"((size_t) smem) : "memory");
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_DEVICE, (basic_mdspan_access_test();))
  return 0;
}
