//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef COMMON_CUH
#define COMMON_CUH

#include <cuda/std/algorithm>
#include <cuda/std/cstddef>
#include <cuda/std/string_view>

#include <stdio.h>

class KernelName
{
  static constexpr cuda::std::size_t max_size = 128;

  char name_[max_size]; // The name buffer.

public:
  __host__ __device__ KernelName(cuda::std::string_view name)
  {
    assert(name.size() < max_size);

    // Copy the name.
    cuda::std::copy_n(name.data(), name.size(), name_);

    // Zero terminate the string.
    name_[name.size()] = '\0';
  }

  // Returns the stored name.
  __host__ __device__ const char* get() const
  {
    return name_;
  }
};

__device__ void say_hello(uint3 from_tindex, const KernelName& kernel_name)
{
  const auto this_tindex = cuda::gpu_thread.index(cuda::block);

  printf("[%u, %u]: Hello from thread [%u, %u] launched as %s!\n",
         this_tindex.x,
         this_tindex.y,
         from_tindex.x,
         from_tindex.y,
         kernel_name.get());

  // Wait for all threads in block to print the output.
  __syncthreads();

  // Print additional new line once.
  if (this_tindex.x == 0 && this_tindex.y == 0)
  {
    printf("\n");
  }
}

#endif // COMMON_CUH
