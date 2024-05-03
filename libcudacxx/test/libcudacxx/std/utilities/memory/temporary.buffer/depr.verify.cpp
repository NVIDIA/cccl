//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// Ensure allocator<void> is deprecated

#include <cuda/std/__algorithm_>

__host__ __device__ void test()
{
  auto a = cuda::std::get_temporary_buffer<int>(1); // expected-warning {{'get_temporary_buffer<int>' is deprecated}}
  cuda::std::return_temporary_buffer(a.first); // expected-warning {{'return_temporary_buffer<int>' is deprecated}}
}

int main(int, char**)
{
  return 0;
}
