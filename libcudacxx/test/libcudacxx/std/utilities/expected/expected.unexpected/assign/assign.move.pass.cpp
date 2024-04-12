//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// constexpr unexpected& operator=(unexpected&&) = default;

#include <cuda/std/cassert>
#include <cuda/std/expected>
#include <cuda/std/utility>

#include "test_macros.h"

struct Error
{
  int i;
  __host__ __device__ constexpr Error(int ii)
      : i(ii)
  {}
  __host__ __device__ constexpr Error& operator=(Error&& other)
  {
    i       = other.i;
    other.i = 0;
    return *this;
  }
};

__host__ __device__ constexpr bool test()
{
  cuda::std::unexpected<Error> unex1(4);
  cuda::std::unexpected<Error> unex2(5);
  unex1 = cuda::std::move(unex2);
  assert(unex1.error().i == 5);
  assert(unex2.error().i == 0);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
