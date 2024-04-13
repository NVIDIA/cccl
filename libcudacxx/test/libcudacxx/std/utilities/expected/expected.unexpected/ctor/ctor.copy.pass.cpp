//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// constexpr unexpected(const unexpected&) = default;

#include <cuda/std/cassert>
#include <cuda/std/expected>

#include "test_macros.h"

struct Error
{
  int i;
  __host__ __device__ constexpr Error(int ii)
      : i(ii)
  {}
};

__host__ __device__ constexpr bool test()
{
  const cuda::std::unexpected<Error> unex(5);
  auto unex2 = unex;
  assert(unex2.error().i == 5);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
