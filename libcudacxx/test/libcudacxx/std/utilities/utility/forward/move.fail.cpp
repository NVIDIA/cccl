//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// This test should pass in C++03 with Clang extensions because Clang does
// not implicitly delete the copy constructor when move constructors are
// defaulted using extensions.

// XFAIL: c++03

// test move

#include <cuda/std/cassert>
#include <cuda/std/utility>

struct move_only
{
  __host__ __device__ move_only() {}
  move_only(move_only&&)            = default;
  move_only& operator=(move_only&&) = default;
};

__host__ __device__ move_only source()
{
  return move_only();
}
__host__ __device__ const move_only csource()
{
  return move_only();
}

__host__ __device__ void test(move_only) {}

int main(int, char**)
{
  const move_only ca = move_only();
  // expected-error@+1 {{call to implicitly-deleted copy constructor of 'move_only'}}
  test(std::move(ca));

  return 0;
}
