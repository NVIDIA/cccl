//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <chrono>

// duration

// static constexpr duration max(); // noexcept after C++17

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/limits>

#include "../../rep.h"
#include "test_macros.h"

template <class D>
__host__ __device__ constexpr void test()
{
  static_assert(noexcept(cuda::std::chrono::duration_values<typename D::rep>::max()));

  {
    using DRep   = typename D::rep;
    DRep max_rep = cuda::std::chrono::duration_values<DRep>::max();
    assert(D::max().count() == max_rep);
  }
}

__host__ __device__ constexpr bool test()
{
  test<cuda::std::chrono::duration<int>>();
  test<cuda::std::chrono::duration<Rep>>();
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
