//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: gcc-8, gcc-9

// template<class F, class... Args>
// concept predicate;

#include <cuda/std/concepts>

#include "test_macros.h"
#if TEST_STD_VER > 2017

__host__ __device__ constexpr bool check_subsumption(cuda::std::regular_invocable auto)
{
  return false;
}

// clang-format off
template<class F>
requires cuda::std::predicate<F> && true
__host__ __device__ constexpr bool check_subsumption(F)
{
  return true;
}
// clang-format on
struct not_predicate
{
  __host__ __device__ constexpr void operator()() const {}
};

struct predicate
{
  __host__ __device__ constexpr bool operator()() const
  {
    return true;
  }
};

static_assert(!check_subsumption(not_predicate{}), "");
static_assert(check_subsumption(predicate{}), "");

#endif // TEST_STD_VER > 2017

int main(int, char**)
{
  return 0;
}
