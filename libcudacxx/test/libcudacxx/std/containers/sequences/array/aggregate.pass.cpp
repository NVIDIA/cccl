//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Make sure cuda::std::array is an aggregate type.
// We can only check this in C++17 and above, because we don't have the
// trait before that.
// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: gcc-6

#include <cuda/std/array>
#include <cuda/std/type_traits>

#include "test_macros.h"

TEST_NV_DIAG_SUPPRESS(cuda_demote_unsupported_floating_point)

template <typename T>
__host__ __device__ void check_aggregate()
{
  static_assert(cuda::std::is_aggregate<cuda::std::array<T, 0>>::value, "");
  static_assert(cuda::std::is_aggregate<cuda::std::array<T, 1>>::value, "");
  static_assert(cuda::std::is_aggregate<cuda::std::array<T, 2>>::value, "");
  static_assert(cuda::std::is_aggregate<cuda::std::array<T, 3>>::value, "");
  static_assert(cuda::std::is_aggregate<cuda::std::array<T, 4>>::value, "");
}

struct Empty
{};
struct Trivial
{
  int i;
  int j;
};
struct NonTrivial
{
  int i;
  int j;
  __host__ __device__ NonTrivial(NonTrivial const&) {}
};

int main(int, char**)
{
  check_aggregate<char>();
  check_aggregate<int>();
  check_aggregate<long>();
  check_aggregate<float>();
  check_aggregate<double>();
  check_aggregate<long double>();
  check_aggregate<Empty>();
  check_aggregate<Trivial>();
  check_aggregate<NonTrivial>();

  return 0;
}
