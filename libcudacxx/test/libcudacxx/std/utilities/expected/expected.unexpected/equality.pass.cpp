//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// template<class E2>
// friend constexpr bool operator==(const unexpected& x, const unexpected<E2>& y);
//
// Mandates: The expression x.error() == y.error() is well-formed and its result is convertible to bool.
//
// Returns: x.error() == y.error().

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/utility>

#include "test_macros.h"

struct Error
{
  int i;
#if TEST_STD_VER > 2017
  __host__ __device__ friend constexpr bool operator==(const Error&, const Error&) = default;
#else
  __host__ __device__ friend constexpr bool operator==(const Error& lhs, const Error& rhs) noexcept
  {
    return lhs.i == rhs.i;
  }
  __host__ __device__ friend constexpr bool operator!=(const Error& lhs, const Error& rhs) noexcept
  {
    return lhs.i != rhs.i;
  }
#endif
};

__host__ __device__ constexpr bool test()
{
  cuda::std::unexpected<Error> unex1(Error{2});
  cuda::std::unexpected<Error> unex2(Error{3});
  cuda::std::unexpected<Error> unex3(Error{2});
  assert(unex1 == unex3);
  assert(unex1 != unex2);
  assert(unex2 != unex3);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");
  return 0;
}
