//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <cuda/std/optional>

// template <class U> constexpr T optional<T>::value_or(U&& v) const&;

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::optional;

struct Y
{
  int i_;

  __host__ __device__ constexpr Y(int i)
      : i_(i)
  {}
};

struct X
{
  int i_;

  __host__ __device__ constexpr X(int i)
      : i_(i)
  {}
  __host__ __device__ constexpr X(const Y& y)
      : i_(y.i_)
  {}
  __host__ __device__ constexpr X(Y&& y)
      : i_(y.i_ + 1)
  {}
  __host__ __device__ friend constexpr bool operator==(const X& x, const X& y)
  {
    return x.i_ == y.i_;
  }
};

int main(int, char**)
{
#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  {
    constexpr optional<X> opt(2);
    constexpr Y y(3);
    static_assert(opt.value_or(y) == 2, "");
  }
  {
    constexpr optional<X> opt(2);
    static_assert(opt.value_or(Y(3)) == 2, "");
  }
  {
    constexpr optional<X> opt;
    constexpr Y y(3);
    static_assert(opt.value_or(y) == 3, "");
  }
  {
    constexpr optional<X> opt;
    static_assert(opt.value_or(Y(3)) == 4, "");
  }
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  {
    const optional<X> opt(2);
    const Y y(3);
    assert(opt.value_or(y) == 2);
  }
  {
    const optional<X> opt(2);
    assert(opt.value_or(Y(3)) == 2);
  }
  {
    const optional<X> opt;
    const Y y(3);
    assert(opt.value_or(y) == 3);
  }
  {
    const optional<X> opt;
    assert(opt.value_or(Y(3)) == 4);
  }

  return 0;
}
