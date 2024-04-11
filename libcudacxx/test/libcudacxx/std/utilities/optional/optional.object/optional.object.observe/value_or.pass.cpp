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

// template <class U> constexpr T optional<T>::value_or(U&& v) &&;

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "test_macros.h"

using cuda::std::in_place;
using cuda::std::in_place_t;
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
  __host__ __device__ constexpr X(X&& x)
      : i_(x.i_)
  {
    x.i_ = 0;
  }
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

__host__ __device__ constexpr int test()
{
  {
    optional<X> opt(in_place, 2);
    Y y(3);
    assert(cuda::std::move(opt).value_or(y) == 2);
    assert(*opt == 0);
  }
  {
    optional<X> opt(in_place, 2);
    assert(cuda::std::move(opt).value_or(Y(3)) == 2);
    assert(*opt == 0);
  }
  {
    optional<X> opt;
    Y y(3);
    assert(cuda::std::move(opt).value_or(y) == 3);
    assert(!opt);
  }
  {
    optional<X> opt;
    assert(cuda::std::move(opt).value_or(Y(3)) == 4);
    assert(!opt);
  }
  return 0;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017
  static_assert(test() == 0);
#endif

  return 0;
}
