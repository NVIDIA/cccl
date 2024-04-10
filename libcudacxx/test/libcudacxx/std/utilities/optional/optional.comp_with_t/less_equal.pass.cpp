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

// template <class T, class U> constexpr bool operator<=(const optional<T>& x, const U& v);
// template <class T, class U> constexpr bool operator<=(const U& v, const optional<T>& x);

#include <cuda/std/cassert>
#include <cuda/std/optional>

#include "test_macros.h"

using cuda::std::optional;

struct X
{
  int i_;

  __host__ __device__ constexpr X(int i)
      : i_(i)
  {}
};

__host__ __device__ constexpr bool operator<=(const X& lhs, const X& rhs)
{
  return lhs.i_ <= rhs.i_;
}

__host__ __device__ constexpr bool test()
{
  {
    typedef X T;
    typedef optional<T> O;

    constexpr T val(2);
    O o1; // disengaged
    O o2{1}; // engaged
    O o3{val}; // engaged

    assert((o1 <= T(1)));
    assert((o2 <= T(1))); // equal
    assert(!(o3 <= T(1)));
    assert((o2 <= val));
    assert((o3 <= val)); // equal
    assert((o3 <= T(3)));

    assert(!(T(1) <= o1));
    assert((T(1) <= o2)); // equal
    assert((T(1) <= o3));
    assert(!(val <= o2));
    assert((val <= o3)); // equal
    assert(!(T(3) <= o3));
  }
  {
    using O = optional<int>;
    O o1(42);
    assert(o1 <= 42l);
    assert(!(101l <= o1));
  }
  {
    using O = optional<const int>;
    O o1(42);
    assert(o1 <= 42);
    assert(!(101 <= o1));
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2017
#  if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  static_assert(test());
#  endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
#endif

  return 0;
}
