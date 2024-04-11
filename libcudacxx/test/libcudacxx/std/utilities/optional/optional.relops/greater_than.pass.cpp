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

// template <class T, class U> constexpr bool operator> (const optional<T>& x, const optional<U>& y);

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

__host__ __device__ constexpr bool operator>(const X& lhs, const X& rhs)
{
  return lhs.i_ > rhs.i_;
}

__host__ __device__ constexpr bool test()
{
  {
    typedef optional<X> O;

    O o1; // disengaged
    O o2; // disengaged
    O o3{1}; // engaged
    O o4{2}; // engaged
    O o5{1}; // engaged

    assert(!(o1 > o1));
    assert(!(o1 > o2));
    assert(!(o1 > o3));
    assert(!(o1 > o4));
    assert(!(o1 > o5));

    assert(!(o2 > o1));
    assert(!(o2 > o2));
    assert(!(o2 > o3));
    assert(!(o2 > o4));
    assert(!(o2 > o5));

    assert((o3 > o1));
    assert((o3 > o2));
    assert(!(o3 > o3));
    assert(!(o3 > o4));
    assert(!(o3 > o5));

    assert((o4 > o1));
    assert((o4 > o2));
    assert((o4 > o3));
    assert(!(o4 > o4));
    assert((o4 > o5));

    assert((o5 > o1));
    assert((o5 > o2));
    assert(!(o5 > o3));
    assert(!(o5 > o4));
    assert(!(o5 > o5));
  }
  {
    using O1 = optional<int>;
    using O2 = optional<long>;
    O1 o1(42);
    assert(o1 > O2(1));
    assert(!(O2(42) > o1));
  }
  {
    using O1 = optional<int>;
    using O2 = optional<const int>;
    O1 o1(42);
    assert(o1 > O2(1));
    assert(!(O2(42) > o1));
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
