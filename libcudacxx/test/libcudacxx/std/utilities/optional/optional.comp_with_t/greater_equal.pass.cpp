//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// template <class T, class U> constexpr bool operator>=(const optional<T>& x, const U& v);
// template <class T, class U> constexpr bool operator>=(const U& v, const optional<T>& x);

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

__host__ __device__ constexpr bool operator>=(const X& lhs, const X& rhs)
{
  return lhs.i_ >= rhs.i_;
}

template <class T>
__host__ __device__ constexpr void test()
{
  {
    using O = optional<X>;

    X val(2);
    O o1{}; // disengaged
    O o2{1}; // engaged
    O o3{val}; // engaged

    assert(!(o1 >= X(1)));
    assert((o2 >= X(1))); // equal
    assert((o3 >= X(1)));
    assert(!(o2 >= val));
    assert((o3 >= val)); // equal
    assert(!(o3 >= X(3)));

    assert((X(1) >= o1));
    assert((X(1) >= o2)); // equal
    assert(!(X(1) >= o3));
    assert((val >= o2));
    assert((val >= o3)); // equal
    assert((X(3) >= o3));
  }

  cuda::std::remove_reference_t<T> val1{42};
  cuda::std::remove_reference_t<T> val2{11};
  {
    using O = optional<T>;
    O o1(val1);
    assert(o1 >= val1);
    assert(!(val2 >= o1));
  }
  {
    using O = optional<const T>;
    O o1(val1);
    assert(o1 >= val1);
    assert(!(val2 >= o1));
  }
}

__host__ __device__ constexpr bool test()
{
  test<int>();
#ifdef CCCL_ENABLE_OPTIONAL_REF
  test<int&>();
#endif // CCCL_ENABLE_OPTIONAL_REF

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
