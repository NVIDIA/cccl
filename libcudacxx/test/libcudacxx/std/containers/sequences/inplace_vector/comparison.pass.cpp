//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++11

#include <cuda/std/cassert>
#include <cuda/std/initializer_list>
#include <cuda/std/inplace_vector>
#include <cuda/std/type_traits>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class T>
__host__ __device__ constexpr void test_equality()
{
  using inplace_vector = cuda::std::inplace_vector<T, 42ull>;

  inplace_vector vec{T(1), T(42), T(1337), T(0)};
  inplace_vector other_vec{T(0), T(1), T(2), T(3), T(4)};

  auto res_equality = vec == vec;
  static_assert(cuda::std::is_same<decltype(res_equality), bool>::value, "");
  assert(res_equality);

  auto res_inequality = vec != other_vec;
  static_assert(cuda::std::is_same<decltype(res_inequality), bool>::value, "");
  assert(res_inequality);
}

template <class T>
__host__ __device__ constexpr void test_relation()
{
  using inplace_vector = cuda::std::inplace_vector<T, 42ull>;

  inplace_vector vec{T(0), T(1), T(1), T(3), T(4)};
  inplace_vector other_vec{T(0), T(1), T(2), T(3), T(4)};

  auto res_less = vec < other_vec;
  static_assert(cuda::std::is_same<decltype(res_less), bool>::value, "");
  assert(res_less);

  auto res_less_equal = vec <= other_vec;
  static_assert(cuda::std::is_same<decltype(res_less_equal), bool>::value, "");
  assert(res_less_equal);

  auto res_greater = vec > other_vec;
  static_assert(cuda::std::is_same<decltype(res_greater), bool>::value, "");
  assert(!res_greater);

  auto res_greater_equal = vec >= other_vec;
  static_assert(cuda::std::is_same<decltype(res_greater_equal), bool>::value, "");
  assert(!res_greater_equal);
}

template <class T>
__host__ __device__ constexpr void test()
{
  test_equality<T>();
  test_relation<T>();
}

__host__ __device__ constexpr bool test()
{
  test<int>();
  test<Trivial>();

  if (!cuda::std::__libcpp_is_constant_evaluated())
  {
    test<NonTrivial>();
    test<NonTrivialDestructor>();
    test<ThrowingDefaultConstruct>();
  }

  return true;
}

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_IS_CONSTANT_EVALUATED)
  static_assert(test(), "");
#endif // _CCCL_BUILTIN_IS_CONSTANT_EVALUATED

  return 0;
}
