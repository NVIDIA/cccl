//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>
// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: gcc-6, gcc-7, gcc-8
// UNSUPPORTED: nvcc-11.1

// template <typename T, size_t Size>
// constexpr auto to_array(T (&arr)[Size])
//    -> array<remove_cv_t<T>, Size>;

// template <typename T, size_t Size>
// constexpr auto to_array(T (&&arr)[Size])
//    -> array<remove_cv_t<T>, Size>;

#include <cuda/std/array>
#include <cuda/std/cassert>

#include "MoveOnly.h"
#include "test_macros.h"

__host__ __device__ constexpr bool tests()
{
  //  Test deduced type.
  {
    auto arr = cuda::std::to_array({1, 2, 3});
    ASSERT_SAME_TYPE(decltype(arr), cuda::std::array<int, 3>);
    assert(arr[0] == 1);
    assert(arr[1] == 2);
    assert(arr[2] == 3);
  }

#if !defined(TEST_COMPILER_MSVC_2017)
  {
    const long l1 = 42;
    auto arr      = cuda::std::to_array({1L, 4L, 9L, l1});
    ASSERT_SAME_TYPE(decltype(arr)::value_type, long);
    static_assert(arr.size() == 4, "");
    assert(arr[0] == 1);
    assert(arr[1] == 4);
    assert(arr[2] == 9);
    assert(arr[3] == l1);
  }
#endif // !TEST_COMPILER_MSVC_2017

  {
    auto arr = cuda::std::to_array("meow");
    ASSERT_SAME_TYPE(decltype(arr), cuda::std::array<char, 5>);
    assert(arr[0] == 'm');
    assert(arr[1] == 'e');
    assert(arr[2] == 'o');
    assert(arr[3] == 'w');
    assert(arr[4] == '\0');
  }

  {
    double source[3] = {4.0, 5.0, 6.0};
    auto arr         = cuda::std::to_array(source);
    ASSERT_SAME_TYPE(decltype(arr), cuda::std::array<double, 3>);
    assert(arr[0] == 4.0);
    assert(arr[1] == 5.0);
    assert(arr[2] == 6.0);
  }

  {
    double source[3] = {4.0, 5.0, 6.0};
    auto arr         = cuda::std::to_array(cuda::std::move(source));
    ASSERT_SAME_TYPE(decltype(arr), cuda::std::array<double, 3>);
    assert(arr[0] == 4.0);
    assert(arr[1] == 5.0);
    assert(arr[2] == 6.0);
  }

#if !defined(TEST_COMPILER_MSVC_2017)
  {
    MoveOnly source[] = {MoveOnly{0}, MoveOnly{1}, MoveOnly{2}};

    auto arr = cuda::std::to_array(cuda::std::move(source));
    ASSERT_SAME_TYPE(decltype(arr), cuda::std::array<MoveOnly, 3>);
    for (int i = 0; i < 3; ++i)
    {
      assert(arr[i].get() == i && source[i].get() == 0);
    }
  }
#endif // !TEST_COMPILER_MSVC_2017

#if defined(TEST_COMPILER_NVRTC) && defined(TEST_COMPILER_MSVC)
  // Test C99 compound literal.
  {
    auto arr = cuda::std::to_array((int[]){3, 4});
    ASSERT_SAME_TYPE(decltype(arr), cuda::std::array<int, 2>);
    assert(arr[0] == 3);
    assert(arr[1] == 4);
  }
#endif // !TEST_COMPILER_NVRTC && !TEST_COMPILER_MSVC

  //  Test explicit type.
  {
    auto arr = cuda::std::to_array<long>({1, 2, 3});
    ASSERT_SAME_TYPE(decltype(arr), cuda::std::array<long, 3>);
    assert(arr[0] == 1);
    assert(arr[1] == 2);
    assert(arr[2] == 3);
  }

  {
    struct A
    {
      int a;
      double b;
    };

    auto arr = cuda::std::to_array<A>({{3, .1}});
    ASSERT_SAME_TYPE(decltype(arr), cuda::std::array<A, 1>);
    assert(arr[0].a == 3);
    assert(arr[0].b == .1);
  }

  return true;
}

int main(int, char**)
{
  tests();
  static_assert(tests(), "");
  return 0;
}
