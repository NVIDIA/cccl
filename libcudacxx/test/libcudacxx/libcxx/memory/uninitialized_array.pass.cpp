//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// XFAIL: enable-tile
// nvbug6077640: error: Internal Compiler Error (tile codegen): "call to unknown tile builtin function!"

#include <cuda/__memory/uninitialized_array.h>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

TEST_FUNC void test_size_and_alignment()
{
  using arr_t = cuda::__uninitialized_array<int, 4>;
  static_assert(sizeof(arr_t) >= 4 * sizeof(int), "");
  static_assert(alignof(arr_t) == alignof(int), "");
}

TEST_FUNC void test_custom_alignment()
{
  using arr_t = cuda::__uninitialized_array<int, 4, 32>;
  static_assert(alignof(arr_t) == 32, "");
}

TEST_FUNC void test_element_access()
{
  cuda::__uninitialized_array<int, 4> arr{};
  arr[0] = 10;
  arr[1] = 20;
  arr[2] = 30;
  arr[3] = 40;
  assert(arr[0] == 10);
  assert(arr[1] == 20);
  assert(arr[2] == 30);
  assert(arr[3] == 40);
}

TEST_FUNC void test_data_pointer_const_correctness()
{
  using arr_t = cuda::__uninitialized_array<int, 4>;
  static_assert(cuda::std::is_same<decltype(cuda::std::declval<arr_t&>().data()), int*>::value, "");
  static_assert(cuda::std::is_same<decltype(cuda::std::declval<const arr_t&>().data()), const int*>::value, "");
  arr_t arr{};
  assert(arr.data() != nullptr);
}

TEST_FUNC void test_no_value_initialization()
{
  struct with_default_member_t
  {
    int x = 42;
  };
  cuda::__uninitialized_array<with_default_member_t, 4> arr{};
  const auto* raw = reinterpret_cast<const unsigned char*>(arr.__data);
  for (size_t i = 0; i < sizeof(arr); ++i)
  {
    assert(raw[i] == 0);
  }
}

TEST_FUNC void test_basic_integers()
{
  cuda::__uninitialized_array<char, 4> arr_char{};
  cuda::__uninitialized_array<short, 4> arr_short{};
  cuda::__uninitialized_array<int, 4> arr_int{};
  cuda::__uninitialized_array<long, 4> arr_long{};
  cuda::__uninitialized_array<long long, 4> arr_longlong{};

  arr_char[0] = 'a';
  assert(arr_char[0] == 'a');
  arr_short[0] = 123;
  assert(arr_short[0] == 123);
  arr_int[0] = 42;
  assert(arr_int[0] == 42);
  arr_long[0] = 100L;
  assert(arr_long[0] == 100L);
  arr_longlong[0] = 999LL;
  assert(arr_longlong[0] == 999LL);
}

TEST_FUNC void test_floating_point()
{
  cuda::__uninitialized_array<float, 4> arr_float{};
  cuda::__uninitialized_array<double, 4> arr_double{};

  arr_float[0] = 3.14f;
  assert(arr_float[0] == 3.14f);
  arr_double[0] = 2.718;
  assert(arr_double[0] == 2.718);
}

TEST_FUNC bool test()
{
  test_size_and_alignment();
  test_custom_alignment();
  test_element_access();
  test_data_pointer_const_correctness();
  test_no_value_initialization();
  test_basic_integers();
  test_floating_point();
  return true;
}

int main(int, char**)
{
  assert(test());
  return 0;
}
