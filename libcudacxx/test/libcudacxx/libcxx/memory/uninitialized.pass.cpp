//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__memory/uninitialized.h>
#include <cuda/std/cassert>
#include <cuda/std/memory>
#include <cuda/std/type_traits>

#include "test_macros.h"

TEST_FUNC void test_size_and_alignment()
{
  using arr_t = cuda::__uninitialized<int[4]>;
  static_assert(sizeof(arr_t) >= 4 * sizeof(int));
  static_assert(alignof(arr_t) == alignof(int));
}

TEST_FUNC void test_custom_alignment()
{
  using arr_t = cuda::__uninitialized<int[4], 32>;
  static_assert(alignof(arr_t) == 32);
}

TEST_FUNC void test_element_access()
{
  cuda::__uninitialized<int[4]> arr{};
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
  using arr_t = cuda::__uninitialized<int[4]>;
  static_assert(cuda::std::is_same<decltype(cuda::std::declval<arr_t&>().data()), int*>::value);
  static_assert(cuda::std::is_same<decltype(cuda::std::declval<const arr_t&>().data()), const int*>::value);
  arr_t arr{};
  assert(arr.data() != nullptr);
}

struct not_default_constructible_t
{
  int x;

  TEST_FUNC explicit not_default_constructible_t(int v)
      : x(v)
  {}
};

TEST_FUNC void test_alias()
{
  using arr_t = cuda::__uninitialized<not_default_constructible_t[4]>;
  static_assert(
    cuda::std::is_same<decltype(cuda::std::declval<arr_t&>().alias()), not_default_constructible_t(&)[4]>::value);
  static_assert(cuda::std::is_same<decltype(cuda::std::declval<const arr_t&>().alias()),
                                   const not_default_constructible_t(&)[4]>::value);

  arr_t arr{};
  auto& a = arr.alias();
  for (int i = 0; i < 4; ++i)
  {
    ::new (static_cast<void*>(&a[i])) not_default_constructible_t(i + 1);
  }
  const arr_t& carr = arr;
  for (int i = 0; i < 4; ++i)
  {
    assert(carr.alias()[i].x == i + 1);
    assert(&arr[i] == &a[i]);
  }
}

template <class T, class = void>
inline constexpr bool has_subscript = false;

template <class T>
inline constexpr bool has_subscript<T, cuda::std::void_t<decltype(cuda::std::declval<T&>()[0])>> = true;

TEST_FUNC void test_scalar()
{
  using val_t = cuda::__uninitialized<not_default_constructible_t>;
  static_assert(sizeof(val_t) == sizeof(not_default_constructible_t));
  static_assert(alignof(val_t) == alignof(not_default_constructible_t));
  static_assert(cuda::std::is_same<val_t::value_type, not_default_constructible_t>::value);
  static_assert(cuda::std::is_same<decltype(cuda::std::declval<val_t&>().data()), not_default_constructible_t*>::value);
  static_assert(
    cuda::std::is_same<decltype(cuda::std::declval<const val_t&>().data()), const not_default_constructible_t*>::value);
  static_assert(
    cuda::std::is_same<decltype(cuda::std::declval<val_t&>().alias()), not_default_constructible_t&>::value);
  static_assert(
    cuda::std::is_same<decltype(cuda::std::declval<const val_t&>().alias()), const not_default_constructible_t&>::value);

  // the subscript operator is only available for array types
  static_assert(!has_subscript<val_t>);
  static_assert(has_subscript<cuda::__uninitialized<int[4]>>);
  static_assert(has_subscript<const cuda::__uninitialized<int[4]>>);

  val_t val{};
  ::new (static_cast<void*>(val.data())) not_default_constructible_t(42);
  const val_t& cval = val;
  assert(val.alias().x == 42);
  assert(cval.alias().x == 42);
  assert(&val.alias() == val.data());
}

TEST_FUNC void test_scalar_custom_alignment()
{
  using val_t = cuda::__uninitialized<int, 16>;
  static_assert(sizeof(val_t) == 16);
  static_assert(alignof(val_t) == 16);
}

TEST_FUNC void test_no_value_initialization()
{
  struct with_default_member_t
  {
    int x = 42;
  };
  cuda::__uninitialized<with_default_member_t[4]> arr{};
  const auto* raw = reinterpret_cast<const unsigned char*>(arr.__data);
  for (size_t i = 0; i < sizeof(arr); ++i)
  {
    assert(raw[i] == 0);
  }
}

TEST_FUNC void test_basic_integers()
{
  cuda::__uninitialized<char[4]> arr_char{};
  cuda::__uninitialized<short[4]> arr_short{};
  cuda::__uninitialized<int[4]> arr_int{};
  cuda::__uninitialized<long[4]> arr_long{};
  cuda::__uninitialized<long long[4]> arr_longlong{};

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
  cuda::__uninitialized<float[4]> arr_float{};
  cuda::__uninitialized<double[4]> arr_double{};

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
  test_alias();
  test_scalar();
  test_scalar_custom_alignment();
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
