//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/utility>

// template <class T1, class T2> struct pair

// template <pair-like Pair> EXPLICIT constexpr pair(Pair&& p);

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#if _CCCL_HAS_HOST_STD_LIB()
#  include <tuple>
#  include <utility>
#endif // _CCCL_HAS_HOST_STD_LIB()

#include "copy_move_types.h"
#include "test_macros.h"

template <bool Expected, class Tuple, class... Args>
TEST_FUNC void assert_constref_constructible_single()
{
  static_assert(cuda::std::is_constructible_v<Tuple, Args&...> == Expected);
  static_assert(cuda::std::is_constructible_v<Tuple, Args...> == Expected);
  static_assert(cuda::std::is_constructible_v<Tuple, const Args&...> == Expected);
  static_assert(cuda::std::is_constructible_v<Tuple, const Args...> == Expected);
  if constexpr (sizeof...(Args) == 2)
  {
    static_assert(cuda::std::is_constructible_v<Tuple, cuda::std::pair<Args...>&> == Expected);
    static_assert(cuda::std::is_constructible_v<Tuple, const cuda::std::pair<Args...>&> == Expected);
    static_assert(cuda::std::is_constructible_v<Tuple, cuda::std::pair<Args...>> == Expected);
    static_assert(cuda::std::is_constructible_v<Tuple, const cuda::std::pair<Args...>> == Expected);

#if _CCCL_HAS_HOST_STD_LIB()
    static_assert(cuda::std::is_constructible_v<Tuple, std::pair<Args...>&> == Expected);
    static_assert(cuda::std::is_constructible_v<Tuple, const std::pair<Args...>&> == Expected);
    static_assert(cuda::std::is_constructible_v<Tuple, std::pair<Args...>> == Expected);
    static_assert(cuda::std::is_constructible_v<Tuple, const std::pair<Args...>> == Expected);
#endif // _CCCL_HAS_HOST_STD_LIB()
  }

  static_assert(cuda::std::is_constructible_v<Tuple, cuda::std::tuple<Args...>&> == Expected);
  static_assert(cuda::std::is_constructible_v<Tuple, const cuda::std::tuple<Args...>&> == Expected);
  static_assert(cuda::std::is_constructible_v<Tuple, cuda::std::tuple<Args...>> == Expected);
  static_assert(cuda::std::is_constructible_v<Tuple, const cuda::std::tuple<Args...>> == Expected);

#if _CCCL_HAS_HOST_STD_LIB()
  static_assert(cuda::std::is_constructible_v<Tuple, std::tuple<Args...>&> == Expected);
  static_assert(cuda::std::is_constructible_v<Tuple, const std::tuple<Args...>&> == Expected);
  static_assert(cuda::std::is_constructible_v<Tuple, std::tuple<Args...>> == Expected);
  static_assert(cuda::std::is_constructible_v<Tuple, const std::tuple<Args...>> == Expected);
#endif // _CCCL_HAS_HOST_STD_LIB()
}

template <bool Expected, class Tuple, class... Args>
TEST_FUNC void assert_mutref_constructible_single()
{
  static_assert(cuda::std::is_constructible_v<Tuple, Args...> == Expected);
  if constexpr (sizeof...(Args) == 2)
  {
    static_assert(cuda::std::is_constructible_v<Tuple, cuda::std::pair<Args...>> == Expected);
  }
  static_assert(cuda::std::is_constructible_v<Tuple, cuda::std::tuple<Args...>> == Expected);

#if _CCCL_HAS_HOST_STD_LIB()
  if constexpr (sizeof...(Args) == 2)
  {
    static_assert(cuda::std::is_constructible_v<Tuple, std::pair<Args...>> == Expected);
  }
  static_assert(cuda::std::is_constructible_v<Tuple, std::tuple<Args...>> == Expected);
#endif // _CCCL_HAS_HOST_STD_LIB()
}

TEST_FUNC void assert_normal_constructible()
{
  assert_constref_constructible_single<true, cuda::std::tuple<const int&>, int>();
#if defined(_CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY)
  assert_constref_constructible_single<false, cuda::std::tuple<const int&>, long>();
#endif // _CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY

  assert_constref_constructible_single<true, cuda::std::tuple<const int&, const int&>, int, int>();
#if defined(_CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY)
  assert_constref_constructible_single<false, cuda::std::tuple<const int&, const int&>, long, int>();
  assert_constref_constructible_single<false, cuda::std::tuple<const int&, const int&>, long, long>();
  assert_constref_constructible_single<false, cuda::std::tuple<const int&, const int&>, int, long>();
#endif // _CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY

  assert_mutref_constructible_single<true, cuda::std::tuple<int&&>, int>();
#if defined(_CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY)
  assert_mutref_constructible_single<false, cuda::std::tuple<int&&>, long>();
#endif // _CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY

  assert_mutref_constructible_single<true, cuda::std::tuple<int&&, int&&>, int, int>();
#if defined(_CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY)
  assert_mutref_constructible_single<false, cuda::std::tuple<int&&, int&&>, long, int>();
  assert_mutref_constructible_single<false, cuda::std::tuple<int&&, int&&>, long, long>();
  assert_mutref_constructible_single<false, cuda::std::tuple<int&&, int&&>, int, long>();
#endif // _CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY
}

struct LvalueTempConverter
{
  TEST_FUNC operator int() &;
  TEST_FUNC operator int&&() &&;
  TEST_FUNC operator const int&() &&;
};

template <class Tuple, class... Args>
TEST_FUNC void assert_lvalue_temp_converter_single()
{
  static_assert(cuda::std::is_constructible_v<Tuple, Args..., LvalueTempConverter>);
  if constexpr (sizeof...(Args) == 1)
  {
    static_assert(cuda::std::is_constructible_v<Tuple, cuda::std::pair<Args..., LvalueTempConverter>>);
  }
  static_assert(cuda::std::is_constructible_v<Tuple, cuda::std::tuple<Args..., LvalueTempConverter>>);

#if _CCCL_HAS_HOST_STD_LIB()
  if constexpr (sizeof...(Args) == 1)
  {
    static_assert(cuda::std::is_constructible_v<Tuple, std::pair<Args..., LvalueTempConverter>>);
  }
  static_assert(cuda::std::is_constructible_v<Tuple, std::tuple<Args..., LvalueTempConverter>>);
#endif // _CCCL_HAS_HOST_STD_LIB()

// GCC 13 fails to properly consider the LvalueTempConverter
#if defined(_CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY) && !TEST_COMPILER(GCC, <, 14)
  static_assert(!cuda::std::is_constructible_v<Tuple, Args&..., LvalueTempConverter&>);
  static_assert(!cuda::std::is_constructible_v<Tuple, cuda::std::tuple<Args&..., LvalueTempConverter&>>);
  if constexpr (sizeof...(Args) == 1)
  {
    static_assert(!cuda::std::is_constructible_v<Tuple, cuda::std::pair<Args&..., LvalueTempConverter&>>);
  }

#  if _CCCL_HAS_HOST_STD_LIB()
  static_assert(!cuda::std::is_constructible_v<Tuple, std::tuple<Args&..., LvalueTempConverter&>>);
  if constexpr (sizeof...(Args) == 1)
  {
    static_assert(!cuda::std::is_constructible_v<Tuple, std::pair<Args&..., LvalueTempConverter&>>);
  }
#  endif // _CCCL_HAS_HOST_STD_LIB()
#endif // _CCCL_BUILTIN_REFERENCE_CONSTRUCTS_FROM_TEMPORARY && !TEST_COMPILER(GCC, <, 14)
}

TEST_FUNC void assert_lvalue_temp_converter()
{
  assert_lvalue_temp_converter_single<cuda::std::tuple<const int&>>();
  assert_lvalue_temp_converter_single<cuda::std::tuple<int&&>>();

  assert_lvalue_temp_converter_single<cuda::std::tuple<const int&, const int&>, const int&>();
}

TEST_FUNC void test()
{
  assert_normal_constructible();
  assert_lvalue_temp_converter();
}

int main(int, char**)
{
  test();

  return 0;
}
