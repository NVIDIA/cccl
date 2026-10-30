//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// cuda::std::tuple<> is constructible from an empty tuple or array, including
// host std::tuple<> and std::array<T, 0>, for every cv-ref qualification.

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#if _CCCL_HAS_HOST_STD_LIB()
#  include <array>
#  include <tuple>
#endif // _CCCL_HAS_HOST_STD_LIB()

#include "test_macros.h"

template <class Source>
TEST_FUNC constexpr bool test_construct()
{
  Source source{};
  const Source const_source{};
  cuda::std::tuple<> from_lvalue(source);
  cuda::std::tuple<> from_const_lvalue(const_source);
  cuda::std::tuple<> from_rvalue(Source{});
  cuda::std::tuple<> from_const_rvalue(static_cast<const Source&&>(const_source));
  assert(from_lvalue == cuda::std::tuple<>{});
  assert(from_const_lvalue == cuda::std::tuple<>{});
  assert(from_rvalue == cuda::std::tuple<>{});
  assert(from_const_rvalue == cuda::std::tuple<>{});
  return true;
}

template <class Source>
TEST_FUNC void test_cvref()
{
  using Tuple = cuda::std::tuple<>;

  static_assert(cuda::std::is_constructible_v<Tuple, Source>);
  static_assert(cuda::std::is_constructible_v<Tuple, const Source>);
  static_assert(cuda::std::is_constructible_v<Tuple, volatile Source>);
  static_assert(cuda::std::is_constructible_v<Tuple, const volatile Source>);
  static_assert(cuda::std::is_constructible_v<Tuple, Source&>);
  static_assert(cuda::std::is_constructible_v<Tuple, const Source&>);
  static_assert(cuda::std::is_constructible_v<Tuple, volatile Source&>);
  static_assert(cuda::std::is_constructible_v<Tuple, const volatile Source&>);
  static_assert(cuda::std::is_constructible_v<Tuple, Source&&>);
  static_assert(cuda::std::is_constructible_v<Tuple, const Source&&>);
  static_assert(cuda::std::is_constructible_v<Tuple, volatile Source&&>);
  static_assert(cuda::std::is_constructible_v<Tuple, const volatile Source&&>);

  volatile Source volatile_source{};
  const volatile Source const_volatile_source{};
  Tuple from_volatile_lvalue(volatile_source);
  Tuple from_const_volatile_lvalue(const_volatile_source);
  Tuple from_volatile_rvalue(static_cast<volatile Source&&>(volatile_source));
  Tuple from_const_volatile_rvalue(static_cast<const volatile Source&&>(const_volatile_source));
  assert(from_volatile_lvalue == Tuple{});
  assert(from_const_volatile_lvalue == Tuple{});
  assert(from_volatile_rvalue == Tuple{});
  assert(from_const_volatile_rvalue == Tuple{});
}

TEST_FUNC void test_host_std()
{
#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 test_construct<std::tuple<>>();
                 test_construct<std::array<int, 0>>();
                 test_cvref<std::tuple<>>();
                 test_cvref<std::array<int, 0>>();
               }))
#endif // _CCCL_HAS_HOST_STD_LIB()
}

int main(int, char**)
{
  test_construct<cuda::std::tuple<>>();
  test_construct<cuda::std::array<int, 0>>();
  static_assert(test_construct<cuda::std::tuple<>>());
  static_assert(test_construct<cuda::std::array<int, 0>>());

  test_cvref<cuda::std::tuple<>>();
  test_cvref<cuda::std::array<int, 0>>();
  test_host_std();
  return 0;
}
