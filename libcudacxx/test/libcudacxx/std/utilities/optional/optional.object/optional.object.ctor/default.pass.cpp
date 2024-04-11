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

// constexpr optional() noexcept;

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "archetypes.h"
#include "test_macros.h"

using cuda::std::optional;

template <class Opt>
__host__ __device__ constexpr bool test_constexpr()
{
  static_assert(cuda::std::is_nothrow_default_constructible<Opt>::value, "");
  static_assert(cuda::std::is_trivially_destructible<Opt>::value, "");
  static_assert(cuda::std::is_trivially_destructible<typename Opt::value_type>::value, "");

  Opt opt;
  assert(static_cast<bool>(opt) == false);

  struct test_constexpr_ctor : public Opt
  {
    __host__ __device__ constexpr test_constexpr_ctor() {}
  };

  return true;
}

template <class Opt>
__host__ __device__ constexpr bool test()
{
  static_assert(cuda::std::is_nothrow_default_constructible<Opt>::value, "");
  static_assert(!cuda::std::is_trivially_destructible<Opt>::value, "");
  static_assert(!cuda::std::is_trivially_destructible<typename Opt::value_type>::value, "");
  {
    Opt opt;
    assert(static_cast<bool>(opt) == false);
  }
  {
    const Opt opt;
    assert(static_cast<bool>(opt) == false);
  }

  struct test_constexpr_ctor : public Opt
  {
    __host__ __device__ constexpr test_constexpr_ctor() {}
  };

  return true;
}

int main(int, char**)
{
  test_constexpr<optional<int>>();
  test_constexpr<optional<int*>>();
#if !defined(TEST_COMPILER_MSVC) || TEST_STD_VER >= 2017
  test_constexpr<optional<ImplicitTypes::NoCtors>>();
  test_constexpr<optional<NonTrivialTypes::NoCtors>>();
  test_constexpr<optional<NonConstexprTypes::NoCtors>>();
#endif
#ifndef TEST_COMPILER_ICC
  test<optional<NonLiteralTypes::NoCtors>>();
#endif // TEST_COMPILER_ICC

#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  static_assert(test_constexpr<optional<int>>(), "");
  static_assert(test_constexpr<optional<int*>>(), "");
#  if !defined(TEST_COMPILER_MSVC) || TEST_STD_VER >= 2017
  static_assert(test_constexpr<optional<ImplicitTypes::NoCtors>>(), "");
  static_assert(test_constexpr<optional<NonTrivialTypes::NoCtors>>(), "");
  static_assert(test_constexpr<optional<NonConstexprTypes::NoCtors>>(), "");
#  endif
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))

  return 0;
}
