//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// constexpr optional() noexcept;

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>

#include "archetypes.h"
#include "test_macros.h"

using cuda::std::optional;

template <class Opt>
__host__ __device__ constexpr void test()
{
  static_assert(cuda::std::is_nothrow_default_constructible<Opt>::value, "");
  static_assert(cuda::std::is_trivially_destructible<Opt>::value
                  == cuda::std::is_trivially_destructible<typename Opt::value_type>::value,
                "");
  {
    Opt opt{};
    assert(static_cast<bool>(opt) == false);
  }
  {
    const Opt opt{};
    assert(static_cast<bool>(opt) == false);
  }
}

__host__ __device__ constexpr bool test()
{
  test<optional<int>>();
  test<optional<int*>>();
  test<optional<ImplicitTypes::NoCtors>>();
  test<optional<NonTrivialTypes::NoCtors>>();
  test<optional<NonConstexprTypes::NoCtors>>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  test<optional<NonLiteralTypes::NoCtors>>();

  return 0;
}
