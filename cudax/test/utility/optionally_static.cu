//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__utility/optionally_static.cuh>

#include <cstddef>

#include <catch2/catch_test_macros.hpp>

namespace cudax = cuda::experimental;

TEST_CASE("optionally_static static value", "[utility]")
{
  constexpr cudax::optionally_static<::std::size_t(42), 0> v1;
  static_assert(v1.get() == 42);
  static_assert(v1 == v1);
  static_assert(v1 == 42UL);

  constexpr cudax::optionally_static<::std::size_t(43), 0> v2;
  static_assert(v2.get() == 43UL);
}

TEST_CASE("optionally_static dynamic value", "[utility]")
{
  cudax::optionally_static<::std::size_t(0), 0> v3;
  REQUIRE(v3.get() == 0);
  v3 = 44;
  REQUIRE(v3.get() == 44UL);
}

TEST_CASE("optionally_static multiplication", "[utility]")
{
  constexpr cudax::optionally_static<::std::size_t(42), 0> v1;
  constexpr cudax::optionally_static<::std::size_t(43), 0> v2;

  static_assert(v1 * v1 == 42UL * 42UL);
  static_assert(v1 * v2 == 42UL * 43UL);
  static_assert(v1 * 44 == 42UL * 44UL);
  static_assert(44 * v1 == 42UL * 44UL);

  cudax::optionally_static<::std::size_t(0), 0> v3;
  v3 = 44;
  REQUIRE(v1 * v3 == 42 * 44);
}

TEST_CASE("optionally_static reserved product", "[utility]")
{
  constexpr cudax::optionally_static<3, 18> v4;
  constexpr cudax::optionally_static<6, 18> v5;
  static_assert(v4 * v5 == 18UL);
  static_assert(v4 * v5 == (cudax::optionally_static<18, 18>(18)));
}
