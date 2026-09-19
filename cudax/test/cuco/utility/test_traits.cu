//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__cuco/detail/bitwise_compare.cuh>
#include <cuda/experimental/__cuco/detail/utility/traits.cuh>

#include <testing.cuh>

namespace custom
{
struct payload
{
  double value;
};
} // namespace custom

CUDAX_CUCO_DECLARE_BITWISE_COMPARABLE(float);
CUDAX_CUCO_DECLARE_BITWISE_COMPARABLE(custom::payload);

C2H_TEST("cuco bitwise comparison opt-in", "[traits]")
{
  static_assert(::cuda::is_bitwise_comparable_v<float>);
  static_assert(::cuda::is_bitwise_comparable_v<const float>);
  static_assert(::cuda::is_bitwise_comparable_v<volatile float>);
  static_assert(::cuda::is_bitwise_comparable_v<const volatile float>);
  static_assert(::cuda::is_bitwise_comparable_v<custom::payload>);
  static_assert(::cuda::is_bitwise_comparable_v<const custom::payload>);
  static_assert(!::cuda::is_bitwise_comparable_v<double>);

  REQUIRE(::cuda::experimental::cuco::detail::__bitwise_compare(1.0f, 1.0f));
  REQUIRE(!::cuda::experimental::cuco::detail::__bitwise_compare(1.0f, 2.0f));
  REQUIRE(::cuda::experimental::cuco::detail::__bitwise_compare(custom::payload{1.0}, custom::payload{1.0}));
  REQUIRE(!::cuda::experimental::cuco::detail::__bitwise_compare(custom::payload{1.0}, custom::payload{2.0}));
}
