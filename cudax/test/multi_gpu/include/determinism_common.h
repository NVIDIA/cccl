//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/buffer>
#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/random>
#include <cuda/std/type_traits>

#include <algorithm>
#include <vector>

#include <c2h/catch2_test_helper.h>

// Large enough to force a multi-block, multi-pass device fold.
inline constexpr cuda::std::size_t large_values_per_rank = 100'000;

// The host folds left to right, the device folds as a tree.
inline constexpr double float_reference_tolerance = 1e-10;

[[nodiscard]] inline cuda::std::minstd_rand make_rng(const c2h::seed_t& seed)
{
  return cuda::std::minstd_rand(static_cast<cuda::std::minstd_rand::result_type>(seed.get()));
}

// `total_count` is the size of the whole global input. The bound derived from it keeps a fold over
// the whole input from overflowing. Signed overflow in the host reference is undefined behavior,
// and an infinite floating-point sum is the same for every fold order, so it would hide a
// determinism failure.
template <class T, class RNG>
[[nodiscard]] std::vector<T> make_random_values(cuda::std::size_t count, cuda::std::size_t total_count, RNG& rng)
{
  static_assert(cuda::std::is_arithmetic_v<T>);

  // The factor of two keeps the extreme case away from the exact limit.
  const auto bound = static_cast<T>(cuda::std::numeric_limits<T>::max() / static_cast<T>(2 * total_count));

  using distribution =
    cuda::std::conditional_t<cuda::std::is_floating_point_v<T>,
                             cuda::std::uniform_real_distribution<T>,
                             cuda::std::uniform_int_distribution<T>>;

  distribution dist{static_cast<T>(-bound), bound};
  std::vector<T> values(count);

  std::generate(values.begin(), values.end(), [&] {
    return dist(rng);
  });
  return values;
}

// A floating-point fold gives a different result on the host than on the device, so it needs a
// tolerance. Every other type folds exactly.
template <class Actual, class Expected>
void check_against_reference(const Actual& actual, const Expected& expected)
{
  if constexpr (cuda::std::is_floating_point_v<typename Actual::value_type>)
  {
    REQUIRE_APPROX_EQ_EPSILON(actual, expected, float_reference_tolerance);
  }
  else
  {
    REQUIRE_THAT(actual, Equals(expected));
  }
}
