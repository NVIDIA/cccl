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

#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/random>
#include <cuda/std/type_traits>

#include <algorithm>
#include <vector>

#include <c2h/catch2_test_helper.h>

// Large enough to force a multi-block, multi-pass device fold.
inline constexpr cuda::std::size_t large_values_per_rank = 100'000;

[[nodiscard]] inline cuda::std::minstd_rand make_rng(const c2h::seed_t& seed)
{
  return cuda::std::minstd_rand(static_cast<cuda::std::minstd_rand::result_type>(seed.get()));
}

// `total_count` is the size of the whole global input. The bound derived from it keeps a fold over
// the whole input from overflowing.
//
// A floating-point type gets whole numbers only, small enough that every partial fold stays exact.
// So the host fold and the device fold agree bit for bit. Random real values do not have that
// property: the two fold orders round differently.
template <class T, class RNG>
[[nodiscard]] std::vector<T> make_random_values(cuda::std::size_t count, cuda::std::size_t total_count, RNG& rng)
{
  static_assert(cuda::std::is_arithmetic_v<T>);
  std::vector<T> values(count);

  if (count == 0 || total_count == 0)
  {
    return values;
  }

  if constexpr (cuda::std::is_floating_point_v<T>)
  {
    // Every integer up to 2^digits is exact. The factor of two keeps the extreme case away from
    // the limit.
    constexpr auto exact_limit = 1LL << cuda::std::numeric_limits<T>::digits;
    const auto bound           = exact_limit / (2 * static_cast<long long>(total_count));

    // The input is too large to fold exactly in this floating-point type
    REQUIRE(bound > 0);

    cuda::std::uniform_int_distribution<long long> dist{-bound, bound};

    std::generate(values.begin(), values.end(), [&] {
      return static_cast<T>(dist(rng));
    });
  }
  else
  {
    // The division stays in `size_t` so that a narrow `T` cannot wrap the denominator.
    const auto bound =
      static_cast<T>(static_cast<cuda::std::size_t>(cuda::std::numeric_limits<T>::max()) / (2 * total_count));

    // The input is too large to fold exactly in this integer type
    REQUIRE(bound > 0);

    // An unsigned type has no negative values to draw from.
    const T lower = cuda::std::is_signed_v<T> ? static_cast<T>(-bound) : static_cast<T>(0);
    cuda::std::uniform_int_distribution<T> dist{lower, bound};

    std::generate(values.begin(), values.end(), [&] {
      return dist(rng);
    });
  }
  return values;
}
