// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/std/array>
#include <cuda/std/bit>
#include <cuda/std/cstdint>

#include <cuda/experimental/__cuco/bloom_filter_policy.cuh>

#include <testing.cuh>

namespace cuco = cuda::experimental::cuco;

using policy12 = cuco::bloom_filter_policy<int, cuda::hash<int, cuda::hash_algorithm::xxhash_64>, unsigned int, 8, 12>;

// This independent reference assigns the first four words two salts each and
// the remaining four words one salt each. Zero hashes also exercise bit collisions.
constexpr cuda::std::array<unsigned int, 8> reference_pattern(cuda::std::uint32_t hash)
{
  constexpr cuda::std::array<cuda::std::uint32_t, 12> salts{
    0x47B6137BU,
    0x44974D91U,
    0x8824AD5BU,
    0xA2B7289DU,
    0x705495C7U,
    0x2DF1424BU,
    0x9EFC4947U,
    0x5C6BFB31U,
    0xB24BCDFFU,
    0xB6843D6DU,
    0x6DB04543U,
    0x3A12EFDDU};
  cuda::std::array<unsigned int, 8> result{};
  int salt = 0;
  for (int word = 0; word < 8; ++word)
  {
    for (int bit = 0; bit < (word < 4 ? 2 : 1); ++bit)
    {
      result[word] |= 1U << ((salts[salt++] * hash) >> 27);
    }
  }
  return result;
}

static_assert(policy12{}.array_pattern<0, 8>(0U) == reference_pattern(0U));
static_assert(policy12{}.array_pattern<0, 8>(0x12345678U) == reference_pattern(0x12345678U));

template <int Horizontal, int Vertical>
void check_layout(cuda::std::uint32_t hash)
{
  const auto expected = reference_pattern(hash);
  policy12 policy;
  constexpr int loops = 8 / (Horizontal * Vertical);
  // Instantiate both iterations for layouts requiring an outer loop.
  for (int lane = 0; lane < Horizontal; ++lane)
  {
    const auto first = policy.template array_pattern<0, Horizontal, Vertical>(hash, lane);
    for (int word = 0; word < Vertical; ++word)
    {
      REQUIRE(first[word] == expected[lane * Vertical + word]);
    }
    if constexpr (loops == 2)
    {
      const auto second = policy.template array_pattern<1, Horizontal, Vertical>(hash, lane);
      for (int word = 0; word < Vertical; ++word)
      {
        REQUIRE(second[word] == expected[(Horizontal + lane) * Vertical + word]);
      }
    }
  }
}

C2H_TEST("bloom_filter fingerprint distribution is independent of layout", "[bloom_filter]")
{
  for (const auto hash : {0U, 1U, 0x12345678U, 0xFFFFFFFFU})
  {
    const auto pattern = policy12{}.array_pattern<0, 8>(hash);
    REQUIRE(pattern == reference_pattern(hash));
    for (int word = 0; word < 8; ++word)
    {
      REQUIRE(pattern[word] != 0U);
      REQUIRE(cuda::std::popcount(pattern[word]) <= (word < 4 ? 2 : 1));
    }
    check_layout<8, 1>(hash);
    check_layout<4, 2>(hash);
    check_layout<2, 4>(hash);
    check_layout<4, 1>(hash);
    check_layout<2, 2>(hash);
    const auto first  = policy12{}.array_pattern<0, 4>(hash);
    const auto second = policy12{}.array_pattern<1, 4>(hash);
    for (int word = 0; word < 4; ++word)
    {
      REQUIRE(first[word] == pattern[word]);
      REQUIRE(second[word] == pattern[word + 4]);
    }
  }
}
