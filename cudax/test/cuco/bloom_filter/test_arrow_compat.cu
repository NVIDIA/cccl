// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Precomputed Apache Arrow Block-Split Bloom Filter bitsets from cuCollections.
#include <thrust/equal.h>
#include <thrust/execution_policy.h>

#include <cuda/functional>
#include <cuda/memory_pool>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/__cuco/bloom_filter.cuh>

#include <stdexcept>
#include <vector>

#include <testing.cuh>

#include <c2h/vector.h>

using cuda::std::int32_t;
using cuda::std::int64_t;
using cuda::std::uint32_t;

namespace
{
template <typename Key>
c2h::device_vector<uint32_t> get_arrow_filter_reference_bitset()
{
  static std::vector<std::vector<uint32_t>> const reference_bitsets{
    {
      3017764846,
      4219371383,
      4160077310,
      3214786543,
      4020088765,
      4294437885,
      2013200345,
      2550116063,
      855631359,
      4290436829,
      2884632042,
      1592483646,
      4281695998,
      2080111551,
      3220060030,
      4021279731,
    }, // type = int32, blocks = 2, num_keys = 100
    {
      860053560,  186397876,  1518788617, 2013987426, 545522943,  79856155,   103371656,  20265733,
      2168586373, 1210138712, 2437452036, 1342183988, 1107366672, 3560981000, 2184221186, 1661010032,
      2317009736, 1442875878, 1116227467, 3458613792, 114398528,  679658134,  206734656,  340863450,
      2220104352, 141846788,  948331524,  2344943952, 4030989912, 3239203139, 2941256193, 4035057968,
    }, // type = int64, blocks = 4, num_keys = 50
    {
      3807057303, 3207519405, 2508188120, 1491024175, 2073585514, 2094743110, 2533287591, 691662424,
      1498889215, 2069126314, 2270481639, 796401059,  1961968732, 3512881027, 3162306144, 2277085974,
      3477648628, 1090385857, 4035761415, 1165385841, 4047856262, 2297893848, 902599838,  418175153,
      1437192944, 3673877288, 1536198910, 98677451,   3620189521, 3794688342, 3625373537, 3550967313,
      2119503598, 1805574667, 4076413870, 2999897588, 3050286944, 4146882307, 3459690182, 167235913,
      2078961096, 1863964920, 1408130860, 4190644775, 532451008,  1563872186, 2529714129, 465761275,
      3161649891, 4204002248, 3931628891, 3251515903, 1421507581, 3849056446, 1748476671, 4223388125,
      1627644727, 2717076288, 2992639576, 3864567831, 190096788,  1885360347, 724608293,  2768994330,
    } // type = float, blocks = 8, num_keys = 200
  };

  if constexpr (cuda::std::is_same_v<Key, int32_t>)
  {
    return c2h::device_vector<uint32_t>(reference_bitsets[0].begin(), reference_bitsets[0].end());
  }
  else if constexpr (cuda::std::is_same_v<Key, int64_t>)
  {
    return c2h::device_vector<uint32_t>(reference_bitsets[1].begin(), reference_bitsets[1].end());
  }
  else if constexpr (cuda::std::is_same_v<Key, float>)
  {
    return c2h::device_vector<uint32_t>(reference_bitsets[2].begin(), reference_bitsets[2].end());
  }
  else
  {
    throw std::invalid_argument("Reference bitsets available for int32, int64, float only.\n\n");
  }
}

template <typename Key>
cuda::std::pair<size_t, size_t> get_arrow_filter_test_settings()
{
  static std::vector<cuda::std::pair<size_t, size_t>> const test_settings = {
    {2, 100}, // type = int32, blocks = 2, num_keys = 100
    {4, 50}, // type = int64, blocks = 4, num_keys = 50
    {8, 200} // type = float, blocks = 8, num_keys = 200
  };

  if constexpr (cuda::std::is_same_v<Key, int32_t>)
  {
    return test_settings[0];
  }
  else if constexpr (cuda::std::is_same_v<Key, int64_t>)
  {
    return test_settings[1];
  }
  else if constexpr (cuda::std::is_same_v<Key, float>)
  {
    return test_settings[2];
  }
  else
  {
    throw std::invalid_argument("Test settings available for int32, int64, float only.\n\n");
  }
}
} // namespace

C2H_TEST("bloom_filter Arrow-compatible bitset", "[bloom_filter]", c2h::type_list<int32_t, int64_t, float>)
{
  using key_type                    = c2h::get<0, TestType>;
  const auto [num_blocks, num_keys] = get_arrow_filter_test_settings<key_type>();
  const cuda::stream stream{cuda::device_ref{0}};
  auto mr = cuda::device_default_memory_pool(cuda::device_ref{0});
  cuda::experimental::cuco::bloom_filter<key_type> filter{stream, mr, num_blocks};
  std::vector<key_type> host_keys(num_keys);
  for (cuda::std::size_t i = 0; i < num_keys; ++i)
  {
    host_keys[i] = static_cast<key_type>(i + 1);
  }
  const c2h::device_vector<key_type> keys(host_keys.begin(), host_keys.end());
  filter.add(stream, keys.begin(), keys.end());
  const auto reference = get_arrow_filter_reference_bitset<key_type>();
  REQUIRE(filter.num_words() == reference.size());
  REQUIRE(thrust::equal(
    thrust::cuda::par.on(stream.get()), filter.data(), filter.data() + filter.num_words(), reference.begin()));
}
