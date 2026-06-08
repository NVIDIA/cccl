// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <thrust/device_vector.h>
#include <thrust/tabulate.h>

#include <cuda/random>
#include <cuda/std/algorithm>
#include <cuda/std/cmath>
#include <cuda/std/cstdint>
#include <cuda/std/random>

#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include <nvbench_helper.cuh>

namespace
{
enum class pattern_kind : int
{
  random = 0,
  quantized_random,
  relu_quantized,
  tie_heavy,
  pivot_tie
};

[[nodiscard]] pattern_kind string_to_pattern(const std::string& pattern)
{
  if (pattern == "random")
  {
    return pattern_kind::random;
  }
  if (pattern == "quantized_random")
  {
    return pattern_kind::quantized_random;
  }
  if (pattern == "relu_quantized")
  {
    return pattern_kind::relu_quantized;
  }
  if (pattern == "tie_heavy")
  {
    return pattern_kind::tie_heavy;
  }
  if (pattern == "pivot_tie")
  {
    return pattern_kind::pivot_tie;
  }
  throw std::runtime_error("Invalid Pattern axis value: " + pattern);
}

template <int MaxSegmentSize, int K>
[[nodiscard]] thrust::device_vector<float>
gen_data(int num_segments, pattern_kind pattern, const cuda::std::int64_t* d_seg_sizes)
{
  const auto num_keys = static_cast<std::size_t>(num_segments) * static_cast<std::size_t>(MaxSegmentSize);
  auto d_keys         = thrust::device_vector<float>{num_keys, thrust::no_init};

  // gt_count == "greater-than count": number of 2.0 values placed at the tail of each segment's live region.
  constexpr int gt_count = cuda::std::max(1, cuda::std::min(K / 4, MaxSegmentSize / 8));

  thrust::tabulate(d_keys.begin(), d_keys.end(), [pattern, d_seg_sizes] __device__(std::size_t idx) -> float {
    auto quantize = [](float base) -> float {
      const auto r         = cuda::std::rint(base);
      const auto scaled_fr = cuda::std::rint((base - r) * 32.0f);
      return r + (scaled_fr / 32.0f);
    };

    auto random_value = [](unsigned long long idx) -> float {
      cuda::pcg64 rng(42);
      rng.discard(idx);
      cuda::std::normal_distribution<float> normal(0.f, 1.f);
      return normal(rng);
    };

    const auto j = static_cast<int>(idx % MaxSegmentSize);
    switch (pattern)
    {
      //               ##
      //              ####
      //            ########
      //          ############
      //        ################
      //     ######################
      //  ##############################
      //  ------------------------------
      //  -3             0             3
      case pattern_kind::random:
        return random_value(idx);

      //                |
      //                |
      //            |   |   |
      //        |   |   |   |   |
      //    |   |   |   |   |   |   |
      //  ----------------------------
      //  -3            0            3
      case pattern_kind::quantized_random:
        return quantize(random_value(idx));

      //  |
      //  |
      //  |
      //  |
      //  |
      //  |   |
      //  |   |   |
      //  |   |   |   |   |
      //  |   |   |   |   |   |   |
      //  ----------------------------
      //  0                          3
      case pattern_kind::relu_quantized:
        return quantize(cuda::std::max(random_value(idx), 0.f));

      //  |   |   |   |   |   |   |   |
      //  |   |   |   |   |   |   |   |
      //  |   |   |   |   |   |   |   |
      //  --------------------------------
      //  0/64                      63/64
      case pattern_kind::tie_heavy:
        return static_cast<float>(j % 64) / 64.f;

      //  |
      //  |
      //  |
      //  |
      //  |
      //  |
      //  |
      //  |                         |
      //  ----------------------------
      //  1.0                       2.0
      case pattern_kind::pivot_tie: {
        const auto seg_size = static_cast<int>(d_seg_sizes[idx / MaxSegmentSize]);
        return (j >= seg_size - gt_count) ? 2.f : 1.f;
      }
      default:
        _CCCL_UNREACHABLE();
    }
  });

  return d_keys;
}
} // namespace

const std::vector<std::string> valid_patterns = {
  "random", "quantized_random", "relu_quantized", "tie_heavy", "pivot_tie"};

using key_type_list = nvbench::type_list<float>;

using max_segment_size_list = nvbench::enum_type_list< //
  512,
  1024,
  2048,
  4096,
  8192
#if 0 // need these, waiting for implementation to catch up
  ,
  16384,
  32768,
  65536,
  131072,
  262144,
  524288,
  1048576
#endif
  >;

using k_list = nvbench::enum_type_list<512, 1024, 2048>;
