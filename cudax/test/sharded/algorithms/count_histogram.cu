//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Correctness of the read-only sharded algorithms `count` / `count_if`
 *        and `histogram_even` against host references, over multiple places —
 *        including on contiguous (VMM-backed) arrays, where read-only
 *        algorithms remain available (only size-mutating ones refuse).
 */

#include <cuda/experimental/sharded.cuh>

#include <cstdio>
#include <stdexcept>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::cuda_try;
using cuda::experimental::places::place_group;

namespace
{
struct is_multiple_of_3
{
  __host__ __device__ bool operator()(long long x) const
  {
    return x % 3 == 0;
  }
};

void test_count(place_group& group)
{
  const size_t n = 1000003;
  auto data      = sharded_array<long long>::allocate(group, n);
  iota(group, data, 0LL); // 0 .. n-1

  // Multiples of 3 in [0, n): ceil(n / 3)
  EXPECT(count_if(group, data, is_multiple_of_3{}) == (n + 2) / 3);

  // count == count_if with equality
  EXPECT(count(group, data, 42LL) == 1UL);
  EXPECT(count(group, data, static_cast<long long>(n)) == 0UL); // absent value
  EXPECT(count(group, data, -1LL) == 0UL);

  fill(group, data, 7LL);
  EXPECT(count(group, data, 7LL) == n);
  EXPECT(count(group, data, 8LL) == 0UL);

  // Empty array
  sharded_array<long long> empty;
  EXPECT(count_if(group, empty, is_multiple_of_3{}) == 0UL);
  EXPECT(count(group, empty, 0LL) == 0UL);
}

void test_count_on_contiguous(place_group& group)
{
  // Read-only algorithms stay available on contiguous arrays
  const size_t n = (1 << 20) + 37;
  auto data      = sharded_array<long long>::allocate_contiguous(group, n);
  iota(group, data, 1LL); // 1 .. n

  EXPECT(count(group, data, 5LL) == 1UL);
  // Multiples of 3 in [1, n]: floor(n / 3)
  EXPECT(count_if(group, data, is_multiple_of_3{}) == n / 3);
}

void test_histogram(place_group& group)
{
  const size_t n = 262147;
  auto data      = sharded_array<long long>::allocate(group, n);
  iota(group, data, 0LL); // 0 .. n-1

  const int num_bins          = 8;
  const long long lower_level = 0;
  const long long upper_level = 262144; // bin width 32768; 3 samples fall outside

  const auto counts = histogram_even(group, data, num_bins, lower_level, upper_level);
  EXPECT(counts.size() == static_cast<size_t>(num_bins));

  // Host reference
  ::std::vector<long long> host(n);
  data.copy_to_host(host.data());
  ::std::vector<size_t> ref(num_bins, 0);
  const long long width = (upper_level - lower_level) / num_bins;
  size_t in_range       = 0;
  for (size_t i = 0; i < n; i++)
  {
    if (host[i] >= lower_level && host[i] < upper_level)
    {
      ref[static_cast<size_t>((host[i] - lower_level) / width)]++;
      in_range++;
    }
  }
  size_t total = 0;
  for (int b = 0; b < num_bins; b++)
  {
    EXPECT(counts[b] == ref[b]);
    total += counts[b];
  }
  EXPECT(total == in_range);

  // Empty array: all-zero histogram
  sharded_array<long long> empty;
  const auto zero = histogram_even(group, empty, 4, 0LL, 100LL);
  EXPECT(zero.size() == 4UL);
  for (const auto c : zero)
  {
    EXPECT(c == 0UL);
  }

  // Invalid arguments are refused at the API boundary
  bool threw = false;
  try
  {
    (void) histogram_even(group, data, 0, 0LL, 100LL);
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);

  threw = false;
  try
  {
    (void) histogram_even(group, data, 4, 100LL, 100LL);
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);
}
} // namespace

int main()
{
  cuda_try(cuInit(0));
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group::by_locality_domains();

  test_count(group);
  test_histogram(group);

  if (contiguous_backing_supported())
  {
    test_count_on_contiguous(group);
  }
  else
  {
    printf("VMM not supported on this device, skipping contiguous-array tests.\n");
  }

  return 0;
}
