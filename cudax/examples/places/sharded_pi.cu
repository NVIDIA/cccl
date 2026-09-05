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
 * @brief Monte Carlo pi over sharded arrays: three tiers composing through
 *        one binding.
 *
 * The pipeline is vendor tier -> generic tier -> native tier, with nothing
 * connecting the stages but the sharded structures themselves:
 *
 *   generate_uniform(x/y)   — cuRAND tier (random.cuh, opt-in)
 *   zip_transform(inside)   — in-circle test per sample
 *   reduce(inside)          — hit count, then pi = 4 * hits / n
 *
 * Because generation is sharding-invariant (bitwise: sample k is a pure
 * function of (seed, k)), the estimate is REPRODUCIBLE ACROSS SHARDINGS:
 * this example computes pi twice with different shard boundaries and checks
 * the hit counts match exactly. (The sum of 0/1 float flags is integer-exact,
 * so even the reduction is grouping-independent.)
 *
 * Caveat, stated on purpose: a production Monte Carlo would FUSE generation
 * into the consumer (device-API `curand_init` inside the functor) and never
 * materialize the samples. This example materializes them deliberately to
 * show the API seams; the fused spelling is the planned iterator-descriptor
 * relaxation recorded in concepts.cuh's v1-simplifications note (a lazy
 * random input view whose dereference is the positional mapping).
 */

#include <cuda/experimental/sharded.cuh>

#include <cstdio>

#include <cuda/experimental/__sharded/random.cuh> // opt-in vendor tier

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
struct in_unit_circle
{
  __host__ __device__ float operator()(float x, float y) const
  {
    return (x * x + y * y <= 1.0f) ? 1.0f : 0.0f;
  }
};

float estimate_hits(place_group& group, const ::std::vector<size_t>& shard_sizes)
{
  auto x      = sharded_array<float>::allocate(group, shard_sizes);
  auto y      = sharded_array<float>::allocate(group, shard_sizes);
  auto inside = sharded_array<float>::allocate(group, shard_sizes);

  generate_uniform(x, default_envs(x), /*seed=*/1234);
  generate_uniform(y, default_envs(y), /*seed=*/5678);

  zip_transform(inside, in_unit_circle{}, x, y);
  return reduce(inside, ::cuda::std::plus<float>{}, 0.0f);
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));
  place_group group{make_locality_domain_grid()};
  const size_t n = 1u << 24;
  printf("sharded Monte Carlo pi: n = %zu samples over %zu places\n", n, group.size());

  // Sharding A: even split. Sharding B: deliberately uneven (odd boundary).
  ::std::vector<size_t> even(group.size(), n / group.size());
  even[0] += n % group.size();
  ::std::vector<size_t> uneven = even;
  if (group.size() > 1)
  {
    const size_t shift = 104729; // prime: odd boundary on purpose
    uneven[0] += shift;
    uneven[1] -= shift;
  }

  const float hits_a = estimate_hits(group, even);
  const float hits_b = estimate_hits(group, uneven);

  printf("  even sharding:   hits = %.0f  ->  pi ~ %.9f\n", hits_a, 4.0 * hits_a / n);
  printf("  uneven sharding: hits = %.0f  ->  pi ~ %.9f\n", hits_b, 4.0 * hits_b / n);

  if (hits_a != hits_b)
  {
    printf("FAILED: the estimate is not sharding-invariant\n");
    return 1;
  }
  printf("estimate is bitwise reproducible across shardings\n");
  return 0;
}
