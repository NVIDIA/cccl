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
 * @brief Rejection sampling over sharded structures: random generation
 *        feeding a RAGGED compaction.
 *
 * Draw candidate pairs (x, u), keep x where u <= f(x): the accepted samples
 * then distribute with density proportional to f. THE PRODUCT IS THE SAMPLE
 * SET ITSELF — a ragged sharded array, already distributed and placed, ready
 * for downstream sharded consumers (ensemble initialization, MCMC seeding,
 * evaluating many functionals later) — not a reduced scalar. That makes this
 * the complementary pole to the pi example: there the samples are an
 * intermediate and fusing away materialization is the right spelling; here
 * the output is data-dependent in size and must persist, so ragged
 * materialized storage is intrinsic, not a style choice.
 *
 * Here f(x) = 4x(1-x) (envelope M = 1), so the normalized target is
 * p(x) = 6x(1-x) with acceptance probability 2/3, E[x] = 1/2 and
 * E[x^2] = 3/10 — three exact checks for free (the moments are
 * verification, not the goal).
 *
 * What the pipeline exercises:
 *  - `generate_uniform` (cuRAND tier) fills ONE sharded float array with 2n
 *    invariant draws; a `float2` VIEW over the same shards pairs them up
 *    (pair k is always draws (2k, 2k+1), so shard boundaries must be even —
 *    element grouping is a sharding contract, stated here on purpose).
 *  - `copy_if` (out-of-place) compacts the accepted pairs into a
 *    co-partitioned destination whose per-shard sizes are DATA-DEPENDENT:
 *    the ragged result is held natively (sizes commit, global offsets
 *    re-tile, `validate()` still holds).
 *  - the POPULATION IS THEN CONSUMED, as it normally would be: `zip_transform`
 *    + `reduce` estimate functionals of p, and `histogram_even` builds the
 *    empirical density over the ragged structure directly — downstream
 *    algorithms take the data-dependent result unchanged.
 *
 * And the reproducibility property compaction inherits from generation: a
 * stable filter of a sharding-invariant stream is itself sharding-invariant,
 * so the CONCATENATED ACCEPTED SEQUENCE is bitwise identical for any
 * sharding — checked below by running the whole pipeline under two different
 * (even) shard boundaries and comparing the accepted bytes.
 */

#include <cuda/experimental/__sharded/random.cuh> // opt-in vendor tier
#include <cuda/experimental/sharded.cuh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
struct accept_under_parabola
{
  __host__ __device__ bool operator()(float2 s) const
  {
    return s.y <= 4.0f * s.x * (1.0f - s.x);
  }
};

struct take_x
{
  __host__ __device__ float operator()(float2 s) const
  {
    return s.x;
  }
};

struct square
{
  __host__ __device__ float operator()(float v) const
  {
    return v * v;
  }
};

struct run_result
{
  size_t kept;
  double mean;
  double mean_sq;
  ::std::vector<size_t> hist; // empirical density of the population
  ::std::vector<float2> accepted; // concatenated in global order
};

run_result sample(place_group& group, const ::std::vector<size_t>& pair_sizes, unsigned long long seed)
{
  // 2 draws per candidate pair; even float boundaries by construction.
  ::std::vector<size_t> draw_sizes;
  for (size_t s : pair_sizes)
  {
    draw_sizes.push_back(2 * s);
  }
  auto draws = sharded_array<float>::allocate(group, draw_sizes);
  generate_uniform(draws, default_envs(draws), seed);

  // Pair view over the same shards: pair k = draws (2k, 2k+1).
  ::std::vector<cuda::std::span<float2>> pieces;
  for (size_t i = 0; i < draws.num_shards(); i++)
  {
    const auto& d = draws.shard(i);
    pieces.push_back({reinterpret_cast<float2*>(d.data), d.size / 2});
  }
  const auto candidates = make_sharded_view(pieces);

  // Ragged compaction: capacity co-partitioned with the candidates, sizes
  // committed from the per-shard acceptance counts.
  auto accepted     = sharded_array<float2>::allocate(group, pair_sizes);
  const size_t kept = copy_if(candidates, accepted, accept_under_parabola{});

  // Moments of the accepted x, over the ragged structure. Co-partitioned
  // scratch comes from allocate_like: same ragged sizes, placements and
  // reference streams as the (post-commit) source.
  auto xs = sharded_array<float>::allocate_like(accepted);
  zip_transform(xs, take_x{}, accepted);
  const double mean = reduce(xs, ::cuda::std::plus<float>{}, 0.0f) / static_cast<double>(kept);
  auto xs2          = sharded_array<float>::allocate_like(accepted);
  zip_transform(xs2, square{}, xs);
  const double mean_sq = reduce(xs2, ::cuda::std::plus<float>{}, 0.0f) / static_cast<double>(kept);

  // Consume the population as a distribution: its empirical density. Each
  // bin's mass should approach the analytic integral of p over the bin.
  const auto hist = histogram_even(xs, /*num_bins=*/10, 0.0f, 1.0f);

  // Concatenate the accepted pairs in global order (host snapshot).
  ::std::vector<float2> host(kept);
  size_t off = 0;
  for (size_t i = 0; i < accepted.num_shards(); i++)
  {
    const auto& s = accepted.shard(i);
    if (s.size != 0)
    {
      cuda_safe_call(cudaMemcpy(host.data() + off, s.data, s.size * sizeof(float2), cudaMemcpyDeviceToHost));
      off += s.size;
    }
  }
  return {kept, mean, mean_sq, hist, ::std::move(host)};
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));
  place_group group{make_locality_domain_grid()};
  const size_t n_pairs = 1u << 23;
  printf("sharded rejection sampling: %zu candidate pairs over %zu places\n", n_pairs, group.size());
  printf("target p(x) = 6x(1-x): acceptance 2/3, E[x] = 0.5, E[x^2] = 0.3\n\n");

  // Two shardings of the candidate index space (both with even draw counts).
  ::std::vector<size_t> even(group.size(), n_pairs / group.size());
  even[0] += n_pairs % group.size();
  ::std::vector<size_t> uneven = even;
  if (group.size() > 1)
  {
    const size_t shift = 1048576;
    uneven[0] += shift;
    uneven[1] -= shift;
  }

  const run_result a = sample(group, even, /*seed=*/2026);
  const run_result b = sample(group, uneven, /*seed=*/2026);

  for (const auto* r : {&a, &b})
  {
    printf("  kept %zu / %zu (%.5f)   E[x] ~ %.5f   E[x^2] ~ %.5f\n",
           r->kept,
           n_pairs,
           static_cast<double>(r->kept) / n_pairs,
           r->mean,
           r->mean_sq);
  }

  // The consumed population against the analytic density: bin mass of
  // p(x) = 6x(1-x) over [l, u] is (3u^2 - 2u^3) - (3l^2 - 2l^3).
  printf("\n  empirical density (10 bins) vs analytic bin mass:\n");
  double max_err = 0.0;
  for (int bin = 0; bin < 10; bin++)
  {
    const double l         = bin / 10.0;
    const double u         = (bin + 1) / 10.0;
    const double analytic  = (3 * u * u - 2 * u * u * u) - (3 * l * l - 2 * l * l * l);
    const double empirical = static_cast<double>(a.hist[bin]) / static_cast<double>(a.kept);
    max_err                = ::std::max(max_err, ::std::abs(empirical - analytic));
    printf("    [%.1f, %.1f): %.5f vs %.5f\n", l, u, empirical, analytic);
  }
  printf("  max bin error: %.5f\n", max_err);

  if (a.kept != b.kept || ::std::memcmp(a.accepted.data(), b.accepted.data(), a.kept * sizeof(float2)) != 0)
  {
    printf("FAILED: the accepted sequence is not sharding-invariant\n");
    return 1;
  }
  printf("accepted sequence is bitwise identical across shardings (%zu pairs)\n", a.kept);
  return 0;
}
