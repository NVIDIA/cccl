// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

// Tests the histogram benchmark input generators.

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <vector>

#include "../benchmarks/bench/histogram/histogram_inputs.cuh"
#include <c2h/catch2_test_helper.h>

namespace
{
// Generate an EVEN input for `spec` and return the per-bin counts, recomputed
// from the sample values with the same formula CUB uses.
template <class SampleT>
std::vector<long long>
bin_counts_even(const ShapeSpec& spec, int64_t n, int num_bins, SampleT lower, SampleT upper, uint64_t seed = 42)
{
  thrust::device_vector<SampleT> d_input =
    generate_histogram_input_even<SampleT>(spec, n, num_bins, lower, upper, seed);
  thrust::host_vector<SampleT> h_input = d_input;

  std::vector<long long> counts(num_bins, 0);
  const double L     = static_cast<double>(lower);
  const double U     = static_cast<double>(upper);
  const double scale = static_cast<double>(num_bins) / (U - L);
  for (SampleT s : h_input)
  {
    if (s < lower || s >= upper)
    {
      continue; // out-of-range samples are not counted by CUB either
    }
    int bin = static_cast<int>((static_cast<double>(s) - L) * scale);
    if (bin >= num_bins)
    {
      bin = num_bins - 1;
    }
    if (bin >= 0)
    {
      ++counts[bin];
    }
  }
  return counts;
}

double normalized_entropy_counts(const std::vector<long long>& counts, int64_t n)
{
  if (counts.size() <= 1 || n == 0)
  {
    return 0.0;
  }
  double h = 0.0;
  for (long long c : counts)
  {
    if (c > 0)
    {
      const double p = static_cast<double>(c) / static_cast<double>(n);
      h -= p * std::log2(p);
    }
  }
  return h / std::log2(static_cast<double>(counts.size()));
}

int nonzero_bins(const std::vector<long long>& counts)
{
  return static_cast<int>(std::count_if(counts.begin(), counts.end(), [](long long c) {
    return c > 0;
  }));
}

long long top_count(const std::vector<long long>& counts)
{
  return *std::max_element(counts.begin(), counts.end());
}

inline constexpr int64_t input_size  = 200000;
inline constexpr int32_t lower_bound = 0;
inline constexpr int32_t upper_bound = 4096;
} // namespace

C2H_TEST("histogram input: concentrated endpoints are exact", "[histogram][input_shapes]")
{
  const int num_bins = 64;

  // entropy 1.0 -> uniform: every bin within a few % of input_size/num_bins.
  {
    auto counts =
      bin_counts_even<int32_t>(parse_input_shape("concentrated:1.0"), input_size, num_bins, lower_bound, upper_bound);
    const double mean = static_cast<double>(input_size) / num_bins;
    REQUIRE(nonzero_bins(counts) == num_bins);
    for (long long c : counts)
    {
      REQUIRE(std::abs(c - mean) < 0.15 * mean);
    }
    REQUIRE(normalized_entropy_counts(counts, input_size) > 0.99);
  }

  // entropy 0.0 -> constant: exactly one nonzero bin holding everything.
  {
    auto counts =
      bin_counts_even<int32_t>(parse_input_shape("concentrated:0.0"), input_size, num_bins, lower_bound, upper_bound);
    REQUIRE(nonzero_bins(counts) == 1);
    REQUIRE(top_count(counts) == input_size);
  }
}

C2H_TEST("histogram input: concentrated entropy knob is monotone", "[histogram][input_shapes]")
{
  const int num_bins = 64;
  // As the entropy knob falls, the top-bin share must rise monotonically.
  double prev_share = -1.0;
  for (double e : {1.0, 0.75, 0.5, 0.25, 0.0})
  {
    auto counts = bin_counts_even<int32_t>(
      parse_input_shape("concentrated:" + std::to_string(e)), input_size, num_bins, lower_bound, upper_bound);
    const double share = static_cast<double>(top_count(counts)) / input_size;
    if (prev_share >= 0.0)
    {
      REQUIRE(share >= prev_share - 0.02); // non-decreasing (small tolerance)
    }
    prev_share = share;
  }
  REQUIRE(prev_share > 0.99); // entropy 0 ends at ~100% top share
}

C2H_TEST("histogram input: hot bin is not pinned to zero", "[histogram][input_shapes]")
{
  const int num_bins = 64;
  // Different seeds should move the hot bin (spike-slab at entropy 0.3).
  std::map<int, int> argmax_seen;
  for (uint64_t seed : {1ull, 2ull, 3ull, 4ull, 5ull})
  {
    auto counts = bin_counts_even<int32_t>(
      parse_input_shape("concentrated:0.3"), input_size, num_bins, lower_bound, upper_bound, seed);
    const int arg = static_cast<int>(std::max_element(counts.begin(), counts.end()) - counts.begin());
    ++argmax_seen[arg];
  }
  // The mode is not always bin 0, and varies across seeds.
  REQUIRE(argmax_seen.count(0) < 5);
  REQUIRE(argmax_seen.size() >= 2);
}

C2H_TEST("histogram input: powerlaw is a decaying warm set", "[histogram][input_shapes]")
{
  const int num_bins = 256;
  auto counts =
    bin_counts_even<int32_t>(parse_input_shape("powerlaw:0.4"), input_size, num_bins, lower_bound, upper_bound);
  // Many hot bins (more than a single spike), with a heavy head: the top bin
  // holds a large share and the warm set (top-K) dominates.
  REQUIRE(nonzero_bins(counts) > 5);
  std::vector<long long> sorted(counts);
  std::sort(sorted.rbegin(), sorted.rend());
  const double top1 = static_cast<double>(sorted[0]) / input_size;
  double top8       = 0.0;
  for (int i = 0; i < 8 && i < static_cast<int>(sorted.size()); ++i)
  {
    top8 += static_cast<double>(sorted[i]) / input_size;
  }
  REQUIRE(top1 > 0.05);
  REQUIRE(top8 > 0.5); // a small warm set carries most of the mass
  REQUIRE(top1 < 0.99); // but not a single spike
}

C2H_TEST("histogram input: powerlaw knob is monotone in entropy", "[histogram][input_shapes]")
{
  const int num_bins = 256;
  // Entropy decreases across the loop, so the top-bin share must RISE
  // (non-decreasing): lower target entropy -> higher concentration.
  double prev_top = -1.0;
  for (double e : {0.8, 0.6, 0.4, 0.2})
  {
    auto counts = bin_counts_even<int32_t>(
      parse_input_shape("powerlaw:" + std::to_string(e)), input_size, num_bins, lower_bound, upper_bound);
    std::vector<long long> sorted(counts);
    std::sort(sorted.rbegin(), sorted.rend());
    const double top1 = static_cast<double>(sorted[0]) / input_size;
    REQUIRE(top1 >= prev_top - 0.02); // non-decreasing as entropy falls
    prev_top = top1;
  }
}

C2H_TEST("histogram input: temporal_phases changes the hot bin across phases", "[histogram][input_shapes]")
{
  const int num_bins                     = 256;
  const int phases                       = 4;
  thrust::device_vector<int32_t> d_input = generate_histogram_input_even<int32_t>(
    parse_input_shape("temporal_phases:" + std::to_string(phases)), input_size, num_bins, lower_bound, upper_bound);
  thrust::host_vector<int32_t> h = d_input;

  const double scale =
    static_cast<double>(num_bins) / (static_cast<double>(upper_bound) - static_cast<double>(lower_bound));
  std::vector<int> phase_mode;
  const int64_t per = input_size / phases;
  for (int p = 0; p < phases; ++p)
  {
    std::map<int, long long> c;
    for (int64_t i = p * per; i < (p + 1) * per; ++i)
    {
      int bin = static_cast<int>(static_cast<double>(h[i]) * scale);
      if (bin >= num_bins)
      {
        bin = num_bins - 1;
      }
      ++c[bin];
    }
    int best        = -1;
    long long bestc = -1;
    for (auto& kv : c)
    {
      if (kv.second > bestc)
      {
        bestc = kv.second;
        best  = kv.first;
      }
    }
    phase_mode.push_back(best);
  }
  // Adjacent phases must have different hot bins.
  for (int p = 1; p < phases; ++p)
  {
    REQUIRE(phase_mode[p] != phase_mode[p - 1]);
  }
}

C2H_TEST("histogram input: strided_sweep is flat in distribution but ordered", "[histogram][input_shapes]")
{
  const int num_bins = 256;
  auto counts =
    bin_counts_even<int32_t>(parse_input_shape("strided_sweep"), input_size, num_bins, lower_bound, upper_bound);
  // A stride coprime to num_bins visits every bin near-equally.
  REQUIRE(nonzero_bins(counts) == num_bins);
  REQUIRE(normalized_entropy_counts(counts, input_size) > 0.98);
}

C2H_TEST("histogram input: RANGE path maps bins into level intervals", "[histogram][input_shapes]")
{
  // Build a simple strictly-increasing level array and check that a constant
  // (entropy 0) input lands every sample inside one [levels[b], levels[b+1]).
  const int num_bins = 32;
  thrust::host_vector<int32_t> h_levels(num_bins + 1);
  for (int i = 0; i <= num_bins; ++i)
  {
    h_levels[i] = i * 10; // levels 0,10,20,...,320
  }
  thrust::device_vector<int32_t> d_levels = h_levels;

  thrust::device_vector<int32_t> d_input = generate_histogram_input_range<int32_t>(
    parse_input_shape("concentrated:0.0"), input_size, num_bins, thrust::raw_pointer_cast(d_levels.data()));
  thrust::host_vector<int32_t> h = d_input;

  // All samples equal (constant) and inside the same level interval.
  const int32_t v0 = h[0];
  for (int32_t v : h)
  {
    REQUIRE(v == v0);
  }
  // v0 must sit within some [levels[b], levels[b+1]).
  REQUIRE(v0 >= h_levels[0]);
  REQUIRE(v0 < h_levels[num_bins]);
}
