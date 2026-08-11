// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

// Input generators for CUB histogram benchmarks. Each shape chooses a bin and
// emits a sample from that bin's interval, allowing the benchmark verifier to
// independently recover and validate the result.

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tabulate.h>

#include <cuda/std/algorithm>
#include <cuda/std/type_traits>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

// ---------------------------------------------------------------------------
// Shape catalog
// ---------------------------------------------------------------------------

enum class InputShape
{
  concentrated, // Normalized entropy in [0, 1].
  powerlaw, // Normalized entropy in [0, 1].
  zipf, // Rank exponent greater than or equal to zero.
  temporal_phases, // Number of phases.
  strided_sweep, // Stride between consecutive bins.
  sawtooth, // Ramp period in bins.
};

// A shape plus an optional knob value. `has_knob == false` means "use the
// shape's default". The knob's meaning is shape-specific (see the enum).
struct ShapeSpec
{
  InputShape shape;
  double knob   = 0.0;
  bool has_knob = false;
};

inline constexpr double default_concentrated_entropy = 0.5;
inline constexpr double default_powerlaw_entropy     = 0.5;
inline constexpr double default_zipf_exponent        = 1.0;
inline constexpr int32_t default_temporal_phases     = 8;
inline constexpr uint64_t default_strided_stride     = 9973;
inline constexpr uint64_t default_sawtooth_period    = 0;

// Parse an InputShape axis value of the form "name" or "name:knob".
//   "concentrated:1.0"    -> uniform (entropy 1.0)
//   "concentrated:0.5"    -> a single hot bin over a floor (was "spike")
//   "concentrated:0.0"    -> a single bin gets 100% (was "constant")
//   "powerlaw:0.3"        -> power law at target entropy 0.3
// There is deliberately ONE concentrated shape spanning uniform<->constant via
// its entropy knob -- no separate uniform/constant/spike names.
inline ShapeSpec parse_input_shape(const std::string& spec)
{
  std::string name = spec;
  ShapeSpec out{};
  const auto colon = spec.find(':');
  if (colon != std::string::npos)
  {
    name         = spec.substr(0, colon);
    out.knob     = std::stod(spec.substr(colon + 1));
    out.has_knob = true;
  }

  if (name == "concentrated")
  {
    out.shape = InputShape::concentrated;
  }
  else if (name == "powerlaw")
  {
    out.shape = InputShape::powerlaw;
  }
  else if (name == "zipf")
  {
    out.shape = InputShape::zipf;
  }
  else if (name == "temporal_phases")
  {
    out.shape = InputShape::temporal_phases;
  }
  else if (name == "strided_sweep")
  {
    out.shape = InputShape::strided_sweep;
  }
  else if (name == "sawtooth")
  {
    out.shape = InputShape::sawtooth;
  }
  else
  {
    throw std::runtime_error("Unknown InputShape: " + spec);
  }
  return out;
}

// Resolve a knob to a concrete value, applying the shape's default when the
// axis value did not specify one.
inline double knob_or(const ShapeSpec& s, double default_value)
{
  return s.has_knob ? s.knob : default_value;
}

// SplitMix64 finalizer mapped to [0, 1).
__host__ __device__ inline double u01_from_hash(uint64_t x)
{
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  x = x ^ (x >> 31);
  return static_cast<double>(x >> 11) * (1.0 / 9007199254740992.0); // 2^53
}

__host__ __device__ inline uint64_t element_key(uint64_t i, uint64_t seed)
{
  return i * 6364136223846793005ull + 1442695040888963407ull + seed * 0x9E3779B97F4A7C15ull;
}

__host__ __device__ inline int32_t scatter_bin(uint64_t rank, int32_t num_bins, uint64_t offset)
{
  constexpr uint64_t scatter_prime = 2147483647;
  return static_cast<int32_t>((rank * scatter_prime + offset) % static_cast<uint64_t>(num_bins));
}

// ---------------------------------------------------------------------------
// Bin -> sample value mappers (one per histogram path). Both emit a value in
// the *interior* of bin `b` so CUB re-derives exactly `b`.
// ---------------------------------------------------------------------------
template <class SampleT>
struct even_bin_to_value
{
  double lower_level;
  double bin_width;
  SampleT lower;
  SampleT upper;

  __host__ __device__ SampleT operator()(int32_t bin) const
  {
    double v = lower_level + (static_cast<double>(bin) + 0.5) * bin_width;
    if constexpr (::cuda::std::is_integral_v<SampleT>)
    {
      double f = ::floor(v);
      if (f < static_cast<double>(lower))
      {
        f = static_cast<double>(lower);
      }
      if (f > static_cast<double>(upper) - 1.0)
      {
        f = static_cast<double>(upper) - 1.0;
      }
      return static_cast<SampleT>(f);
    }
    else
    {
      if (v < static_cast<double>(lower))
      {
        v = static_cast<double>(lower);
      }
      return static_cast<SampleT>(v);
    }
  }
};

template <class SampleT>
struct range_bin_to_value
{
  const SampleT* levels; // num_bins + 1 strictly increasing levels
  int32_t num_bins;

  __host__ __device__ SampleT operator()(int32_t bin) const
  {
    if (bin < 0)
    {
      bin = 0;
    }
    if (bin > num_bins - 1)
    {
      bin = num_bins - 1;
    }
    const double lo = static_cast<double>(levels[bin]);
    const double hi = static_cast<double>(levels[bin + 1]);
    const double v  = 0.5 * (lo + hi);
    SampleT s;
    if constexpr (::cuda::std::is_integral_v<SampleT>)
    {
      s = static_cast<SampleT>(::floor(v));
    }
    else
    {
      s = static_cast<SampleT>(v);
    }
    // Guarantee s lands in [levels[bin], levels[bin+1]).
    if (s < levels[bin])
    {
      s = levels[bin];
    }
    if (s >= levels[bin + 1])
    {
      s = levels[bin];
    }
    return s;
  }
};

// ---------------------------------------------------------------------------
// Device functors.
// ---------------------------------------------------------------------------

// Inverse-CDF sampler for independent, identically distributed shapes.
template <class SampleT, class Mapper>
struct cdf_sample_functor
{
  const double* cdf; // inclusive prefix sum over bins, cdf[num_bins-1] == 1
  int32_t num_bins;
  uint64_t seed;
  Mapper mapper;

  template <class I>
  __host__ __device__ SampleT operator()(I i) const
  {
    const double u = u01_from_hash(element_key(static_cast<uint64_t>(i), seed));
    int32_t bin    = static_cast<int32_t>(::cuda::std::upper_bound(cdf, cdf + num_bins, u) - cdf);
    if (bin >= num_bins)
    {
      bin = num_bins - 1;
    }
    return mapper(bin);
  }
};

// strided_sweep: bin = stride*i % num_bins.
template <class SampleT, class Mapper>
struct strided_functor
{
  int32_t num_bins;
  uint64_t stride;
  Mapper mapper;

  template <class I>
  __host__ __device__ SampleT operator()(I i) const
  {
    const int32_t bin = static_cast<int32_t>((static_cast<uint64_t>(i) * stride) % static_cast<uint64_t>(num_bins));
    return mapper(bin);
  }
};

// Sawtooth ramp over a configurable bin window.
template <class SampleT, class Mapper>
struct sawtooth_functor
{
  uint64_t period;
  Mapper mapper;

  template <class I>
  __host__ __device__ SampleT operator()(I i) const
  {
    return mapper(static_cast<int32_t>(static_cast<uint64_t>(i) % period));
  }
};

// temporal_phases: contiguous phases, each hammering one scattered bin.
template <class SampleT, class Mapper>
struct phases_functor
{
  int32_t num_bins;
  int32_t num_phases;
  uint64_t n;
  uint64_t offset;
  Mapper mapper;

  template <class I>
  __host__ __device__ SampleT operator()(I i) const
  {
    uint64_t phase = (static_cast<uint64_t>(i) * static_cast<uint64_t>(num_phases)) / n;
    if (phase >= static_cast<uint64_t>(num_phases))
    {
      phase = static_cast<uint64_t>(num_phases) - 1;
    }
    // Spread phases across the bin array, scattered off zero.
    const int32_t bin =
      scatter_bin(phase * (static_cast<uint64_t>(num_bins) / static_cast<uint64_t>(num_phases) + 1), num_bins, offset);
    return mapper(bin);
  }
};

// Keyed permutation of [0, n) using a Feistel network with cycle walking.
__host__ __device__ inline uint64_t feistel_mix(uint64_t x, uint64_t k)
{
  uint64_t z = x + k + 0x9E3779B97F4A7C15ull;
  z          = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
  z          = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
  return z ^ (z >> 31);
}

__host__ __device__ inline uint64_t permute_index(uint64_t i, uint64_t n, uint64_t seed)
{
  if (n <= 1)
  {
    return 0;
  }
  int32_t bits = 0;
  while (bits < 64 && ((n - 1) >> bits))
  {
    ++bits; // bits == ceil(log2(n)) for n >= 2
  }
  const int32_t half  = (bits + 1) / 2;
  const uint64_t mask = (half >= 64) ? ~0ull : ((1ull << half) - 1ull);
  uint64_t x          = i;
  do
  {
    uint64_t l = (x >> half) & mask;
    uint64_t r = x & mask;
    for (int32_t round = 0; round < 4; ++round)
    {
      const uint64_t nl = r;
      const uint64_t nr = l ^ (feistel_mix(r, seed + static_cast<uint64_t>(round)) & mask);
      l                 = nl;
      r                 = nr;
    }
    x = (l << half) | r;
  } while (x >= n); // cycle-walk back into range
  return x;
}

// Exact uniform counts in a pseudo-random sequence order.
template <class SampleT, class Mapper>
struct shuffled_uniform_functor
{
  uint64_t n;
  int32_t num_bins;
  uint64_t seed;
  Mapper mapper;

  template <class I>
  __host__ __device__ SampleT operator()(I i) const
  {
    const uint64_t j = permute_index(static_cast<uint64_t>(i), n, seed);
    return mapper(static_cast<int32_t>(j % static_cast<uint64_t>(num_bins)));
  }
};

// ---------------------------------------------------------------------------
// Host pmf construction for the distribution shapes.
// ---------------------------------------------------------------------------

inline double normalized_entropy(const std::vector<double>& pmf)
{
  if (pmf.size() <= 1)
  {
    return 0.0;
  }
  double h = 0.0;
  for (double p : pmf)
  {
    if (p > 0.0)
    {
      h -= p * std::log2(p);
    }
  }
  return h / std::log2(static_cast<double>(pmf.size()));
}

// Ranked weights w[r] ~ (r+1)^(-s), normalized.
inline std::vector<double> ranked_powerlaw(int32_t num_bins, double s)
{
  std::vector<double> w(num_bins);
  double sum = 0.0;
  for (int32_t r = 0; r < num_bins; ++r)
  {
    w[r] = std::pow(static_cast<double>(r + 1), -s);
    sum += w[r];
  }
  for (double& x : w)
  {
    x /= sum;
  }
  return w;
}

// Solve the power-law exponent so normalized entropy ~= target (monotone
// decreasing in s -> bisection).
inline double solve_powerlaw_exponent(int32_t num_bins, double target)
{
  double lo = 0.0, hi = 60.0;
  for (int32_t it = 0; it < 60; ++it)
  {
    const double mid = 0.5 * (lo + hi);
    const double h   = normalized_entropy(ranked_powerlaw(num_bins, mid));
    // entropy decreases as s grows
    if (h < target)
    {
      hi = mid;
    }
    else
    {
      lo = mid;
    }
  }
  return 0.5 * (lo + hi);
}

// Softmax-over-random-logits pmf: draw a fixed random logit g[b] per bin, then
// pmf[b] = softmax(g/T)[b]. Temperature T dials the spread CONTINUOUSLY and
// SMOOTHLY: T->inf gives uniform (entropy 1.0), T->0 concentrates onto the
// single largest-logit bin (entropy 0.0). Unlike the old spike-slab (one hot bin
// over a flat floor), EVERY bin stays occupied and varies randomly -- e.g. at
// entropy 0.75 the distribution is "mostly uniform with mild random variation",
// not one dominant spike. Normalized entropy is monotincreasing in T, so we
// bisect log(T) to hit a target entropy. The logits are seeded so the shape is
// reproducible and the mode is not pinned to bin 0.
inline std::vector<double> softmax_logits(int32_t num_bins, uint64_t seed)
{
  std::vector<double> g(num_bins);
  for (int32_t b = 0; b < num_bins; ++b)
  {
    // Two hashed uniforms -> a standard-normal logit via Box-Muller. Decorrelated
    // per bin; host/device parity not required (concentrated interior is not swept).
    const double u1 = u01_from_hash(element_key(static_cast<uint64_t>(b), seed));
    const double u2 = u01_from_hash(element_key(static_cast<uint64_t>(b), seed ^ 0xD1B54A32D192ED03ull));
    const double r  = std::sqrt(-2.0 * std::log(u1 > 0.0 ? u1 : 1e-300));
    g[b]            = r * std::cos(6.283185307179586 * u2);
  }
  return g;
}

// Fill pmf with softmax(logits / temperature) and return its normalized entropy.
inline double softmax_pmf_into(const std::vector<double>& logits, double temperature, std::vector<double>& pmf)
{
  const int32_t num_bins = static_cast<int32_t>(logits.size());
  double max_logit       = logits[0];
  for (double value : logits)
  {
    max_logit = value > max_logit ? value : max_logit;
  }
  pmf.resize(num_bins);
  double sum = 0.0;
  for (int32_t b = 0; b < num_bins; ++b)
  {
    pmf[b] = std::exp((logits[b] - max_logit) / temperature);
    sum += pmf[b];
  }
  double entropy = 0.0;
  for (double& value : pmf)
  {
    value /= sum;
    if (value > 0.0)
    {
      entropy -= value * std::log2(value);
    }
  }
  return num_bins <= 1 ? 0.0 : entropy / std::log2(static_cast<double>(num_bins));
}

inline std::vector<double> solve_softmax_pmf(int32_t num_bins, double target, uint64_t seed)
{
  const std::vector<double> logits = softmax_logits(num_bins, seed);
  std::vector<double> pmf;
  // Bisect in log-space: entropy increases with T.
  double lo = 1e-3, hi = 1e3;
  for (int32_t it = 0; it < 40; ++it)
  {
    const double mid = std::sqrt(lo * hi);
    if (softmax_pmf_into(logits, mid, pmf) < target)
    {
      lo = mid; // too concentrated -> raise T
    }
    else
    {
      hi = mid;
    }
  }
  softmax_pmf_into(logits, std::sqrt(lo * hi), pmf);
  return pmf;
}

// Build the probability mass function for a distribution shape.
inline std::vector<double> build_pmf(const ShapeSpec& spec, int32_t num_bins, uint64_t seed)
{
  std::vector<double> pmf(num_bins, 0.0);
  const uint64_t offset = seed % static_cast<uint64_t>(num_bins);

  switch (spec.shape)
  {
    case InputShape::concentrated: {
      const double target = knob_or(spec, default_concentrated_entropy);
      if (target >= 1.0)
      {
        const double p = 1.0 / num_bins; // exact uniform
        for (int32_t b = 0; b < num_bins; ++b)
        {
          pmf[b] = p;
        }
      }
      else if (target <= 0.0)
      {
        pmf[scatter_bin(0, num_bins, offset)] = 1.0; // exact single bin
      }
      else
      {
        pmf = solve_softmax_pmf(num_bins, target, seed);
      }
      break;
    }
    case InputShape::powerlaw:
    case InputShape::zipf: {
      const double s              = spec.shape == InputShape::powerlaw
                                    ? solve_powerlaw_exponent(num_bins, knob_or(spec, default_powerlaw_entropy))
                                    : knob_or(spec, default_zipf_exponent);
      const std::vector<double> w = ranked_powerlaw(num_bins, s);
      for (int32_t r = 0; r < num_bins; ++r)
      {
        pmf[scatter_bin(static_cast<uint64_t>(r), num_bins, offset)] = w[r];
      }
      break;
    }
    default:
      throw std::runtime_error("build_pmf called with a non-distribution shape");
  }
  return pmf;
}

inline bool is_ordering_shape(InputShape shape)
{
  return shape == InputShape::temporal_phases || shape == InputShape::strided_sweep || shape == InputShape::sawtooth;
}

// ---------------------------------------------------------------------------
// Core generation: given a bin->value mapper, fill an output device vector of
// `n` samples according to `spec`.
// ---------------------------------------------------------------------------
template <class SampleT, class OffsetT, class Mapper>
thrust::device_vector<SampleT>
generate_shape_impl(const ShapeSpec& spec, OffsetT n, int32_t num_bins, Mapper mapper, uint64_t seed)
{
  thrust::device_vector<SampleT> out(static_cast<std::size_t>(n));
  const uint64_t offset = seed % static_cast<uint64_t>(num_bins);

  // Use a permutation of round-robin bins for exact uniform counts.
  if (spec.shape == InputShape::concentrated && knob_or(spec, default_concentrated_entropy) >= 1.0)
  {
    shuffled_uniform_functor<SampleT, Mapper> fn{static_cast<uint64_t>(n), num_bins, seed, mapper};
    thrust::tabulate(out.begin(), out.end(), fn);
    return out;
  }

  if (!is_ordering_shape(spec.shape))
  {
    const std::vector<double> pmf = build_pmf(spec, num_bins, seed);
    thrust::host_vector<double> h_cdf(num_bins);
    double acc = 0.0;
    for (int32_t b = 0; b < num_bins; ++b)
    {
      acc += pmf[b];
      h_cdf[b] = acc;
    }
    h_cdf[num_bins - 1]                 = 1.0;
    thrust::device_vector<double> d_cdf = h_cdf;
    cdf_sample_functor<SampleT, Mapper> fn{thrust::raw_pointer_cast(d_cdf.data()), num_bins, seed, mapper};
    thrust::tabulate(out.begin(), out.end(), fn);
    return out;
  }

  switch (spec.shape)
  {
    case InputShape::strided_sweep: {
      const uint64_t stride = spec.has_knob ? static_cast<uint64_t>(std::llround(spec.knob)) : default_strided_stride;
      strided_functor<SampleT, Mapper> fn{num_bins, stride, mapper};
      thrust::tabulate(out.begin(), out.end(), fn);
      break;
    }
    case InputShape::sawtooth: {
      const uint64_t requested =
        spec.has_knob ? static_cast<uint64_t>(std::llround(spec.knob)) : default_sawtooth_period;
      const uint64_t period =
        (requested == 0) ? static_cast<uint64_t>(num_bins) : std::min(requested, static_cast<uint64_t>(num_bins));
      sawtooth_functor<SampleT, Mapper> fn{period, mapper};
      thrust::tabulate(out.begin(), out.end(), fn);
      break;
    }
    case InputShape::temporal_phases: {
      const int32_t requested = spec.has_knob ? static_cast<int32_t>(std::llround(spec.knob)) : default_temporal_phases;
      const int32_t phases    = std::max<int32_t>(1, std::min(requested, num_bins));
      phases_functor<SampleT, Mapper> fn{num_bins, phases, static_cast<uint64_t>(n), offset, mapper};
      thrust::tabulate(out.begin(), out.end(), fn);
      break;
    }
    default:
      throw std::runtime_error("unreachable ordering shape");
  }
  return out;
}

// ---------------------------------------------------------------------------
// Public input generators.
// ---------------------------------------------------------------------------
template <class SampleT, class OffsetT>
thrust::device_vector<SampleT> generate_histogram_input_even(
  const ShapeSpec& spec, OffsetT n, int32_t num_bins, SampleT lower, SampleT upper, uint64_t seed = 42)
{
  const double bin_width = (static_cast<double>(upper) - static_cast<double>(lower)) / static_cast<double>(num_bins);
  even_bin_to_value<SampleT> mapper{static_cast<double>(lower), bin_width, lower, upper};
  return generate_shape_impl<SampleT, OffsetT>(spec, n, num_bins, mapper, seed);
}

template <class SampleT, class OffsetT>
thrust::device_vector<SampleT> generate_histogram_input_range(
  const ShapeSpec& spec, OffsetT n, int32_t num_bins, const SampleT* d_levels, uint64_t seed = 42)
{
  range_bin_to_value<SampleT> mapper{d_levels, num_bins};
  return generate_shape_impl<SampleT, OffsetT>(spec, n, num_bins, mapper, seed);
}
