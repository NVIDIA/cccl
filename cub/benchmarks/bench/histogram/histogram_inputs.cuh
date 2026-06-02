// SPDX-FileCopyrightText: Copyright (c) 2011-2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

// Input-shape generators for the CUB histogram benchmarks.
//
// The legacy `generate(elements, entropy, lower, upper)` knob (bitwise-AND
// "entropy") is non-linear, bunched at the extremes, always pins the hot bin to
// the zero value, and cannot express multi-hot or cache-adversarial inputs.
// This header replaces it with a set of named INPUT SHAPES that control the
// *bin* distribution directly.
//
// Mechanism: every shape decides a per-element BIN index in [0, num_bins), then
// emits a SampleT value that lands inside that bin's value interval. CUB then
// re-derives the bin from the value, so the in-bench verifier
// (`bench_verify_histogram_*` in histogram_common.cuh) validates the mapping
// automatically -- we feed the verifier, never bypass it.
//
//   * EVEN  path: bin b owns [lower + b*w, lower + (b+1)*w), w=(upper-lower)/B.
//                 emit the bin midpoint -> CUB's (s-lower)*B/(upper-lower) == b.
//   * RANGE path: bin b owns [levels[b], levels[b+1]).
//                 emit that interval's midpoint -> UpperBound(levels, s)-1 == b.
//
// Shapes split into i.i.d. DISTRIBUTION shapes (only the pmf differs; positions
// are independent) and ORDERING shapes (the pathology lives in the sequence
// order, so they are generated positionally).

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tabulate.h>

#include <cuda/std/type_traits>

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
  concentrated,   // spike-slab family. KNOB = target normalized entropy in
                  // [0,1]: 1.0 = uniform, 0.0 = constant (single bin), in
                  // between = one hot bin over a uniform floor. Sweeping the
                  // knob reproduces (and generalizes) the old Entropy sweep --
                  // continuously, and with the hot bin scattered off zero.
  powerlaw,       // decaying warm set (many hot bins). KNOB = target normalized
                  // entropy in [0,1]; the rank exponent is solved to hit it.
                  // Independent of the concentrated knob.
  zipf,           // decaying warm set with a classic exponent. KNOB = exponent
                  // s >= 0 (default 1.0).
  hash_synonym,   // hot bins all collide on one cache slot. KNOB = hot share
                  // in [0,1] (default 0.9).
  capacity_cliff, // equiprobable active set sized around cache capacity. KNOB =
                  // active set as a multiple of cache slots (default => slots+1,
                  // the just-over-capacity cliff).
  stale_resident, // cold prefix claims slots, then a hot bulk (attacks
                  // no-evict). KNOB = prefix coverage as a multiple of cache
                  // slots (default 1.0 => claim `slots` bins).
  temporal_phases,// the hot bin steps to a new location across phases. KNOB =
                  // number of phases (default 8).
  strided_sweep,  // bin = stride*i % B (minimal temporal locality). KNOB =
                  // stride (default a large prime).
};

// A shape plus an optional knob value. `has_knob == false` means "use the
// shape's default". The knob's meaning is shape-specific (see the enum).
struct ShapeSpec
{
  InputShape shape;
  double knob    = 0.0;
  bool has_knob  = false;
};

// kAdversarialCacheSlots mirrors the SMEM cuckoo-cache capacity in
// tuning_histogram.cuh; it is a benchmark *probe* and never affects
// correctness. TODO: wire to the policy's actual slot count rather than
// hardcoding, so the adversarial shapes track the cache as it is tuned.
constexpr int kAdversarialCacheSlots = 4096;
constexpr int kHashSynonymCount      = 32; // number of colliding bins

// Parse an InputShape axis value of the form "name" or "name:knob".
//   "concentrated:1.0"    -> uniform (entropy 1.0)
//   "concentrated:0.5"    -> a single hot bin over a floor (was "spike")
//   "concentrated:0.0"    -> a single bin gets 100% (was "constant")
//   "powerlaw:0.3"        -> power law at target entropy 0.3
//   "capacity_cliff"      -> default (slots+1) cliff
// There is deliberately ONE concentrated shape spanning uniform<->constant via
// its entropy knob -- no separate uniform/constant/spike names.
inline ShapeSpec parse_input_shape(const std::string& spec)
{
  std::string name = spec;
  ShapeSpec out{};
  const auto colon = spec.find(':');
  if (colon != std::string::npos)
  {
    name        = spec.substr(0, colon);
    out.knob    = std::stod(spec.substr(colon + 1));
    out.has_knob = true;
  }

  if (name == "concentrated") { out.shape = InputShape::concentrated; }
  else if (name == "powerlaw") { out.shape = InputShape::powerlaw; }
  else if (name == "zipf")     { out.shape = InputShape::zipf; }
  else if (name == "hash_synonym")    { out.shape = InputShape::hash_synonym; }
  else if (name == "capacity_cliff")  { out.shape = InputShape::capacity_cliff; }
  else if (name == "stale_resident")  { out.shape = InputShape::stale_resident; }
  else if (name == "temporal_phases") { out.shape = InputShape::temporal_phases; }
  else if (name == "strided_sweep")   { out.shape = InputShape::strided_sweep; }
  else { throw std::runtime_error("Unknown InputShape: " + spec); }
  return out;
}

// Resolve a knob to a concrete value, applying the shape's default when the
// axis value did not specify one.
inline double knob_or(const ShapeSpec& s, double default_value)
{
  return s.has_knob ? s.knob : default_value;
}

// A large prime > every bins-axis value; multiplying bin ranks by it modulo
// num_bins is a bijection (a cheap fixed permutation), used to SCATTER hot bins
// across the array so the mode is never forced to bin 0.
constexpr uint64_t kScatterPrime = 2147483647ull; // 2^31 - 1, prime

// ---------------------------------------------------------------------------
// Per-element uniform draw: splitmix64 finalizer on a decorrelated key, mapped
// to [0, 1). Higher quality per index than seeding a thrust engine per element.
// ---------------------------------------------------------------------------
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

__host__ __device__ inline int scatter_bin(uint64_t rank, int num_bins, uint64_t offset)
{
  return static_cast<int>((rank * kScatterPrime + offset) % static_cast<uint64_t>(num_bins));
}

// First index `i` in [0, n) with `val < cdf[i]` (upper-bound binary search).
// Local host/device implementation so this header has no device-only deps
// (cub::UpperBound is _CCCL_DEVICE only and we need a host path for tests).
__host__ __device__ inline int upper_bound_cdf(const double* cdf, int n, double val)
{
  int lo = 0;
  int len = n;
  while (len > 0)
  {
    const int half = len >> 1;
    if (val < cdf[lo + half])
    {
      len = half;
    }
    else
    {
      lo  = lo + half + 1;
      len = len - (half + 1);
    }
  }
  return lo;
}

// ---------------------------------------------------------------------------
// Bin -> sample value mappers (one per histogram path). Both emit a value in
// the *interior* of bin `b` so CUB re-derives exactly `b`.
// ---------------------------------------------------------------------------
template <class SampleT>
struct even_bin_to_value
{
  double L;
  double w; // (upper - lower) / num_bins
  SampleT lower;
  SampleT upper;

  __host__ __device__ SampleT operator()(int bin) const
  {
    double v = L + (static_cast<double>(bin) + 0.5) * w;
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
  int num_bins;

  __host__ __device__ SampleT operator()(int bin) const
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
    double v        = 0.5 * (lo + hi);
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

// Inverse-CDF sampler for the i.i.d. distribution shapes.
template <class SampleT, class Mapper>
struct cdf_sample_functor
{
  const double* cdf; // inclusive prefix sum over bins, cdf[num_bins-1] == 1
  int num_bins;
  uint64_t seed;
  Mapper mapper;

  template <class I>
  __host__ __device__ SampleT operator()(I i) const
  {
    const double u = u01_from_hash(element_key(static_cast<uint64_t>(i), seed));
    int bin        = upper_bound_cdf(cdf, num_bins, u);
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
  int num_bins;
  uint64_t stride;
  Mapper mapper;

  template <class I>
  __host__ __device__ SampleT operator()(I i) const
  {
    int bin = static_cast<int>((static_cast<uint64_t>(i) * stride) % static_cast<uint64_t>(num_bins));
    return mapper(bin);
  }
};

// temporal_phases: contiguous phases, each hammering one scattered bin.
template <class SampleT, class Mapper>
struct phases_functor
{
  int num_bins;
  int num_phases;
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
    int bin = scatter_bin(phase * (static_cast<uint64_t>(num_bins) / static_cast<uint64_t>(num_phases) + 1), num_bins, offset);
    return mapper(bin);
  }
};

// Exact uniform: round-robin tiling bin(i) = i % num_bins. Every bin receives
// exactly floor(n/num_bins) or +1 samples -- ACTUALLY uniform (zero count
// variance), unlike i.i.d. sampling of a uniform pmf which leaves multinomial
// noise and ~37% empty bins when bins ~= elements. This is the canonical
// entropy=1.0 endpoint. Access is sequential (a benign, low-contention
// baseline); contrast strided_sweep, which uses a large coprime stride to
// destroy locality on purpose.
template <class SampleT, class Mapper>
struct uniform_tiling_functor
{
  int num_bins;
  Mapper mapper;

  template <class I>
  __host__ __device__ SampleT operator()(I i) const
  {
    return mapper(static_cast<int>(static_cast<uint64_t>(i) % static_cast<uint64_t>(num_bins)));
  }
};

// stale_resident: a cold prefix sweeps `n_cold` distinct bins once (claiming
// every cache slot), then the hot bulk hammers a single bin disjoint from the
// prefix.
template <class SampleT, class Mapper>
struct stale_functor
{
  int num_bins;
  uint64_t n_cold;
  int hot_bin;
  Mapper mapper;

  template <class I>
  __host__ __device__ SampleT operator()(I i) const
  {
    uint64_t idx = static_cast<uint64_t>(i);
    int bin      = (idx < n_cold) ? static_cast<int>(idx % static_cast<uint64_t>(num_bins)) : hot_bin;
    return mapper(bin);
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
inline std::vector<double> ranked_powerlaw(int num_bins, double s)
{
  std::vector<double> w(num_bins);
  double sum = 0.0;
  for (int r = 0; r < num_bins; ++r)
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
inline double solve_powerlaw_exponent(int num_bins, double target)
{
  double lo = 0.0, hi = 60.0;
  for (int it = 0; it < 60; ++it)
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

// Spike-slab pmf: probability `p` on one hot bin, `(1-p)` spread uniformly over
// all bins. Normalized entropy decreases monotonically from 1.0 (p=0, uniform)
// to 0.0 (p=1, single bin), so we bisect `p` to hit a target entropy.
inline double spike_slab_entropy(int num_bins, double p)
{
  const double base = (1.0 - p) / num_bins;
  std::vector<double> pmf(num_bins, base);
  pmf[0] += p;
  return normalized_entropy(pmf);
}

inline double solve_spike_share(int num_bins, double target)
{
  double lo = 0.0, hi = 1.0;
  for (int it = 0; it < 60; ++it)
  {
    const double mid = 0.5 * (lo + hi);
    const double h   = spike_slab_entropy(num_bins, mid);
    if (h < target)
    {
      hi = mid; // too concentrated -> reduce p
    }
    else
    {
      lo = mid;
    }
  }
  return 0.5 * (lo + hi);
}

// Defaults applied when the axis value supplies no knob.
constexpr double kDefaultConcentratedEntropy = 0.5; // bare "concentrated"
constexpr double kDefaultPowerlawEntropy     = 0.5; // bare "powerlaw"
constexpr double kDefaultZipfExponent        = 1.0; // bare "zipf"
constexpr double kDefaultHashSynonymHotShare = 0.9; // bare "hash_synonym"
constexpr int kDefaultTemporalPhases         = 8;   // bare "temporal_phases"
constexpr uint64_t kDefaultStridedStride     = 9973ull; // bare "strided_sweep"

// Build the per-bin pmf for an i.i.d. distribution shape, honoring the spec's
// knob. Hot ranks are scattered across bins via scatter_bin() so the mode is
// not forced to bin 0.
inline std::vector<double> build_pmf(const ShapeSpec& spec, int num_bins, uint64_t seed)
{
  std::vector<double> pmf(num_bins, 0.0);
  const uint64_t offset = seed % static_cast<uint64_t>(num_bins);

  switch (spec.shape)
  {
    case InputShape::concentrated: {
      // KNOB = target normalized entropy. Exact endpoints, solver in between.
      const double target = knob_or(spec, kDefaultConcentratedEntropy);
      if (target >= 1.0)
      {
        const double p = 1.0 / num_bins; // exact uniform
        for (int b = 0; b < num_bins; ++b)
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
        const double p       = solve_spike_share(num_bins, target);
        const double floor_p = (1.0 - p) / num_bins;
        for (int b = 0; b < num_bins; ++b)
        {
          pmf[b] = floor_p;
        }
        pmf[scatter_bin(0, num_bins, offset)] += p;
      }
      break;
    }
    case InputShape::powerlaw: {
      const double target         = knob_or(spec, kDefaultPowerlawEntropy);
      const double s              = solve_powerlaw_exponent(num_bins, target);
      const std::vector<double> w = ranked_powerlaw(num_bins, s);
      for (int r = 0; r < num_bins; ++r)
      {
        pmf[scatter_bin(static_cast<uint64_t>(r), num_bins, offset)] = w[r];
      }
      break;
    }
    case InputShape::zipf: {
      const double s              = knob_or(spec, kDefaultZipfExponent);
      const std::vector<double> w = ranked_powerlaw(num_bins, s);
      for (int r = 0; r < num_bins; ++r)
      {
        pmf[scatter_bin(static_cast<uint64_t>(r), num_bins, offset)] = w[r];
      }
      break;
    }
    case InputShape::hash_synonym: {
      // KNOB = hot share. kHashSynonymCount bins that all collide on one cache
      // slot share the hot traffic; the rest is uniform background.
      const double hot_share = knob_or(spec, kDefaultHashSynonymHotShare);
      const int slot         = static_cast<int>(offset % static_cast<uint64_t>(kAdversarialCacheSlots));
      std::vector<int> syn;
      for (int k = 0; k < kHashSynonymCount; ++k)
      {
        const int b = slot + k * kAdversarialCacheSlots;
        if (b < num_bins)
        {
          syn.push_back(b);
        }
      }
      const double bg = (1.0 - hot_share) / num_bins;
      for (int b = 0; b < num_bins; ++b)
      {
        pmf[b] = bg;
      }
      if (!syn.empty())
      {
        const double per = hot_share / syn.size();
        for (int b : syn)
        {
          pmf[b] += per;
        }
      }
      else
      {
        pmf[scatter_bin(0, num_bins, offset)] += hot_share; // degenerate fallback
      }
      break;
    }
    case InputShape::capacity_cliff: {
      // KNOB = active set as a multiple of cache slots. Default => slots+1
      // (the just-over-capacity cliff). Equiprobable bins, spread across the
      // array so they do not trivially share slots.
      int k;
      if (spec.has_knob)
      {
        k = static_cast<int>(std::lround(spec.knob * kAdversarialCacheSlots));
      }
      else
      {
        k = kAdversarialCacheSlots + 1;
      }
      k = std::max(1, std::min(k, num_bins));
      const double p = 1.0 / k;
      for (int j = 0; j < k; ++j)
      {
        const int b = static_cast<int>((static_cast<uint64_t>(j) * num_bins) / static_cast<uint64_t>(k));
        pmf[b] = p;
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
  return shape == InputShape::stale_resident || shape == InputShape::temporal_phases
      || shape == InputShape::strided_sweep;
}

// ---------------------------------------------------------------------------
// Core generation: given a bin->value mapper, fill an output device vector of
// `n` samples according to `spec`.
// ---------------------------------------------------------------------------
template <class SampleT, class OffsetT, class Mapper>
thrust::device_vector<SampleT>
generate_shape_impl(const ShapeSpec& spec, OffsetT n, int num_bins, Mapper mapper, uint64_t seed)
{
  thrust::device_vector<SampleT> out(static_cast<std::size_t>(n));
  const uint64_t offset = seed % static_cast<uint64_t>(num_bins);

  // Exact-uniform endpoint: emitted as equal-count tiling rather than i.i.d.
  // sampling, so every bin gets exactly n/num_bins (+-1) -- truly uniform.
  if (spec.shape == InputShape::concentrated && spec.has_knob && spec.knob >= 1.0)
  {
    uniform_tiling_functor<SampleT, Mapper> fn{num_bins, mapper};
    thrust::tabulate(out.begin(), out.end(), fn);
    return out;
  }

  if (!is_ordering_shape(spec.shape))
  {
    // Distribution shape: host pmf -> inclusive CDF -> device -> inverse-CDF sample.
    std::vector<double> pmf = build_pmf(spec, num_bins, seed);
    thrust::host_vector<double> h_cdf(num_bins);
    double acc = 0.0;
    for (int b = 0; b < num_bins; ++b)
    {
      acc += pmf[b];
      h_cdf[b] = acc;
    }
    h_cdf[num_bins - 1] = 1.0; // guard against fp drift on the last bin
    thrust::device_vector<double> d_cdf = h_cdf;
    cdf_sample_functor<SampleT, Mapper> fn{thrust::raw_pointer_cast(d_cdf.data()), num_bins, seed, mapper};
    thrust::tabulate(out.begin(), out.end(), fn);
    return out;
  }

  switch (spec.shape)
  {
    case InputShape::strided_sweep: {
      // KNOB = stride.
      const uint64_t stride =
        spec.has_knob ? static_cast<uint64_t>(std::llround(spec.knob)) : kDefaultStridedStride;
      strided_functor<SampleT, Mapper> fn{num_bins, stride, mapper};
      thrust::tabulate(out.begin(), out.end(), fn);
      break;
    }
    case InputShape::temporal_phases: {
      // KNOB = number of phases.
      const int requested = spec.has_knob ? static_cast<int>(std::llround(spec.knob)) : kDefaultTemporalPhases;
      const int phases    = std::max(1, std::min(requested, num_bins));
      phases_functor<SampleT, Mapper> fn{num_bins, phases, static_cast<uint64_t>(n), offset, mapper};
      thrust::tabulate(out.begin(), out.end(), fn);
      break;
    }
    case InputShape::stale_resident: {
      // KNOB = prefix coverage as a multiple of cache slots (default 1.0).
      const double cover    = knob_or(spec, 1.0);
      const int64_t want    = static_cast<int64_t>(std::llround(cover * kAdversarialCacheSlots));
      const uint64_t n_cold = static_cast<uint64_t>(std::max<int64_t>(1, std::min<int64_t>(want, num_bins)));
      // hot bin disjoint from the cold prefix [0, n_cold) when possible.
      const int hot_bin =
        (num_bins > static_cast<int>(n_cold))
          ? static_cast<int>(n_cold + (offset % (static_cast<uint64_t>(num_bins) - n_cold)))
          : num_bins / 2;
      stale_functor<SampleT, Mapper> fn{num_bins, n_cold, hot_bin, mapper};
      thrust::tabulate(out.begin(), out.end(), fn);
      break;
    }
    default:
      throw std::runtime_error("unreachable ordering shape");
  }
  return out;
}

// ---------------------------------------------------------------------------
// Public entry points -- drop-in replacements for the legacy generate() call.
// ---------------------------------------------------------------------------
template <class SampleT, class OffsetT>
thrust::device_vector<SampleT> generate_histogram_input_even(
  const ShapeSpec& spec, OffsetT n, int num_bins, SampleT lower, SampleT upper, uint64_t seed = 42)
{
  const double w = (static_cast<double>(upper) - static_cast<double>(lower)) / static_cast<double>(num_bins);
  even_bin_to_value<SampleT> mapper{static_cast<double>(lower), w, lower, upper};
  return generate_shape_impl<SampleT, OffsetT>(spec, n, num_bins, mapper, seed);
}

template <class SampleT, class OffsetT>
thrust::device_vector<SampleT> generate_histogram_input_range(
  const ShapeSpec& spec, OffsetT n, int num_bins, const SampleT* d_levels, uint64_t seed = 42)
{
  range_bin_to_value<SampleT> mapper{d_levels, num_bins};
  return generate_shape_impl<SampleT, OffsetT>(spec, n, num_bins, mapper, seed);
}
