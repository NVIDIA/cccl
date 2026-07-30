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
  concentrated, // spike-slab family. KNOB = target normalized entropy in
                // [0,1]: 1.0 = uniform, 0.0 = constant (single bin), in
                // between = one hot bin over a uniform floor. Sweeping the
                // knob reproduces (and generalizes) the old Entropy sweep --
                // continuously, and with the hot bin scattered off zero.
  powerlaw, // decaying warm set (many hot bins). KNOB = target normalized
            // entropy in [0,1]; the rank exponent is solved to hit it.
            // Independent of the concentrated knob.
  zipf, // decaying warm set with a classic exponent. KNOB = exponent
        // s >= 0 (default 1.0).
  hash_synonym, // hot bins all collide on one cache slot. KNOB = hot share
                // in [0,1] (default 0.9).
  stale_resident, // a cold working set, swept cyclically, that recurs in every
                  // block but overflows the SMEM cache so it cannot stay resident
                  // (thrashes it). KNOB = working-set size as a multiple of cache
                  // slots (default 2.0 => twice the slots, overflowing the cache).
  temporal_phases, // the hot bin steps to a new location across phases. KNOB =
                   // number of phases (default 8).
  strided_sweep, // bin = stride*i % B (minimal temporal locality). KNOB =
                 // stride (default a large prime).
  sawtooth, // bin = i % period: a monotonic ramp 0..period-1 that resets
            // periodically (sequential locality over a bounded working
            // set). KNOB = ramp period in bins (default = num_bins => one
            // full 0..B-1 sweep, the classic sawtooth).
};

// A shape plus an optional knob value. `has_knob == false` means "use the
// shape's default". The knob's meaning is shape-specific (see the enum).
struct ShapeSpec
{
  InputShape shape;
  double knob   = 0.0;
  bool has_knob = false;
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
  else if (name == "hash_synonym")
  {
    out.shape = InputShape::hash_synonym;
  }
  else if (name == "stale_resident")
  {
    out.shape = InputShape::stale_resident;
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
  int lo  = 0;
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

// sawtooth: bin = i % period. A monotonically increasing ramp that resets to 0
// every `period` elements -- sequential access over a bounded working set, with
// a periodic discontinuity. With period == num_bins this is exactly the uniform
// round-robin tiling (the concentrated:1.0 endpoint); smaller periods confine
// the sweep to a `period`-bin window (e.g. period <= cache slots stays cache
// resident, period just over it thrashes).
template <class SampleT, class Mapper>
struct sawtooth_functor
{
  uint64_t period;
  Mapper mapper;

  template <class I>
  __host__ __device__ SampleT operator()(I i) const
  {
    return mapper(static_cast<int>(static_cast<uint64_t>(i) % period));
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
    int bin =
      scatter_bin(phase * (static_cast<uint64_t>(num_bins) / static_cast<uint64_t>(num_phases) + 1), num_bins, offset);
    return mapper(bin);
  }
};

// A keyed pseudo-random bijection on [0, n), evaluated per index with no
// buffers and no global sort: a 4-round balanced Feistel network with
// cycle-walking. The Feistel is a bijection on the padded domain
// [0, 2^(2*half)); cycle-walking (re-enciphering any result >= n) restricts it
// to an exact bijection on [0, n). Used to put the exact-count uniform tiling
// into a RANDOM sequence order (see shuffled_uniform_functor).
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
  int bits = 0;
  while ((n - 1) >> bits)
  {
    ++bits; // bits == ceil(log2(n)) for n >= 2
  }
  const int half      = (bits + 1) / 2;
  const uint64_t mask = (half >= 64) ? ~0ull : ((1ull << half) - 1ull);
  uint64_t x          = i;
  do
  {
    uint64_t l = (x >> half) & mask;
    uint64_t r = x & mask;
    for (int round = 0; round < 4; ++round)
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

// Exact uniform with RANDOM sequence order: every bin still receives exactly
// floor(n/num_bins) or +1 samples (the round-robin tiling multiset), but the
// positions are a pseudo-random permutation, so access is NOT sequential. This
// is the entropy=1.0 endpoint: uniform counts, randomly distributed in the
// input. (Pure sequential tiling bin(i)=i%num_bins is available as the
// `sawtooth` shape; strided_sweep destroys locality with a coprime stride.)
template <class SampleT, class Mapper>
struct shuffled_uniform_functor
{
  uint64_t n;
  int num_bins;
  uint64_t seed;
  Mapper mapper;

  template <class I>
  __host__ __device__ SampleT operator()(I i) const
  {
    const uint64_t j = permute_index(static_cast<uint64_t>(i), n, seed);
    return mapper(static_cast<int>(j % static_cast<uint64_t>(num_bins)));
  }
};

// stale_resident: a cold WORKING SET of `span` distinct bins (sized as a multiple
// of the SMEM cache capacity via the cover knob), scattered across the bin array
// and swept CYCLICALLY. The same `span` keys recur in every block/tile -- so the
// per-block cache is hit by them on every block -- but when span > cache slots the
// set cannot stay resident: each key is evicted before it comes around again, so
// the cache yields ~no benefit (the keys are "stale residents" that thrash it).
// span <= slots fits and caches well; the default (cover=2 => 2*slots) overflows
// it. The cyclic SEQUENTIAL reuse (each key revisited at a fixed stride apart) is
// what makes the cache's capacity boundary actually show. The cyclic counter advances
// by a large odd stride so a thread's grid-strided positions still walk distinct
// keys (it does not alias the launch's grid stride).
template <class SampleT, class Mapper>
struct stale_functor
{
  int num_bins;
  uint64_t span; // size of the cold working set (number of distinct bins)
  uint64_t offset;
  Mapper mapper;

  template <class I>
  __host__ __device__ SampleT operator()(I i) const
  {
    // Position -> which cold key. A large odd multiplier decorrelates the key
    // index from the grid-stride launch geometry while still cycling [0, span).
    const uint64_t k = (static_cast<uint64_t>(i) * 2654435761ull) % span;
    const int bin    = scatter_bin(k, num_bins, offset); // scatter across the array
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

// Softmax-over-random-logits pmf: draw a fixed random logit g[b] per bin, then
// pmf[b] = softmax(g/T)[b]. Temperature T dials the spread CONTINUOUSLY and
// SMOOTHLY: T->inf gives uniform (entropy 1.0), T->0 concentrates onto the
// single largest-logit bin (entropy 0.0). Unlike the old spike-slab (one hot bin
// over a flat floor), EVERY bin stays occupied and varies randomly -- e.g. at
// entropy 0.75 the distribution is "mostly uniform with mild random variation",
// not one dominant spike. Normalized entropy is monotincreasing in T, so we
// bisect log(T) to hit a target entropy. The logits are seeded so the shape is
// reproducible and the mode is not pinned to bin 0.
inline std::vector<double> softmax_logits(int num_bins, uint64_t seed)
{
  std::vector<double> g(num_bins);
  for (int b = 0; b < num_bins; ++b)
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

inline std::vector<double> softmax_pmf(const std::vector<double>& g, double T)
{
  const int num_bins = static_cast<int>(g.size());
  double gmax        = g[0];
  for (double v : g)
  {
    gmax = v > gmax ? v : gmax;
  }
  std::vector<double> pmf(num_bins);
  double sum = 0.0;
  for (int b = 0; b < num_bins; ++b)
  {
    pmf[b] = std::exp((g[b] - gmax) / T);
    sum += pmf[b];
  }
  for (double& v : pmf)
  {
    v /= sum;
  }
  return pmf;
}

inline std::vector<double> solve_softmax_pmf(int num_bins, double target, uint64_t seed)
{
  const std::vector<double> g = softmax_logits(num_bins, seed);
  // Bisect in log-space: entropy increases with T.
  double lo = 1e-3, hi = 1e3;
  for (int it = 0; it < 80; ++it)
  {
    const double mid = std::sqrt(lo * hi);
    if (normalized_entropy(softmax_pmf(g, mid)) < target)
    {
      lo = mid; // too concentrated -> raise T
    }
    else
    {
      hi = mid;
    }
  }
  return softmax_pmf(g, std::sqrt(lo * hi));
}

// Defaults applied when the axis value supplies no knob.
constexpr double kDefaultConcentratedEntropy = 0.5; // bare "concentrated"
constexpr double kDefaultPowerlawEntropy     = 0.5; // bare "powerlaw"
constexpr double kDefaultZipfExponent        = 1.0; // bare "zipf"
constexpr double kDefaultHashSynonymHotShare = 0.9; // bare "hash_synonym"
constexpr int kDefaultTemporalPhases         = 8; // bare "temporal_phases"
constexpr uint64_t kDefaultStridedStride     = 9973ull; // bare "strided_sweep"
// bare "sawtooth" => period 0 sentinel, resolved to num_bins at generation time.
constexpr uint64_t kDefaultSawtoothPeriod = 0ull;

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
        // Completely random bin probabilities dialed to the target entropy: a
        // softmax over per-bin random logits. All bins occupied, smoothly more
        // uniform as the knob -> 1 (no single dominant "hot" bin).
        pmf = solve_softmax_pmf(num_bins, target, seed);
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
    default:
      throw std::runtime_error("build_pmf called with a non-distribution shape");
  }
  return pmf;
}

inline bool is_ordering_shape(InputShape shape)
{
  return shape == InputShape::stale_resident || shape == InputShape::temporal_phases
      || shape == InputShape::strided_sweep || shape == InputShape::sawtooth;
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

  // Exact-uniform endpoint: every bin gets exactly n/num_bins (+-1) samples, in
  // a pseudo-random sequence order (a Feistel permutation of the round-robin
  // tiling). Uniform counts, randomly distributed in the input -- not the
  // sequential ramp (that is the `sawtooth` shape). Emitted directly rather than
  // via i.i.d. uniform sampling so the per-bin counts are exact (no multinomial
  // noise / empty bins when bins ~= elements).
  if (spec.shape == InputShape::concentrated && spec.has_knob && spec.knob >= 1.0)
  {
    shuffled_uniform_functor<SampleT, Mapper> fn{static_cast<uint64_t>(n), num_bins, seed, mapper};
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
    h_cdf[num_bins - 1]                 = 1.0; // guard against fp drift on the last bin
    thrust::device_vector<double> d_cdf = h_cdf;
    cdf_sample_functor<SampleT, Mapper> fn{thrust::raw_pointer_cast(d_cdf.data()), num_bins, seed, mapper};
    thrust::tabulate(out.begin(), out.end(), fn);
    return out;
  }

  switch (spec.shape)
  {
    case InputShape::strided_sweep: {
      // KNOB = stride.
      const uint64_t stride = spec.has_knob ? static_cast<uint64_t>(std::llround(spec.knob)) : kDefaultStridedStride;
      strided_functor<SampleT, Mapper> fn{num_bins, stride, mapper};
      thrust::tabulate(out.begin(), out.end(), fn);
      break;
    }
    case InputShape::sawtooth: {
      // KNOB = ramp period in bins; 0 (the default) means a full num_bins sweep.
      const uint64_t requested =
        spec.has_knob ? static_cast<uint64_t>(std::llround(spec.knob)) : kDefaultSawtoothPeriod;
      const uint64_t period =
        (requested == 0) ? static_cast<uint64_t>(num_bins) : std::min(requested, static_cast<uint64_t>(num_bins));
      sawtooth_functor<SampleT, Mapper> fn{period, mapper};
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
      // KNOB = cold working-set size as a multiple of cache slots (default 2.0 =>
      // twice the slots, so it overflows and thrashes a per-block cache). The set
      // recurs in every block but cannot stay resident when span > slots.
      const double cover  = knob_or(spec, 2.0);
      const int64_t want  = static_cast<int64_t>(std::llround(cover * kAdversarialCacheSlots));
      const uint64_t span = static_cast<uint64_t>(std::max<int64_t>(1, std::min<int64_t>(want, num_bins)));
      stale_functor<SampleT, Mapper> fn{num_bins, span, offset, mapper};
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
