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

#include <cub/detail/histogram_cache_hash.cuh>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/tabulate.h>

#include <cuda/std/type_traits>

#include <cmath>
#include <cstdint>
#include <cstdlib>
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
  hash_synonym, // hot bins all collide on one cache slot. KNOB = hot share
                // in [0,1] (default 0.9).
  stale_resident, // a cold working set of W distinct bins that recurs in every
                  // block, sized against the per-block no-eviction SMEM cache of S
                  // slots to yield a controllable hit rate. KNOB = target cache hit
                  // rate in (0,1] (default 0.5); W is solved from S so the measured
                  // rate tracks the knob (see stale_working_set()).
  poison, // prime the cache slots probed by one poison bin, then hammer
          // that bin so every steady-state update spills to one globally
          // contended output counter. No knob.
  temporal_phases, // the hot bin steps to a new location across 8 phases. KNOB =
                   // fraction of uniformly random samples in [0,1] (default 0.10).
                   // Optional KNOB2 = number of phases (default 8).
  strided_sweep, // bin = stride*i % B (minimal temporal locality). KNOB =
                 // stride (default a large prime).
  sawtooth, // bin = scatter((stride*i) % period): a configurable ramp over
            // a bounded working set. KNOBS = period:stride:scatter, with
            // defaults num_bins:1:0 (the classic monotonic sawtooth).
};

// A shape plus up to three optional knob values. Most shapes use only `knob`;
// sawtooth uses all three for period:stride:scatter. A false `has_knob*` flag
// means "use that parameter's default".
struct ShapeSpec
{
  InputShape shape;
  double knob    = 0.0;
  bool has_knob  = false;
  double knob2   = 0.0;
  bool has_knob2 = false;
  double knob3   = 0.0;
  bool has_knob3 = false;
};

// Single-channel fallback used only when a cache-sensitive shape is called by a
// legacy benchmark that cannot query the dispatch policy. Current benchmarks pass
// the actual runtime slot count from query_direct_atomic_cache_slots_for_extent.
constexpr int kAdversarialCacheSlots = 4096;
constexpr int kHashSynonymCount      = 32; // number of colliding bins

// Parse an InputShape axis value of the form "name" or "name:knob[:knob2[:knob3]]".
//   "concentrated:1.0"    -> uniform (entropy 1.0)
//   "concentrated:0.5"    -> a single hot bin over a floor (was "spike")
//   "concentrated:0.0"    -> a single bin gets 100% (was "constant")
//   "powerlaw:0.3"        -> power law at target entropy 0.3
//   "temporal_phases:0.10" -> 10% uniformly random samples in each of 8 phases
//   "sawtooth:8192:2654435761:1" -> strided, scattered 8192-bin ramp
// There is deliberately ONE concentrated shape spanning uniform<->constant via
// its entropy knob -- no separate uniform/constant/spike names.
inline ShapeSpec parse_input_shape(const std::string& spec)
{
  std::string name = spec;
  ShapeSpec out{};
  const auto colon = spec.find(':');
  if (colon != std::string::npos)
  {
    name                  = spec.substr(0, colon);
    double* knob_values[] = {&out.knob, &out.knob2, &out.knob3};
    bool* knob_flags[]    = {&out.has_knob, &out.has_knob2, &out.has_knob3};
    std::size_t begin     = colon + 1;
    for (std::size_t param = 0;; ++param)
    {
      if (param >= 3)
      {
        throw std::runtime_error("Too many InputShape parameters: " + spec);
      }
      const std::size_t end = spec.find(':', begin);
      *knob_values[param]   = std::stod(spec.substr(begin, end - begin));
      *knob_flags[param]    = true;
      if (end == std::string::npos)
      {
        break;
      }
      begin = end + 1;
    }
  }

  if (name == "concentrated")
  {
    out.shape = InputShape::concentrated;
  }
  else if (name == "powerlaw")
  {
    out.shape = InputShape::powerlaw;
  }
  else if (name == "hash_synonym")
  {
    out.shape = InputShape::hash_synonym;
  }
  else if (name == "stale_resident")
  {
    out.shape = InputShape::stale_resident;
  }
  else if (name == "poison")
  {
    out.shape = InputShape::poison;
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

// A large prime greater than every bins-axis value. It is coprime to every
// supported bin count, so the affine map below is a permutation. On the common
// power-of-two axes it is congruent to -1. Bounded working sets use the separate
// upper_window_bin mapping and therefore cannot be remapped by this constant.
constexpr uint64_t kScatterPrime      = 2147483647ull; // 2^31 - 1, prime
constexpr uint64_t kChannelSeedStride = 0xD1B54A32D192ED03ull;

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

// Map a bounded rank range [0, span) into the contiguous upper-bin window
// [num_bins-span, num_bins). This is deliberately independent of scatter_bin:
// changing the global rank permutation must not move bounded sawtooth or
// stale-resident supports, and this remains correct for non-power-of-two bins.
__host__ __device__ inline int upper_window_bin(uint64_t rank, int num_bins)
{
  return num_bins - 1 - static_cast<int>(rank);
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
  uint64_t sample_stride;
  Mapper mapper;

  template <class I>
  __host__ __device__ SampleT operator()(I i) const
  {
    const uint64_t index   = static_cast<uint64_t>(i);
    const uint64_t channel = index % sample_stride;
    const uint64_t sample  = index / sample_stride;
    const double u         = u01_from_hash(element_key(sample, seed + channel * kChannelSeedStride));
    int bin                = upper_bound_cdf(cdf, num_bins, u);
    if (bin >= num_bins)
    {
      bin = num_bins - 1;
    }
    return mapper(bin);
  }
};

// strided_sweep: every interleaved channel independently follows
// bin = stride*sample % num_bins.
template <class SampleT, class Mapper>
struct strided_functor
{
  int num_bins;
  uint64_t stride;
  uint64_t sample_stride;
  Mapper mapper;

  template <class I>
  __host__ __device__ SampleT operator()(I i) const
  {
    const uint64_t sample = static_cast<uint64_t>(i) / sample_stride;
    int bin               = static_cast<int>((sample * stride) % static_cast<uint64_t>(num_bins));
    return mapper(bin);
  }
};

// sawtooth: rank = stride*i % period, optionally scattered across the bin array.
// The default stride=1, scatter=false path is a monotonically increasing ramp
// that resets to 0 every `period` elements. With period == num_bins this is
// exactly the uniform round-robin tiling (the concentrated:1.0 endpoint);
// smaller periods confine the sweep to a `period`-bin window. A non-unit stride
// changes the visitation order. With scatter=true, the caller maps each rank
// into a contiguous upper-bin window, avoiding a wrapped low-bin tail without
// changing cardinality or visitation order. Every interleaved channel follows
// the full sequence independently.
template <class SampleT, class Mapper>
struct sawtooth_functor
{
  uint64_t period;
  int num_bins;
  uint64_t stride;
  uint64_t sample_stride;
  bool scatter;
  Mapper mapper;

  template <class I>
  __host__ __device__ SampleT operator()(I i) const
  {
    const uint64_t sample = static_cast<uint64_t>(i) / sample_stride;
    const uint64_t rank   = (sample * stride) % period;
    const int bin         = scatter ? upper_window_bin(rank, num_bins) : static_cast<int>(rank);
    return mapper(bin);
  }
};

// poison: each power-of-two sample window starts with a short claim phase that
// alternates two blocker bins, then hammers one poison bin for the rest of the
// window. The blocker bins are selected on the host so their PRIMARY cache
// hashes occupy the poison bin's primary and secondary candidate slots. Once
// those immutable slots are claimed, the poison bin cannot enter either the
// direct single-probe or two-probe cuckoo cache and permanently spills.
//
// `sample_stride` is 1 for single-channel inputs and 4 for the interleaved RGBA
// multi-channel inputs. Dividing by it before selecting a blocker makes every
// active channel alternate between BOTH blocker bins rather than pinning each
// channel to one blocker based on `i % 4`.
template <class SampleT, class Mapper>
struct poison_functor
{
  int blocker_bin0;
  int blocker_bin1;
  int poison_bin;
  uint64_t window_mask;
  uint64_t claim_prefix;
  uint64_t sample_stride;
  bool enabled;
  Mapper mapper;

  template <class I>
  __host__ __device__ SampleT operator()(I i) const
  {
    const uint64_t index = static_cast<uint64_t>(i);
    if (enabled && (index & window_mask) < claim_prefix)
    {
      const int bin = ((index / sample_stride) & 1ull) ? blocker_bin1 : blocker_bin0;
      return mapper(bin);
    }
    return mapper(poison_bin);
  }
};

// temporal_phases: contiguous phases, each centered on one evenly spaced bin,
// with a tunable fraction of samples replaced by uniform random-bin noise.
template <class SampleT, class Mapper>
struct phases_functor
{
  int num_bins;
  int num_phases;
  uint64_t n;
  uint64_t offset;
  double noise;
  uint64_t seed;
  uint64_t sample_stride;
  Mapper mapper;

  template <class I>
  __host__ __device__ SampleT operator()(I i) const
  {
    const uint64_t index       = static_cast<uint64_t>(i);
    const uint64_t channel     = index % sample_stride;
    const uint64_t sample      = index / sample_stride;
    const uint64_t stream_n    = (n + sample_stride - 1 - channel) / sample_stride;
    const uint64_t stream_seed = seed + channel * kChannelSeedStride;
    if (noise > 0.0)
    {
      // Independent hashes choose whether this sample is noise and, if so, its
      // uniformly random bin. At noise=0 the historical pure-phase path remains
      // byte-identical; at noise=1 every sample is random.
      constexpr uint64_t select_salt = 0x74656D706F72616Cull; // "temporal"
      constexpr uint64_t value_salt  = 0x6A69747465725661ull; // "jitterVa"
      const double select            = u01_from_hash(element_key(sample, stream_seed ^ select_salt));
      if (select < noise)
      {
        const double u      = u01_from_hash(element_key(sample, stream_seed ^ value_salt));
        uint64_t random_bin = static_cast<uint64_t>(u * static_cast<double>(num_bins));
        if (random_bin >= static_cast<uint64_t>(num_bins))
        {
          random_bin = static_cast<uint64_t>(num_bins - 1);
        }
        return mapper(static_cast<int>(random_bin));
      }
    }

    uint64_t phase = (sample * static_cast<uint64_t>(num_phases)) / stream_n;
    if (phase >= static_cast<uint64_t>(num_phases))
    {
      phase = static_cast<uint64_t>(num_phases) - 1;
    }
    // Place P phase ranks as evenly as possible across B bins. Computing the
    // product before the division avoids the accumulated error from repeatedly
    // adding floor(B/P)+1. The affine permutation changes their visitation order
    // and rotates them off zero without changing the even spacing when P divides B.
    const uint64_t rank = (phase * static_cast<uint64_t>(num_bins)) / static_cast<uint64_t>(num_phases);
    const int bin       = scatter_bin(rank, num_bins, offset);
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
  uint64_t sample_stride;
  Mapper mapper;

  template <class I>
  __host__ __device__ SampleT operator()(I i) const
  {
    const uint64_t index    = static_cast<uint64_t>(i);
    const uint64_t channel  = index % sample_stride;
    const uint64_t sample   = index / sample_stride;
    const uint64_t stream_n = (n + sample_stride - 1 - channel) / sample_stride;
    const uint64_t j        = permute_index(sample, stream_n, seed + channel * kChannelSeedStride);
    return mapper(static_cast<int>(j % static_cast<uint64_t>(num_bins)));
  }
};

// stale_resident: a cold WORKING SET of `span` (= W) distinct bins in a stable
// upper-bin window, accessed so that EVERY block (whatever its grid-strided slice of the
// input) sees the full working set. It is sized against the per-block no-eviction SMEM
// cache of S slots (W from stale_working_set) to yield a controllable hit rate: the
// cache claims a slot for the FIRST bin that reaches it (a hit for all of that bin's
// accesses) and every other bin that hashes to that occupied slot permanently misses
// (spilling to GMEM). With W > S only a fraction of bins can own a slot, so the hit
// rate falls predictably as W grows -- there is NO eviction and nothing "thrashes".
//
// The per-element key is a well-mixed UNIFORM hash of the per-channel sample index,
// NOT a cyclic `i % span` counter: the hit rate is a PER-BLOCK property (each block
// has its own S-slot cache), and under a grid-stride launch a cyclic counter aliases
// the stride and hands each block only a gcd(total_threads, span)-reduced fraction of the span --
// inflating the measured hit rate. A uniform draw guarantees each block sees all `span`
// distinct keys (for the benchmark's N >> W). Access ORDER does not affect a
// no-eviction cache's hit rate (only the distinct-key set and its slot collisions do),
// so the uniform draw is both faithful to the model and robust to launch geometry.
template <class SampleT, class Mapper>
struct stale_functor
{
  int num_bins;
  uint64_t span; // size of the cold working set (number of distinct bins), = W
  uint64_t seed;
  uint64_t sample_stride;
  Mapper mapper;

  template <class I>
  __host__ __device__ SampleT operator()(I i) const
  {
    // Uniform key in [0, span): a well-mixed hash of the per-channel sample
    // index, so any grid-strided per-block slice still covers all `span` keys
    // approximately uniformly.
    const uint64_t index       = static_cast<uint64_t>(i);
    const uint64_t channel     = index % sample_stride;
    const uint64_t sample      = index / sample_stride;
    const uint64_t stream_seed = seed + channel * kChannelSeedStride;
    const double u             = u01_from_hash(element_key(sample, stream_seed ^ 0x5ADE57A1E00FF5E7ull));
    uint64_t k                 = static_cast<uint64_t>(u * static_cast<double>(span));
    if (k >= span)
    {
      k = span - 1; // guard the u==1-epsilon corner
    }
    const int bin = upper_window_bin(k, num_bins);
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
constexpr double kDefaultHashSynonymHotShare = 0.9; // bare "hash_synonym"
constexpr double kDefaultTemporalNoise       = 0.10; // bare "temporal_phases"
constexpr int kDefaultTemporalPhases         = 8;
constexpr double kDefaultStaleHitRate        = 0.5; // bare "stale_resident"
constexpr uint64_t kDefaultStridedStride     = 9973ull; // bare "strided_sweep"
// bare "sawtooth" => period 0 sentinel, resolved to num_bins at generation time.
constexpr uint64_t kDefaultSawtoothPeriod = 0ull;
constexpr uint64_t kDefaultSawtoothStride = 1ull;
constexpr bool kDefaultSawtoothScatter    = false;

// B200 poison schedule. A direct-atomic launch has up to 303104 sample/pixel
// threads (single-channel RANGE and the 4-sample/pixel multi paths; EVEN uses
// 227328). These claim prefixes cover every thread's first (u=0) access, so every
// warp claims both blocker bins before it can reach a poison sample. Repeating
// the phase once per minimum-size 2^24-pixel benchmark window keeps the shape
// valid for persistent grid-stride launches at every larger N.
constexpr uint64_t kPoisonSingleWindow      = 1ull << 24;
constexpr uint64_t kPoisonSingleClaimPrefix = 5ull << 16; // > 303104 samples
constexpr uint64_t kPoisonMultiWindow       = 1ull << 26; // 4 * 2^24 interleaved samples
constexpr uint64_t kPoisonMultiClaimPrefix  = 5ull << 18; // > 4 * 303104 samples

// Resolve the cache size supplied by the compiled dispatch policy. The generic
// environment override keeps generator overlays byte-identical to the branch when
// benchmarking an upstream binary without the query hook. Honor the former
// stale-specific name as a compatibility fallback for existing run drivers.
inline int64_t resolve_input_cache_slots(int64_t cache_slots)
{
  const char* env = std::getenv("CUB_HISTO_INPUT_CACHE_SLOTS");
  if (env == nullptr)
  {
    env = std::getenv("CUB_HISTO_STALE_SLOTS");
  }
  if (env != nullptr)
  {
    const long long forced = std::atoll(env);
    if (forced > 0)
    {
      return static_cast<int64_t>(forced);
    }
  }
  return cache_slots;
}

// Host-side use of the cache's actual source-of-truth hash function and constants.
// `cache_slot_from_hash` is shared with the device probe implementation, so changing
// CUB_HISTO_CACHE_HASH_MODE or either multiplier automatically changes this generator.
inline int benchmark_cache_slot(int bin, int slots, uint32_t multiplier)
{
  if (slots <= 0 || (slots & (slots - 1)) != 0)
  {
    throw std::runtime_error("Histogram input cache slots must be a positive power of two");
  }
  int slot_log2 = 0;
  for (int value = slots; value > 1; value >>= 1)
  {
    ++slot_log2;
  }
  const uint32_t product = static_cast<uint32_t>(bin) * multiplier;
  return cub::detail::histogram::cache_slot_from_hash(product, slots - 1, slot_log2);
}

// Find up to kHashSynonymCount real synonyms of one PRIMARY cache slot. Choose a
// maximally populated slot (ties broken deterministically from `seed`) so small B/S
// ratios still exercise as many colliding keys as the bin domain permits.
inline std::vector<int> find_hash_synonyms(int num_bins, int slots, uint64_t seed)
{
  std::vector<int> occupancy(static_cast<std::size_t>(slots), 0);
  for (int bin = 0; bin < num_bins; ++bin)
  {
    const int slot = benchmark_cache_slot(bin, slots, cub::detail::histogram::cache_primary_hash_multiplier);
    ++occupancy[static_cast<std::size_t>(slot)];
  }

  int max_occupancy = 0;
  for (const int count : occupancy)
  {
    max_occupancy = std::max(max_occupancy, count);
  }
  const int start = static_cast<int>(seed % static_cast<uint64_t>(slots));
  int target_slot = start;
  for (int delta = 0; delta < slots; ++delta)
  {
    const int slot = (start + delta) & (slots - 1);
    if (occupancy[static_cast<std::size_t>(slot)] == max_occupancy)
    {
      target_slot = slot;
      break;
    }
  }

  std::vector<int> synonyms;
  synonyms.reserve(static_cast<std::size_t>(std::min(kHashSynonymCount, max_occupancy)));
  for (int bin = 0; bin < num_bins && static_cast<int>(synonyms.size()) < kHashSynonymCount; ++bin)
  {
    if (benchmark_cache_slot(bin, slots, cub::detail::histogram::cache_primary_hash_multiplier) == target_slot)
    {
      synonyms.push_back(bin);
    }
  }
  return synonyms;
}

// Build the per-bin pmf for an i.i.d. distribution shape, honoring the spec's
// knob. Hot ranks are scattered across bins via scatter_bin() so the mode is
// not forced to bin 0.
inline std::vector<double> build_pmf(const ShapeSpec& spec, int num_bins, uint64_t seed, int64_t cache_slots)
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
    case InputShape::hash_synonym: {
      // KNOB = hot share. Up to kHashSynonymCount bins that ACTUALLY collide on
      // one primary cache slot share the hot traffic; the rest is uniform
      // background. S comes from the compiled policy's runtime query.
      const double hot_share = knob_or(spec, kDefaultHashSynonymHotShare);
      if (cache_slots <= 0)
      {
        throw std::runtime_error("hash_synonym requires the runtime histogram cache-slot count");
      }
      const std::vector<int> syn = find_hash_synonyms(num_bins, static_cast<int>(cache_slots), seed);
      const double bg            = (1.0 - hot_share) / num_bins;
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

struct poison_bin_set
{
  int blocker0 = 0;
  int blocker1 = 0;
  int poison   = 0;
  bool valid   = false;
};

// Find a poison bin P and blocker bins A/B such that
//
//   primary_slot(A) == primary_slot(P)
//   primary_slot(B) == secondary_slot(P).
//
// A/B therefore occupy both immutable cuckoo candidates before P arrives; A
// alone also poisons the direct-mapped single-probe cache. At the >=262144-bin
// gated-cuckoo tier the secondary probe is compiled out, so the same A remains
// sufficient and B is harmless. The high-bin sweep has at least four bins per
// cache slot (B >= 32768, S <= 8192), making a solution plentiful. For B <= S
// poison is not a meaningful capacity adversary, and the caller deliberately
// falls back to a valid constant input.
inline poison_bin_set find_poison_bins(int num_bins, int slots)
{
  poison_bin_set result{};
  result.poison = num_bins > 0 ? num_bins - 1 : 0;
  if (num_bins <= slots || slots <= 0 || (slots & (slots - 1)) != 0)
  {
    return result;
  }

  // Keep two representatives per primary slot. One may be P itself; the other
  // is then its blocker. One representative is enough for P's distinct
  // secondary slot because a bin cannot belong to two different primary slots.
  std::vector<int> owner0(static_cast<std::size_t>(slots), -1);
  std::vector<int> owner1(static_cast<std::size_t>(slots), -1);
  for (int bin = 0; bin < num_bins; ++bin)
  {
    const int slot = benchmark_cache_slot(bin, slots, cub::detail::histogram::cache_primary_hash_multiplier);
    if (owner0[slot] == -1)
    {
      owner0[slot] = bin;
    }
    else if (owner1[slot] == -1)
    {
      owner1[slot] = bin;
    }
  }

  // Prefer a high poison bin, but search all bins so unusual B/S combinations
  // still get a valid construction whenever one exists.
  for (int poison = num_bins - 1; poison >= 0; --poison)
  {
    const int primary_slot = benchmark_cache_slot(poison, slots, cub::detail::histogram::cache_primary_hash_multiplier);
    const int secondary_slot =
      benchmark_cache_slot(poison, slots, cub::detail::histogram::cache_secondary_hash_multiplier);
    const int blocker0 = (owner0[primary_slot] == poison) ? owner1[primary_slot] : owner0[primary_slot];
    if (blocker0 < 0)
    {
      continue;
    }

    // If both poison hashes name the same slot, one blocker fills both roles.
    const int blocker1 = (secondary_slot == primary_slot) ? blocker0 : owner0[secondary_slot];
    if (blocker1 < 0)
    {
      continue;
    }

    result.blocker0 = blocker0;
    result.blocker1 = blocker1;
    result.poison   = poison;
    result.valid    = true;
    return result;
  }
  return result;
}

// stale_resident working-set size W for a target cache hit rate X against S no-eviction
// slots.
//
// MECHANISM. The per-block SMEM cache has S slots and never evicts: the FIRST distinct
// bin to reach a slot claims it (all of that bin's accesses then hit block-scope), and
// every OTHER bin that hashes to the same slot permanently misses (spills to GMEM). With
// W equiprobable distinct bins, the fraction of accesses that hit equals the fraction of
// the W bins that own a slot.
//
// CALIBRATED LAW (measured -- do NOT assume the naive occupancy curve). One might expect
// the random balls-into-bins hit rate H = (S/W)(1 - (1-1/S)^W) ~= (1-e^{-r})/r, r=W/S.
// That is WRONG here: the bins are scattered by an AFFINE map (scatter_bin: bin =
// multiplier*rank + offset mod num_bins) and then hashed to slots by the cache's
// MULTIPLICATIVE Fibonacci hash, and that composition spreads the W ranks across the S
// slots ANTI-collisively -- far more evenly than independent random draws. Measured on
// B200 (hit-rate-instrumented direct_single_probe AND direct_cuckoo, converged large-N,
// bins 32768..1048576, I32+F64, S in {1024,4096,8192}; see the run_2026-07-07
// calibration). Two robust facts:
//
//   1. The curve is a function of r = W/S ONLY (S-invariant to <1% across all three S).
//   2. Once the set overflows the cache (r >~ 1.34, i.e. X <~ 0.744) the hit rate is
//
//          H(W)  ==  S / W        (perfect spread),
//
//      measured accurate to <1% over the whole ADVERSARIAL range -- e.g. W=2S -> 0.50,
//      W=4S -> 0.25, W=1.33S -> 0.75. So for the target range that matters we invert
//      exactly: W = round(S / X).
//
// Below r ~= 1.34 (near-resident, X > 0.744) the rate bends up toward 1.0 (a finite
// anti-collisive cache still suffers a few birthday collisions, so it never quite
// reaches 1.0 until the set is far smaller than S). S/X would UNDERSHOOT there, so we
// invert the MEASURED near-resident curve via the small S-invariant table below
// (kStaleNearRate/kStaleNearR), continuous with S/X at the X=0.744 seam.
//
// NOTE the hit rate is a converged / large-N property: at small N a block processes too
// few pixels to see all W distinct keys (coupon-collector under-coverage), so the rate
// runs high; it settles onto this law once N >> W (true for the benchmark's element grid
// at the bins these cached kernels run). The knob targets that converged rate.

// Measured near-resident branch, r = W/S vs hit rate (S-invariant; B200, converged N).
// rate DEScending, r AScending. Seam point (0.744, 1.344) coincides with S/X.
inline constexpr double kStaleNearRate[] = {1.000, 0.990, 0.944, 0.888, 0.800, 0.744};
inline constexpr double kStaleNearR[]    = {0.300, 0.519, 0.697, 0.898, 1.126, 1.344};
inline constexpr int kStaleNearN         = 6;

inline uint64_t stale_working_set(double target_hit_rate, int64_t cache_slots, int num_bins)
{
  const double S = static_cast<double>(cache_slots > 0 ? cache_slots : 1);
  double X       = target_hit_rate;
  // Clamp into the achievable interval; X->0 diverges, and X at/above the finite cache's
  // collision ceiling is pinned to the smallest tabulated working set.
  if (X > 0.999)
  {
    X = 0.999;
  }
  if (X < 1e-3)
  {
    X = 1e-3;
  }

  double r; // = W / S
  constexpr double kStaleSpreadX = 0.744; // seam: at/below here rate == S/W exactly
  if (X <= kStaleSpreadX)
  {
    r = 1.0 / X; // perfect-spread inverse, W = S / X
  }
  else
  {
    // Invert the measured near-resident table: find X in kStaleNearRate (descending) and
    // linearly interpolate the corresponding r. Above the top sample (X > ~0.99) pin to
    // the smallest tabulated r (the collision ceiling); the arrays bracket otherwise.
    if (X >= kStaleNearRate[0])
    {
      r = kStaleNearR[0];
    }
    else
    {
      r = kStaleNearR[kStaleNearN - 1];
      for (int i = 0; i < kStaleNearN - 1; ++i)
      {
        // Descending: kStaleNearRate[i] >= X > kStaleNearRate[i+1].
        if (X <= kStaleNearRate[i] && X > kStaleNearRate[i + 1])
        {
          const double t = (kStaleNearRate[i] - X) / (kStaleNearRate[i] - kStaleNearRate[i + 1]);
          r              = kStaleNearR[i] + t * (kStaleNearR[i + 1] - kStaleNearR[i]);
          break;
        }
      }
    }
  }

  int64_t w_int = static_cast<int64_t>(std::llround(r * S));
  if (w_int < 1)
  {
    w_int = 1;
  }
  if (w_int > num_bins)
  {
    w_int = num_bins; // cannot have more distinct bins than exist
  }
  return static_cast<uint64_t>(w_int);
}

inline bool is_ordering_shape(InputShape shape)
{
  return shape == InputShape::stale_resident || shape == InputShape::temporal_phases
      || shape == InputShape::strided_sweep || shape == InputShape::sawtooth || shape == InputShape::poison;
}

// ---------------------------------------------------------------------------
// Core generation: given a bin->value mapper, fill an output device vector of
// `n` samples according to `spec`.
// ---------------------------------------------------------------------------
template <class SampleT, class OffsetT, class Mapper>
thrust::device_vector<SampleT> generate_shape_impl(
  const ShapeSpec& spec,
  OffsetT n,
  int num_bins,
  Mapper mapper,
  uint64_t seed,
  int64_t cache_slots,
  int64_t sample_stride)
{
  if (sample_stride <= 0)
  {
    throw std::runtime_error("Histogram input sample stride must be positive");
  }
  thrust::device_vector<SampleT> out(static_cast<std::size_t>(n));
  const uint64_t offset              = seed % static_cast<uint64_t>(num_bins);
  const int64_t resolved_cache_slots = resolve_input_cache_slots(cache_slots);
  const uint64_t stream_stride       = static_cast<uint64_t>(sample_stride);

  // Exact-uniform endpoint: every bin gets exactly n/num_bins (+-1) samples, in
  // a pseudo-random sequence order (a Feistel permutation of the round-robin
  // tiling). Uniform counts, randomly distributed in the input -- not the
  // sequential ramp (that is the `sawtooth` shape). Emitted directly rather than
  // via i.i.d. uniform sampling so the per-bin counts are exact (no multinomial
  // noise / empty bins when bins ~= elements).
  if (spec.shape == InputShape::concentrated && spec.has_knob && spec.knob >= 1.0)
  {
    shuffled_uniform_functor<SampleT, Mapper> fn{static_cast<uint64_t>(n), num_bins, seed, stream_stride, mapper};
    thrust::tabulate(out.begin(), out.end(), fn);
    return out;
  }

  if (!is_ordering_shape(spec.shape))
  {
    // Distribution shape: host pmf -> inclusive CDF -> device -> inverse-CDF sample.
    std::vector<double> pmf = build_pmf(spec, num_bins, seed, resolved_cache_slots);
    thrust::host_vector<double> h_cdf(num_bins);
    double acc = 0.0;
    for (int b = 0; b < num_bins; ++b)
    {
      acc += pmf[b];
      h_cdf[b] = acc;
    }
    h_cdf[num_bins - 1]                 = 1.0; // guard against fp drift on the last bin
    thrust::device_vector<double> d_cdf = h_cdf;
    cdf_sample_functor<SampleT, Mapper> fn{
      thrust::raw_pointer_cast(d_cdf.data()), num_bins, seed, stream_stride, mapper};
    thrust::tabulate(out.begin(), out.end(), fn);
    return out;
  }

  switch (spec.shape)
  {
    case InputShape::strided_sweep: {
      // KNOB = stride.
      const uint64_t stride = spec.has_knob ? static_cast<uint64_t>(std::llround(spec.knob)) : kDefaultStridedStride;
      strided_functor<SampleT, Mapper> fn{num_bins, stride, stream_stride, mapper};
      thrust::tabulate(out.begin(), out.end(), fn);
      break;
    }
    case InputShape::sawtooth: {
      // KNOBS = period:stride:scatter. A zero/default period means a full
      // num_bins sweep; the other defaults preserve the classic monotonic ramp.
      const uint64_t requested =
        spec.has_knob ? static_cast<uint64_t>(std::llround(spec.knob)) : kDefaultSawtoothPeriod;
      const uint64_t period =
        (requested == 0) ? static_cast<uint64_t>(num_bins) : std::min(requested, static_cast<uint64_t>(num_bins));
      const uint64_t stride = spec.has_knob2 ? static_cast<uint64_t>(std::llround(spec.knob2)) : kDefaultSawtoothStride;
      const bool scatter    = spec.has_knob3 ? (std::llround(spec.knob3) != 0) : kDefaultSawtoothScatter;
      sawtooth_functor<SampleT, Mapper> fn{period, num_bins, stride, stream_stride, scatter, mapper};
      thrust::tabulate(out.begin(), out.end(), fn);
      break;
    }
    case InputShape::poison: {
      // A literal "fill all S slots, then poison" construction cannot meet the
      // shape's near-zero-hit goal at the smallest measured N: every empty-slot
      // claim is instrumented as a hit, so on B200 single-channel N=2^24 would
      // have at least 296 blocks * 8192 claims / 2^24 = 14.45% hits. Instead,
      // occupy exactly the poison bin's candidate slot(s): one blocker for the
      // single-probe / gated-cuckoo path, plus one for full two-probe cuckoo.
      // This is sufficient because cache keys are immutable after their claim.
      //
      // The periodic claim prefix covers every block's u=0 indices. Each warp's
      // u=0 inputs alternate both blockers before that warp reaches u=1 poison
      // inputs, avoiding a scheduling race in which P could claim an empty slot.
      // The resulting claim fraction is exactly 1.953125% for N >= 2^24 pixels,
      // with every remaining contribution forced to the same device-scope
      // output atomic.
      const int64_t slots = (resolved_cache_slots > 0) ? resolved_cache_slots : kAdversarialCacheSlots;

      const bool multi_layout     = stream_stride > 1;
      const uint64_t window       = multi_layout ? kPoisonMultiWindow : kPoisonSingleWindow;
      const uint64_t claim_prefix = multi_layout ? kPoisonMultiClaimPrefix : kPoisonSingleClaimPrefix;
      const poison_bin_set bins   = (slots < num_bins) ? find_poison_bins(num_bins, static_cast<int>(slots))
                                                       : poison_bin_set{0, 0, num_bins - 1, false};
      poison_functor<SampleT, Mapper> fn{
        bins.blocker0, bins.blocker1, bins.poison, window - 1, claim_prefix, stream_stride, bins.valid, mapper};
      thrust::tabulate(out.begin(), out.end(), fn);
      break;
    }
    case InputShape::temporal_phases: {
      // KNOB = uniform-random noise fraction in [0,1]; optional KNOB2 = phases.
      const double noise  = std::max(0.0, std::min(knob_or(spec, kDefaultTemporalNoise), 1.0));
      const int requested = spec.has_knob2 ? static_cast<int>(std::llround(spec.knob2)) : kDefaultTemporalPhases;
      const int phases    = std::max(1, std::min(requested, num_bins));
      phases_functor<SampleT, Mapper> fn{
        num_bins, phases, static_cast<uint64_t>(n), offset, noise, seed, stream_stride, mapper};
      thrust::tabulate(out.begin(), out.end(), fn);
      break;
    }
    case InputShape::stale_resident: {
      // KNOB = target cache HIT RATE in (0,1] (default 0.5). The working set of W
      // distinct bins is solved from the ACTUAL per-block cache slot count S
      // (`cache_slots`, queried at runtime from CUB's occupancy sizer and passed in;
      // falls back to the single-channel floor if unavailable) so the MEASURED hit
      // rate tracks the knob: W = stale_working_set(X, S). Larger W (lower X) means
      // more bins competing for the S no-eviction slots, so more permanent misses.
      const double target = knob_or(spec, kDefaultStaleHitRate);
      // Slot count S that sizes the working set. Normally the caller passes the value
      // queried from CUB's occupancy sizer (`cache_slots`).
      // CUB_HISTO_INPUT_CACHE_SLOTS overrides it with a fixed S so a build WITHOUT
      // the query hook (e.g. the stock-`main` baseline binary, whose dispatch has no
      // direct-atomic cache) can
      // generate the byte-IDENTICAL working set the branch used -- keeping the shape
      // apples-to-apples across the two dispatch variants. Falls back to the passed
      // value, then the single-channel floor.
      const int64_t slots = (resolved_cache_slots > 0) ? resolved_cache_slots : kAdversarialCacheSlots;
      const uint64_t span = stale_working_set(target, slots, num_bins);
      stale_functor<SampleT, Mapper> fn{num_bins, span, seed, stream_stride, mapper};
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
// `cache_slots` is the per-block direct-atomic SMEM cache slot count S used to size the
// hash_synonym primary collisions, the stale_resident working set, and poison's
// blocker bins; pass the value from
// cub::detail::histogram::query_direct_atomic_cache_slots (0 or negative => legacy
// fallbacks where available). It is ignored by every other shape.
// `sample_stride` describes interleaved multi-channel storage (1 for single,
// 4 for RGBA); every channel receives an independent full shape rather than a
// strided subset. Both layout parameters are trailing defaults so legacy
// callers remain source-compatible.
template <class SampleT, class OffsetT>
thrust::device_vector<SampleT> generate_histogram_input_even(
  const ShapeSpec& spec,
  OffsetT n,
  int num_bins,
  SampleT lower,
  SampleT upper,
  uint64_t seed         = 42,
  int64_t cache_slots   = 0,
  int64_t sample_stride = 1)
{
  const double w = (static_cast<double>(upper) - static_cast<double>(lower)) / static_cast<double>(num_bins);
  even_bin_to_value<SampleT> mapper{static_cast<double>(lower), w, lower, upper};
  return generate_shape_impl<SampleT, OffsetT>(spec, n, num_bins, mapper, seed, cache_slots, sample_stride);
}

template <class SampleT, class OffsetT>
thrust::device_vector<SampleT> generate_histogram_input_range(
  const ShapeSpec& spec,
  OffsetT n,
  int num_bins,
  const SampleT* d_levels,
  uint64_t seed         = 42,
  int64_t cache_slots   = 0,
  int64_t sample_stride = 1)
{
  range_bin_to_value<SampleT> mapper{d_levels, num_bins};
  return generate_shape_impl<SampleT, OffsetT>(spec, n, num_bins, mapper, seed, cache_slots, sample_stride);
}
