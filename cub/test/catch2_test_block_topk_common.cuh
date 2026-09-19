// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared helpers for block-level top-k API tests: deterministic and
// random key generation (incl. fine control over the top-k boundary),
// host-side top-k references, and small CUDA-test glue.

#pragma once

#include <thrust/random.h>
#include <thrust/shuffle.h>

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__bit/bit_cast.h>
#include <cuda/std/__cmath/isfinite.h>
#include <cuda/std/__cmath/rounding_functions.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/optional>
#include <cuda/std/span>
#include <cuda/type_traits>

#include <algorithm>
#include <initializer_list>

#include "catch2_test_device_topk_common.cuh"
#include <c2h/catch2_test_helper.h>

// --- RNG and data generation ---

// Shared RNG type; each test constructs one and threads it through every
// helper that consumes randomness.
using rng_t = thrust::default_random_engine;

// `n` distinct, shuffled values of `T` centered at `0` (or `max() / 2` for unsigned).
template <typename T>
c2h::host_vector<T> distinct_keys(cuda::std::size_t n, rng_t& rng)
{
  constexpr T mean = cuda::std::is_signed_v<T> ? T{0} : (cuda::std::numeric_limits<T>::max() / T{2});
  const T start    = static_cast<T>(mean - static_cast<T>(n / 2)); // NOLINT(bugprone-integer-division)
  c2h::host_vector<T> v(n);
  for (cuda::std::size_t i = 0; i < n; ++i)
  {
    v[i] = static_cast<T>(start + static_cast<T>(i));
  }
  thrust::shuffle(v.begin(), v.end(), rng);
  return v;
}

// Random `boundary_key` valid for `gen_keys_from_boundary_key<KeyT>`
// (strictly inside the representable range; FP halves the span to satisfy
// `uniform_real_distribution`'s `b - a <= max()` precondition).
template <typename KeyT>
KeyT random_boundary_key(rng_t& rng)
{
  if constexpr (cuda::std::is_integral_v<KeyT>)
  {
    constexpr KeyT type_lo = cuda::std::numeric_limits<KeyT>::lowest();
    constexpr KeyT type_hi = cuda::std::numeric_limits<KeyT>::max();
    thrust::random::uniform_int_distribution<KeyT> dist(
      static_cast<KeyT>(type_lo + KeyT{1}), static_cast<KeyT>(type_hi - KeyT{1}));
    return dist(rng);
  }
  else
  {
    // FP: half-span to satisfy `uniform_real_distribution`'s `b - a <= max()` precondition.
    constexpr KeyT type_hi = cuda::std::numeric_limits<KeyT>::max() / KeyT{2};
    constexpr KeyT type_lo = -type_hi;
    thrust::random::uniform_real_distribution<KeyT> dist(
      cuda::std::nextafter(type_lo, KeyT{0}), cuda::std::nextafter(type_hi, KeyT{0}));
    return dist(rng);
  }
}

// Generates `n` keys with knobs for top-k boundary behavior: the boundary
// value (`boundary_key`), the number of boundary ties (`num_tied >=
// overhang + 1`), and built-in coverage of near-boundary values
// (`boundary_succ` / `boundary_pred`) and FP specials (`+/-inf`, and a
// `+0.0` / `-0.0` mix when `boundary_key == 0.0`). Remaining slots are
// random above-/below-k fillers.
//
// `overhang > 0` forces ranking to pick a `num_tied - overhang` subset of
// boundary copies; `overhang == 0` only fixes the *order* of selected
// items. Requires `effective_k + overhang <= num_items` (with
// `effective_k = min(k, num_items)`). `boundary_key` must be strictly
// inside the integer range, or finite for FP (`lowest()` / `max()`
// collapse the corresponding above-/below-k slots to `+/-inf`).
template <bool SelectMax, typename KeyT>
c2h::host_vector<KeyT> gen_keys_from_boundary_key(int num_items, int k, int overhang, KeyT boundary_key, rng_t& rng)
{
  REQUIRE(0 < k);
  REQUIRE(0 <= overhang);
  const int effective_k = cuda::std::min(k, num_items);
  REQUIRE(effective_k + overhang <= num_items);

  // overhang only specifies the lower bound of the number of tied keys.
  thrust::random::uniform_int_distribution<int> num_tied_dist(overhang + 1, effective_k + overhang);
  const int num_tied        = num_tied_dist(rng);
  const int num_after_sieve = effective_k + overhang;
  const int num_winners     = num_after_sieve - num_tied;
  const int num_losers      = num_items - num_after_sieve;

  // Map selection direction onto the two sides of `boundary_key` once;
  // rest is direction-agnostic.
  const int num_above_k = SelectMax ? num_winners : num_losers;
  const int num_below_k = SelectMax ? num_losers : num_winners;
  REQUIRE(num_above_k + num_tied + num_below_k == num_items);

  c2h::host_vector<KeyT> keys(num_items);

  // Slot 0 of each non-empty side is pinned to `boundary_succ` /
  // `boundary_pred` so the sieve discriminates adjacent bit patterns.
  auto fill_keys = [&](auto above_gen, auto below_gen, auto boundary_key_gen) {
    int idx = 0;
    for (int i = 0; i < num_above_k; ++i, ++idx)
    {
      keys[idx] = above_gen(i);
    }
    for (int i = 0; i < num_below_k; ++i, ++idx)
    {
      keys[idx] = below_gen(i);
    }
    for (int i = 0; i < num_tied; ++i, ++idx)
    {
      keys[idx] = boundary_key_gen(i);
    }
  };

  if constexpr (cuda::std::is_integral_v<KeyT>)
  {
    constexpr KeyT type_lo = cuda::std::numeric_limits<KeyT>::lowest();
    constexpr KeyT type_hi = cuda::std::numeric_limits<KeyT>::max();

    // Strict inequality so `boundary_key +/- 1` stay representable.
    REQUIRE(type_lo < boundary_key);
    REQUIRE(boundary_key < type_hi);
    const KeyT boundary_succ = static_cast<KeyT>(boundary_key + KeyT{1});
    const KeyT boundary_pred = static_cast<KeyT>(boundary_key - KeyT{1});

    thrust::random::uniform_int_distribution<KeyT> above_dist(boundary_succ, type_hi);
    thrust::random::uniform_int_distribution<KeyT> below_dist(type_lo, boundary_pred);

    // First slot of each side is pinned to `boundary_succ` / `boundary_pred` so the sieve discriminates adjacent bit
    // patterns.
    fill_keys(
      [&](int i) {
        return (i == 0) ? boundary_succ : above_dist(rng);
      },
      [&](int i) {
        return (i == 0) ? boundary_pred : below_dist(rng);
      },
      [&](int) {
        return boundary_key;
      });
  }
  else
  {
    constexpr KeyT inf      = cuda::std::numeric_limits<KeyT>::infinity();
    constexpr KeyT type_max = cuda::std::numeric_limits<KeyT>::max();
    constexpr KeyT type_min = -type_max;

    REQUIRE(cuda::std::isfinite(boundary_key));

    // `boundary_key == max()` / `lowest()` saturates `nextafter` to
    // `+/-inf` (slot 0 then carries `+/-inf` for free); the non-edge side
    // uses the optional, clamped to `max()` to satisfy
    // `uniform_real_distribution`'s `b - a <= max()` precondition.
    const KeyT boundary_succ = cuda::std::nextafter(boundary_key, +inf);
    const KeyT boundary_pred = cuda::std::nextafter(boundary_key, -inf);

    // Make sure the distributions are valid.
    cuda::std::optional<thrust::random::uniform_real_distribution<KeyT>> above_dist;
    cuda::std::optional<thrust::random::uniform_real_distribution<KeyT>> below_dist;
    if (boundary_key != type_max)
    {
      above_dist.emplace(boundary_succ, cuda::std::min(boundary_succ + type_max, type_max));
    }
    if (boundary_key != type_min)
    {
      below_dist.emplace(cuda::std::max(boundary_pred - type_max, type_min), boundary_pred);
    }

    fill_keys(
      [&](int i) -> KeyT {
        if (i == 0)
        {
          return boundary_succ;
        }
        else if (i == 1)
        {
          return inf;
        }
        else
        {
          return above_dist ? (*above_dist)(rng) : inf;
        }
      },
      [&](int i) -> KeyT {
        if (i == 0)
        {
          return boundary_pred;
        }
        else if (i == 1)
        {
          return -inf;
        }
        else
        {
          return below_dist ? (*below_dist)(rng) : -inf;
        }
      },
      // For `boundary_key == +/-0.0`, alternate signs to mix `+0.0` and
      // `-0.0` ties; otherwise every tied slot is identical.
      [&](int i) -> KeyT {
        if ((i % 2 == 0) || (boundary_key != KeyT{0.0}))
        {
          return boundary_key;
        }
        return -boundary_key;
      });
  }

  thrust::shuffle(keys.begin(), keys.end(), rng);
  return keys;
}

// Catch2 generator over type-specific deterministic `boundary_key` values:
// FP gets `0.0`, the smallest positive/negative values, the subnormal/
// normal boundary on each side, and `max()` / `lowest()`. Signed integers
// get just `0`; unsigned integers get `max() / 2` (the top-bit transition).
template <typename KeyT>
auto boundary_key_generator()
{
  using namespace Catch::Generators;
  if constexpr (cuda::is_floating_point_v<KeyT>)
  {
    constexpr KeyT inf     = cuda::std::numeric_limits<KeyT>::infinity();
    constexpr KeyT pos_min = cuda::std::numeric_limits<KeyT>::min();
    return values<KeyT>({
      KeyT{0}, // -0.0 is automatically generated by the FP branch of gen_keys_from_boundary_key
      cuda::std::nextafter(KeyT{0}, +inf), // smallest finite positive value (possibly subnormal)
      cuda::std::nextafter(KeyT{0}, -inf), // largest finite negative value (possibly subnormal)
      // Subnormal/normal boundary on each side -- the bit-cast unsigned representation jumps when the exponent flips
      // from all-zeros to a leading 1.
      cuda::std::nextafter(pos_min, KeyT{0}), // largest positive subnormal
      pos_min, // smallest positive normal
      -pos_min, // largest negative normal (closest to 0)
      cuda::std::nextafter(-pos_min, KeyT{0}), // largest-magnitude negative subnormal
      cuda::std::numeric_limits<KeyT>::max(), // largest finite -- collapses above-k slots to +inf
      cuda::std::numeric_limits<KeyT>::lowest(), // smallest finite (== -max()) -- collapses below-k slots to -inf
    });
  }
  else
  {
    constexpr KeyT fixed = cuda::std::is_signed_v<KeyT> ? KeyT{0} : (cuda::std::numeric_limits<KeyT>::max() / KeyT{2});
    return values<KeyT>({fixed});
  }
}

// Catch2 generator over `options`, narrowed to the leading entry when
// `narrow` (callers must list the always-applicable value first).
inline auto overhang_generator(bool narrow, std::initializer_list<int> options)
{
  using namespace Catch::Generators;
  return narrow ? take(1, values<int>(options)) : values<int>(options);
}

// --- Host top-k reference ---

// Up to `min(k, in.size())` items so callers can pass a `k` exceeding the input length.
template <bool SelectMax, typename T>
c2h::host_vector<T> sorted_top_k(const c2h::host_vector<T>& in, int k)
{
  constexpr auto direction = SelectMax ? cub::detail::topk::select::max : cub::detail::topk::select::min;
  c2h::host_vector<T> ref  = in;
  const auto out_size      = cuda::std::min(static_cast<cuda::std::size_t>(k), ref.size());
  std::partial_sort(ref.begin(), ref.begin() + out_size, ref.end(), direction_to_comparator_t<direction>{});
  ref.resize(out_size);
  return ref;
}

// --- Kernel-launch glue ---

// Wraps a thrust-compatible contiguous container as a dynamic-extent
// `cuda::std::span` (static-extent spans are built manually at call sites).
template <typename Container>
auto to_span(Container&& v)
{
  using element_t = cuda::std::remove_pointer_t<decltype(cuda::std::to_address(v.data()))>;
  return cuda::std::span<element_t>{cuda::std::to_address(v.data()), v.size()};
}

// --- Post-processing / comparison ---

// For FP `T`, returns each element's same-sized unsigned-int bit pattern
// (no-op for non-FP `T`). Used in CAPTURE so mismatch diagnostics stay
// readable when values would otherwise print as `0.0` (e.g. `+/-0.0`). Note
// that `host_vector` equality treats `-0.0` and `+0.0` as equal; the
// dedicated `-0.0` regression test's `signbit` count is the guard for that.
template <typename T>
auto bit_repr(const c2h::host_vector<T>& v)
{
  if constexpr (cuda::is_floating_point_v<T>)
  {
    using U = cuda::std::conditional_t<
      sizeof(T) == sizeof(cuda::std::uint64_t),
      cuda::std::uint64_t,
      cuda::std::conditional_t<sizeof(T) == sizeof(cuda::std::uint32_t), cuda::std::uint32_t, cuda::std::uint16_t>>;
    static_assert(sizeof(U) == sizeof(T), "bit_repr: no matching unsigned-int width for T");
    c2h::host_vector<U> out(v.size());
    for (cuda::std::size_t i = 0; i < v.size(); ++i)
    {
      out[i] = cuda::std::bit_cast<U>(v[i]);
    }
    return out;
  }
  else
  {
    return v;
  }
}
