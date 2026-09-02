//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/algorithm>
#include <cuda/buffer>
#include <cuda/hierarchy>
#include <cuda/iterator>
#include <cuda/launch>
#include <cuda/memory_resource>
#include <cuda/std/cstddef>
#include <cuda/std/limits>
#include <cuda/std/mdspan>
#include <cuda/std/random>
#include <cuda/std/span>
#include <cuda/stream>

#include <cmath>
#include <unordered_map>
#include <vector>

#include "test_macros.h"
#include <c2h/catch2_test_helper.h>

namespace
{
template <class Rng>
struct sample_kernel
{
  cuda::std::size_t n_;
  cuda::std::size_t k_;
  Rng rng_;

  TEST_DEVICE_FUNC void operator()(cuda::std::span<cuda::std::size_t> out, cuda::std::span<cuda::std::size_t> written)
  {
    if (cuda::gpu_thread.rank(cuda::grid) != 0)
    {
      return;
    }

    cuda::counting_iterator<cuda::std::size_t> first{0};
    cuda::counting_iterator<cuda::std::size_t> last{n_};

    auto end   = cuda::sample(first, last, out.begin(), k_, rng_);
    written[0] = end - out.begin();
  }
};

// Samples `k` indices out of a population of `n` on the device, then returns the result on the
// host. The population is 0, 1, ... n - 1, so each selected element is also its own index.
template <class Rng = cuda::std::philox4x64>
std::vector<cuda::std::size_t> device_sample(cuda::std::size_t n, cuda::std::size_t k, Rng rng = Rng{})
{
  auto stream   = cuda::stream{cuda::device_ref{0}};
  auto resource = cuda::device_default_memory_pool(cuda::device_ref{0});

  const auto expected = cuda::std::min(k, n);

  auto out = cuda::make_buffer<cuda::std::size_t>(
    stream, resource, expected, cuda::std::numeric_limits<cuda::std::size_t>::max());
  auto written = cuda::make_buffer<cuda::std::size_t>(stream, resource, cuda::std::size_t{1}, cuda::std::size_t{0});

  cuda::launch(
    stream, cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<1>()), sample_kernel<Rng>{n, k, rng}, out, written);

  std::vector<cuda::std::size_t> host_out(expected);
  std::vector<cuda::std::size_t> host_written(1);
  cuda::copy_bytes(stream, out, host_out);
  cuda::copy_bytes(stream, written, host_written);
  stream.sync();

  REQUIRE(host_written[0] == expected);
  return host_out;
}

// Every sample must be strictly increasing, and every index must lie in [0, n). Strict increase is
// the combined statement of stability and sampling without replacement.
void check_sample(const std::vector<cuda::std::size_t>& sample, cuda::std::size_t n)
{
  for (cuda::std::size_t i = 0; i < sample.size(); ++i)
  {
    REQUIRE(sample[i] < n);
    if (i > 0)
    {
      REQUIRE(sample[i] > sample[i - 1]);
    }
  }
}

// Draws one sample per row of `out`, which holds `iterations` rows of `k` indices. Work spreads
// across the grid: thread `t` handles rows t, t + grid_size, t + 2 * grid_size, ... Each row offsets
// the generator by its own row index, so the streams do not overlap and the result does not depend
// on the launch shape. A `sample` call consumes O(k) variates, and the stride below far exceeds
// every `k` tested here.
template <class Rng>
struct batched_sample_kernel
{
  cuda::std::size_t n_;
  Rng rng_;

  TEST_DEVICE_FUNC void operator()(cuda::std::mdspan<cuda::std::size_t, cuda::std::dims<2>> out) const
  {
    const auto rank = static_cast<cuda::std::size_t>(cuda::gpu_thread.rank(cuda::grid));
    const auto size = static_cast<cuda::std::size_t>(cuda::gpu_thread.count(cuda::grid));

    cuda::counting_iterator<cuda::std::size_t> first{0};
    cuda::counting_iterator<cuda::std::size_t> last{n_};

    for (cuda::std::size_t i = rank; i < out.extent(0); i += size)
    {
      constexpr auto max_draws = 4096;
      auto rng                 = rng_;

      // We assume here that the following sample call will poll the rng less than max_draws
      // per call. If it doesn't, then subsequent iterations of the loop will get overlapping
      // streamz of random numbers as a previous iteration which pollutes the result.
      rng.discard(i * max_draws);
      cuda::sample(first, last, &out(i, 0), out.extent(1), rng);
    }
  }
};

// Returns `iterations` samples of `k` out of `n`, one per row. Requires `0 < k <= n`.
template <class Rng = cuda::std::philox4x64>
std::vector<cuda::std::size_t>
device_sample_batch(cuda::std::size_t n, cuda::std::size_t k, cuda::std::size_t iterations, Rng rng = Rng{})
{
  auto stream   = cuda::stream{cuda::device_ref{0}};
  auto resource = cuda::device_default_memory_pool(cuda::device_ref{0});

  auto out = cuda::make_buffer<cuda::std::size_t>(stream, resource, iterations * k, static_cast<cuda::std::size_t>(-1));

  cuda::launch(stream,
               cuda::make_config(cuda::grid_dims(256), cuda::block_dims(128)),
               batched_sample_kernel<Rng>{n, rng},
               cuda::std::mdspan<cuda::std::size_t, cuda::std::dims<2>>{out.data(), iterations, k});

  std::vector<cuda::std::size_t> host_out(iterations * k);
  cuda::copy_bytes(stream, out, host_out);
  stream.sync();

  return host_out;
}

// Maximum score that the callers below accept. A correct sampler gets a non-zero score, because
// random draws do not land exactly on the ideal counts. This ceiling separates that noise from actual
// bias.
//
// `df` is the number of counts minus one. The counts add up to a known total, so the last count
// follows from the others.
//
// A correct sampler goes above this ceiling with probability 1e-9. The usual level of 0.05 fails
// about one run in twenty. A biased sampler still fails, because its score increases with the
// number of draws.
//
// The formula is the Wilson-Hilferty approximation. It gives a high result in the tail, so the
// ceiling is safe.
double chi_squared_critical_value(double df)
{
  constexpr double z = 6.0;

  const double a = 2.0 / (9.0 * df);
  const double t = 1.0 - a + (z * std::sqrt(a));

  return df * t * t * t;
}
} // namespace

C2H_TEST("cuda::sample yields a strictly increasing sample", "[algorithm][sample]")
{
  // Exhaustive over every k for a range of small populations. This covers k == 0, k == 1,
  // k == n - 1, k == n, and every sampling fraction between, including the Method D to Method A
  // crossover.
  const cuda::std::size_t n = GENERATE(0, 1, 2, 3, 7, 13, 14, 27, 31, 32, 33, 63, 64, 65);

  for (cuda::std::size_t k = 0; k <= n; ++k)
  {
    const auto sample = device_sample(n, k);

    REQUIRE(sample.size() == k);
    check_sample(sample, n);
  }
}

C2H_TEST("cuda::sample returns the whole population when k >= n", "[algorithm][sample]")
{
  // A sample size at or above the population size has exactly one valid answer.
  const cuda::std::size_t n    = GENERATE(0, 1, 2, 5, 13, 32, 47);
  const cuda::std::size_t over = GENERATE(0, 1, 2, 17, 100000);

  const auto sample = device_sample(n, n + over);

  REQUIRE(sample.size() == n);

  for (cuda::std::size_t i = 0; i < n; ++i)
  {
    REQUIRE(sample[i] == i);
  }
}

C2H_TEST("cuda::sample returns the whole population for a huge unsigned k", "[algorithm][sample]")
{
  // A large unsigned k (e.g. size_t::max) must not wrap to -1 after a narrowing cast to int64_t.
  // Before the fix, static_cast<int64_t>(size_t::max) == -1 caused __k <= 0 and an early return
  // with no output written.
  constexpr cuda::std::size_t n = 5;
  constexpr cuda::std::size_t k = cuda::std::numeric_limits<cuda::std::size_t>::max();

  const auto sample = device_sample(n, k);

  REQUIRE(sample.size() == n);

  for (cuda::std::size_t i = 0; i < n; ++i)
  {
    REQUIRE(sample[i] == i);
  }
}

C2H_TEST("cuda::sample writes nothing for a degenerate request", "[algorithm][sample]")
{
  // A sample size of zero, or an empty population, must write nothing. A negative sample size is
  // outside the contract: `sample` asserts on it, so it is not tested here.
  const cuda::std::size_t n = GENERATE(0, 1, 16, 1024);
  const cuda::std::size_t k = GENERATE(0);

  const auto sample = device_sample(n, k);
  REQUIRE(sample.empty());
}

C2H_TEST("cuda::sample draws every subset with equal probability", "[algorithm][sample]")
{
  // A choice of `k` out of `n` has a fixed set of answers. For n = 4 and k = 2 the answers are
  // {0,1} {0,2} {0,3} {1,2} {1,3} {2,3}. A correct sampler returns each answer equally often.
  //
  // A count of each index does not prove this. A sampler that returns only {0,1} and {2,3} gives
  // each index the correct rate, but it never returns {0,2}. Only a count of whole answers finds
  // this fault.
  //
  // These populations are small, so all answers fit in memory.
  const cuda::std::size_t n = GENERATE(range(2, 11));

  for (cuda::std::size_t k = 1; k < n; ++k)
  {
    // Number of answers: `n` choose `k`.
    cuda::std::size_t n_choose_k = 1;
    for (cuda::std::size_t i = 0; i < k; ++i)
    {
      n_choose_k = n_choose_k * (n - i) / (i + 1);
    }

    // Draw enough samples for 50 hits per answer. Fewer than 5 hits makes the score below
    // unreliable.
    const auto iterations = 50 * n_choose_k;

    auto storage = device_sample_batch(n, k, iterations);
    const cuda::std::mdspan<cuda::std::size_t, cuda::std::dims<2>> batch{storage.data(), iterations, k};

    // One counter per answer. The key holds the answer as bits: bit `i` is set when index `i` is in
    // the sample, so {0, 2, 3} becomes binary 1101.
    std::unordered_map<cuda::std::size_t, cuda::std::size_t> counts;

    for (cuda::std::size_t i = 0; i < batch.extent(0); ++i)
    {
      cuda::std::size_t mask = 0;

      for (cuda::std::size_t j = 0; j < batch.extent(1); ++j)
      {
        REQUIRE(batch(i, j) < n);
        if (j > 0)
        {
          REQUIRE(batch(i, j) > batch(i, j - 1));
        }
        mask |= cuda::std::size_t{1} << batch(i, j);
      }

      ++counts[mask];
    }

    // The map holds only the answers that occurred. An equal size shows that all answers occurred,
    // and that the loop below finds all counters.
    REQUIRE(counts.size() == n_choose_k);

    // A correct sampler keeps each counter near this value.
    const double expected = static_cast<double>(iterations) / static_cast<double>(n_choose_k);

    // Score the distance from that value. Each counter adds its error squared, divided by the
    // expected value. The division scales the error: an error of 10 is large at 50, but small at
    // 5000. The square prevents a high count and a low count from cancelling.
    double stat = 0.0;
    for (const auto& entry : counts)
    {
      const double delta = static_cast<double>(entry.second) - expected;
      stat += (delta * delta) / expected;
    }

    INFO("n = " << n << ", k = " << k << ", n_choose_k = " << n_choose_k << ", chi2 = " << stat);
    REQUIRE(stat < chi_squared_critical_value(static_cast<double>(n_choose_k) - 1.0));
  }
}

C2H_TEST("cuda::sample covers a large population uniformly", "[algorithm][sample]")
{
  // The test above lists all answers, so it applies only to small populations. A choice of 32 out
  // of 64 has 1.8e18 answers, which is too many to count.
  //
  // This test counts each index instead. This check is weaker, but it finds an index that the
  // sampler never selects, and it finds a sampler that prefers one end of the population.
  //
  // The pairs start at a low sampling fraction, which uses Method D, and end at a high one, which
  // uses the Method A fallback.
  struct shape
  {
    cuda::std::size_t n_;
    cuda::std::size_t k_;
  };

  const auto s = GENERATE(shape{64, 1}, shape{64, 8}, shape{64, 32}, shape{24, 20}, shape{16, 15}, shape{13, 12});

  // Give each index the same number of hits in all shapes. Each draw selects `k` of the `n`
  // indices, so a small `k` needs more draws. A fixed draw count gives those shapes too few hits.
  constexpr double target_hits = 2000.0;

  const auto iterations =
    static_cast<cuda::std::size_t>(target_hits * static_cast<double>(s.n_) / static_cast<double>(s.k_));

  auto storage = device_sample_batch(s.n_, s.k_, iterations);
  const cuda::std::mdspan<cuda::std::size_t, cuda::std::dims<2>> batch{storage.data(), iterations, s.k_};

  std::vector<cuda::std::size_t> hits(s.n_, 0);

  for (cuda::std::size_t i = 0; i < batch.extent(0); ++i)
  {
    for (cuda::std::size_t j = 0; j < batch.extent(1); ++j)
    {
      REQUIRE(batch(i, j) < s.n_);
      if (j > 0)
      {
        REQUIRE(batch(i, j) > batch(i, j - 1));
      }
      ++hits[batch(i, j)];
    }
  }

  const double expected = static_cast<double>(iterations) * static_cast<double>(s.k_) / static_cast<double>(s.n_);

  double stat = 0.0;
  for (auto count : hits)
  {
    const double delta = static_cast<double>(count) - expected;
    stat += (delta * delta) / expected;
  }

  // Each draw picks `k` different indices, so the same index cannot appear twice in one draw.
  // This reduces how much the counts vary compared to what the formula above assumes. The
  // true variance is smaller by the factor `(n - k) / (n - 1)`. Divide `stat` by that factor
  // to get a value that follows the chi-squared distribution. Without this step the test
  // accepts a biased sampler.
  stat *= (static_cast<double>(s.n_) - 1.0) / (static_cast<double>(s.n_) - static_cast<double>(s.k_));

  INFO("n = " << s.n_ << ", k = " << s.k_ << ", chi2 = " << stat);
  REQUIRE(stat < chi_squared_critical_value(static_cast<double>(s.n_) - 1.0));
}

C2H_TEST("cuda::sample crosses from Method D to Method A", "[algorithm][sample]")
{
  // The crossover happens when the remaining population falls below 13 times the remaining sample
  // size. These ratios start above that threshold and fall below it partway through, so a single
  // call uses both methods.
  const cuda::std::size_t n       = GENERATE(200, 500, 1300, 5000);
  const cuda::std::size_t divisor = GENERATE(13, 12, 10, 4, 2);
  const cuda::std::size_t k       = n / divisor;

  const auto sample = device_sample(n, k);

  REQUIRE(sample.size() == k);
  check_sample(sample, n);
}

C2H_TEST("cuda::sample is O(k) for a large population", "[algorithm][sample]")
{
  // These populations are far too large to visit element by element. Completing the call at all is
  // the assertion. 2^53 is the documented upper bound, because the gap distribution is computed in
  // double and the population size must stay exactly representable.
  const cuda::std::size_t n = GENERATE(
    cuda::std::size_t{1} << 20, cuda::std::size_t{1} << 32, cuda::std::size_t{1} << 40, cuda::std::size_t{1} << 53);
  const cuda::std::size_t k = GENERATE(1, 2, 16, 1024);

  const auto sample = device_sample(n, k);
  REQUIRE(sample.size() == k);
  check_sample(sample, n);
}

C2H_TEST("cuda::sample spreads a sample across a large population", "[algorithm][sample]")
{
  // A correct gap distribution puts the selected indices across the whole range. An implementation
  // that drew gaps too small would cluster every index near the start.
  constexpr cuda::std::size_t n = cuda::std::size_t{1} << 32;
  constexpr cuda::std::size_t k = 4096;

  const auto sample = device_sample(n, k);

  REQUIRE(sample.size() == k);
  check_sample(sample, n);

  int in_last_quarter = 0;
  for (auto index : sample)
  {
    if (index >= (n / 4) * 3)
    {
      ++in_last_quarter;
    }
  }

  // A quarter of 4096 draws is 1024. A count near zero would mean the gaps are too small.
  REQUIRE(in_last_quarter > 850);
  REQUIRE(in_last_quarter < 1200);
}

C2H_TEST("cuda::sample is a pure function of the generator state", "[algorithm][sample]")
{
  // Two identically seeded runs must agree exactly, and two different states are likely to
  // disagree. In this case, since the seeds are constant, there is no likelihood but if we had
  // a random seed here as well it could spuriously fail.
  const cuda::std::size_t n = GENERATE(256, 4096);
  const cuda::std::size_t k = GENERATE(1, 32);

  const auto first  = device_sample(n, k, cuda::std::philox4x64{42});
  const auto second = device_sample(n, k, cuda::std::philox4x64{42});

  REQUIRE(first == second);

  const auto other = device_sample(n, k, cuda::std::philox4x64{1337});

  REQUIRE(first != other);
}

C2H_TEST("cuda::sample works with several generator types", "[algorithm][sample]")
{
  constexpr cuda::std::size_t n = 1024;
  constexpr cuda::std::size_t k = 64;

  SECTION("minstd_rand")
  {
    check_sample(device_sample(n, k, cuda::std::minstd_rand{7}), n);
  }

  SECTION("minstd_rand0")
  {
    check_sample(device_sample(n, k, cuda::std::minstd_rand0{7}), n);
  }

  SECTION("philox4x32")
  {
    check_sample(device_sample(n, k, cuda::std::philox4x32{7}), n);
  }

  SECTION("philox4x64")
  {
    check_sample(device_sample(n, k, cuda::std::philox4x64{7}), n);
  }
}
