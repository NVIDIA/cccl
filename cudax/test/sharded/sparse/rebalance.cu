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
 * @brief The measured-rebalance path: `spmm_shard_times` /
 *        `spmv_shard_times` measure each shard solo through the normal call
 *        path, and `sharded_csr::time_balanced_boundaries` converts the
 *        measurement into a time-equalizing split. On a deliberately
 *        time-skewed matrix the test asserts the DIRECTION the boundaries
 *        move (the slower shard shrinks), that the model's predicted times at
 *        the new boundaries are equalized, and that the re-measured imbalance
 *        decreases — not exact values, which are hardware-dependent.
 */

#include <cuda/experimental/__sharded/sparse.cuh> // opt-in vendor tier
#include <cuda/experimental/sharded.cuh>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
struct host_csr
{
  ::std::int64_t rows = 0, cols = 0;
  ::std::vector<int> offsets;
  ::std::vector<int> colinds;
  ::std::vector<double> values;
};

// Deliberately time-skewed structure: the first half of the rows is a
// short-row region (heavy per-row overhead per nonzero), the second half a
// long-row region. An nnz-balanced split is strongly TIME-imbalanced here.
host_csr make_two_region_csr(::std::int64_t rows, ::std::int64_t cols, unsigned seed)
{
  host_csr m;
  m.rows = rows;
  m.cols = cols;
  m.offsets.push_back(0);
  ::std::mt19937 rng(seed);
  ::std::uniform_int_distribution<int> col_dist(0, static_cast<int>(cols) - 1);
  ::std::uniform_real_distribution<double> val_dist(-1.0, 1.0);
  for (::std::int64_t r = 0; r < rows; r++)
  {
    const int len = (r < rows / 2) ? 4 : 96;
    for (int k = 0; k < len; k++)
    {
      m.colinds.push_back(col_dist(rng));
      m.values.push_back(val_dist(rng));
    }
    m.offsets.push_back(m.offsets.back() + len);
  }
  return m;
}

::std::vector<double> random_vector(size_t n, unsigned seed)
{
  ::std::vector<double> v(n);
  ::std::mt19937 rng(seed);
  ::std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (auto& x : v)
  {
    x = dist(rng);
  }
  return v;
}

double imbalance(const ::std::vector<double>& ms)
{
  const double lo = *::std::min_element(ms.begin(), ms.end());
  const double hi = *::std::max_element(ms.begin(), ms.end());
  EXPECT(lo > 0.0);
  return hi / lo;
}

// Predicted per-new-shard time under the piecewise-rate model built from the
// measured split (the exact model time_balanced_boundaries balances).
::std::vector<double> predicted_times(
  const host_csr& m,
  const ::std::vector<::std::int64_t>& measured_bounds,
  const ::std::vector<double>& measured_ms,
  const ::std::vector<::std::int64_t>& new_bounds)
{
  const size_t P = measured_ms.size();
  ::std::vector<::std::int64_t> mb, nb;
  mb.push_back(0);
  mb.insert(mb.end(), measured_bounds.begin(), measured_bounds.end());
  mb.push_back(m.rows);
  nb.push_back(0);
  nb.insert(nb.end(), new_bounds.begin(), new_bounds.end());
  nb.push_back(m.rows);

  ::std::vector<double> cost_per_nnz(P);
  for (size_t d = 0; d < P; d++)
  {
    const ::std::int64_t nnz_d = m.offsets[mb[d + 1]] - m.offsets[mb[d]];
    cost_per_nnz[d]            = (nnz_d > 0) ? measured_ms[d] / static_cast<double>(nnz_d) : 0.0;
  }

  ::std::vector<double> pred(P, 0.0);
  size_t old_d = 0;
  for (size_t d = 0; d < P; d++)
  {
    for (::std::int64_t r = nb[d]; r < nb[d + 1]; r++)
    {
      while (old_d + 1 < P && r >= mb[old_d + 1])
      {
        old_d++;
      }
      pred[d] += cost_per_nnz[old_d] * static_cast<double>(m.offsets[r + 1] - m.offsets[r]);
    }
  }
  return pred;
}

void test_rebalance_moves_toward_equal_time(place_group& group, const host_csr& m)
{
  const ::std::int64_t n_cols = 16;
  const auto B_h              = random_vector(static_cast<size_t>(m.cols * n_cols), 11);
  double* d_B                 = nullptr;
  cuda_safe_call(cudaMalloc(&d_B, B_h.size() * sizeof(double)));
  cuda_safe_call(cudaMemcpy(d_B, B_h.data(), B_h.size() * sizeof(double), cudaMemcpyDefault));

  // Round 0: the default nnz-balanced split, measured.
  sharded_csr<double> A(group, m.rows, m.cols, m.offsets.data(), m.colinds.data(), m.values.data());
  cusparse_handles handles(group);
  auto C        = A.make_row_partitioned(n_cols);
  const auto b0 = A.interior_boundaries();
  spmm_plan<double> plan(handles, A, n_cols);
  const auto times = spmm_shard_times(plan, d_B, C, 1.0, 0.0, /* warmup */ 2, /* iters */ 6);
  EXPECT(times.size() == group.size());
  const double skew0 = imbalance(times);

  // Keep the round-0 output as the correctness anchor for the rebuilt split.
  ::std::vector<double> C_ref(static_cast<size_t>(m.rows * n_cols));
  spmm(plan, d_B, C);
  C.sync();
  C.copy_to_host(C_ref.data());

  // The rebalanced boundaries from the measurement.
  const auto b1 = sharded_csr<double>::time_balanced_boundaries(m.rows, m.offsets.data(), b0, times);
  EXPECT(b1.size() == group.size() - 1);

  // Model check (deterministic given the measurement): the predicted times at
  // the new boundaries are equalized up to row granularity.
  {
    const auto pred = predicted_times(m, b0, times, b1);
    const double hi = *::std::max_element(pred.begin(), pred.end());
    const double lo = *::std::min_element(pred.begin(), pred.end());
    EXPECT(lo > 0.0);
    EXPECT(hi / lo < 1.10);
  }

  // Direction (2-place layout): when one shard measured decisively slower,
  // the shared boundary moves so that the slower shard SHRINKS.
  if (group.size() == 2 && skew0 > 1.1)
  {
    if (times[0] > times[1])
    {
      EXPECT(b1[0] < b0[0]);
    }
    else
    {
      EXPECT(b1[0] > b0[0]);
    }
  }

  // Round 1: rebuild on the rebalanced boundaries; the measured imbalance
  // must decrease when round 0 was decisively skewed (monotone toward equal
  // time, not exact equality — rates shift as rows change shards).
  sharded_csr<double> A1(group, m.rows, m.cols, m.offsets.data(), m.colinds.data(), m.values.data(), b1);
  cusparse_handles handles1(group);
  spmm_plan<double> plan1(handles1, A1, n_cols);
  auto C1            = A1.make_row_partitioned(n_cols);
  const auto times1  = spmm_shard_times(plan1, d_B, C1, 1.0, 0.0, /* warmup */ 2, /* iters */ 6);
  const double skew1 = imbalance(times1);
  if (skew0 > 1.25)
  {
    EXPECT(skew1 < skew0);
  }

  // The rebalanced split computes the same product (row split is
  // value-preserving; compare to the round-0 result within FP64 noise).
  {
    spmm(plan1, d_B, C1);
    C1.sync();
    ::std::vector<double> got(C_ref.size());
    C1.copy_to_host(got.data());
    for (size_t i = 0; i < got.size(); i++)
    {
      const double err = ::std::abs(got[i] - C_ref[i]) / ::std::max(1.0, ::std::abs(C_ref[i]));
      EXPECT(err <= 1e-13);
    }
  }

  cuda_safe_call(cudaFree(d_B));
}

void test_spmv_shard_times_sane(place_group& group, const host_csr& m)
{
  sharded_csr<double> A(group, m.rows, m.cols, m.offsets.data(), m.colinds.data(), m.values.data());
  cusparse_handles handles(group);
  const auto x_h = random_vector(static_cast<size_t>(m.cols), 21);
  double* d_x    = nullptr;
  cuda_safe_call(cudaMalloc(&d_x, x_h.size() * sizeof(double)));
  cuda_safe_call(cudaMemcpy(d_x, x_h.data(), x_h.size() * sizeof(double), cudaMemcpyDefault));

  auto y = A.make_row_partitioned();
  spmv_plan<double> plan(handles, A);
  const auto times = spmv_shard_times(plan, d_x, y, 1.0, 0.0, /* warmup */ 1, /* iters */ 4);
  EXPECT(times.size() == group.size());
  for (double t : times)
  {
    EXPECT(t > 0.0);
  }

  cuda_safe_call(cudaFree(d_x));
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group{make_locality_domain_grid()};

  const auto m = make_two_region_csr(400000, 2048, 1);
  if (group.size() >= 2)
  {
    test_rebalance_moves_toward_equal_time(group, m);
  }
  test_spmv_shard_times_sane(group, m);

  return 0;
}
