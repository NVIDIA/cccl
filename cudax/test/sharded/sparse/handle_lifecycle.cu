//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief cuSPARSE handle lifecycle across the sparse products: the handle is
 *        PER PLACE and group-owned (one `cusparseHandle_t` per place of a
 *        `place_group`, shared by every `sharded_csr` built over the group),
 *        while descriptors/plans/workspaces stay container-owned. Checks:
 *        two matrices on ONE group share the same per-place handles; two
 *        groups do not share; destroying a container does NOT destroy the
 *        handle (a second container keeps producing correct results after the
 *        first dies, through the SAME handles); results stay correct
 *        throughout.
 */

#include <cuda/experimental/sharded.cuh>
#include <cuda/experimental/sharded_sparse.cuh>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

using namespace cuda::experimental::sharded;
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

host_csr make_csr(::std::int64_t rows, ::std::int64_t cols, unsigned seed)
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
    const int len = 4 + static_cast<int>(r % 5);
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

::std::vector<double> host_spmv(const host_csr& m, const ::std::vector<double>& x)
{
  ::std::vector<double> y(static_cast<size_t>(m.rows));
  for (::std::int64_t r = 0; r < m.rows; r++)
  {
    double acc = 0.0;
    for (int k = m.offsets[r]; k < m.offsets[r + 1]; k++)
    {
      acc += m.values[static_cast<size_t>(k)] * x[static_cast<size_t>(m.colinds[static_cast<size_t>(k)])];
    }
    y[static_cast<size_t>(r)] = acc;
  }
  return y;
}

void expect_close(const ::std::vector<double>& got, const ::std::vector<double>& ref, double tol = 1e-13)
{
  EXPECT(got.size() == ref.size());
  for (size_t i = 0; i < got.size(); i++)
  {
    const double err = ::std::abs(got[i] - ref[i]) / ::std::max(1.0, ::std::abs(ref[i]));
    EXPECT(err <= tol);
  }
}

double* device_upload(const ::std::vector<double>& host)
{
  double* ptr = nullptr;
  cuda_safe_call(cudaMalloc(&ptr, host.size() * sizeof(double)));
  cuda_safe_call(cudaMemcpy(ptr, host.data(), host.size() * sizeof(double), cudaMemcpyDefault));
  return ptr;
}

// The place's handle as the sparse products see it: fetched through the
// group's library-state cache with the place's exec context active, exactly
// like the spmv/spmm call sites.
cusparseHandle_t place_handle(place_group& group, size_t idx)
{
  cuda::experimental::places::exec_place_scope scope(group.place(idx));
  return reserved::get_place_cusparse_handle(group, idx);
}

// Run spmv on a fresh matrix over @p group and validate against the host
// reference; returns nothing but ensures the engine path really executed.
void run_spmv_checked(place_group& group, const host_csr& m, const ::std::vector<double>& x_h, double* d_x)
{
  sharded_csr<double> A(group, m.rows, m.cols, m.offsets.data(), m.colinds.data(), m.values.data());
  auto y = A.make_row_partitioned();
  spmv(group, A, d_x, y);
  y.sync();
  ::std::vector<double> got(static_cast<size_t>(m.rows));
  y.copy_to_host(got.data());
  expect_close(got, host_spmv(m, x_h));
}

void test_shared_within_group_distinct_across_groups(const host_csr& m)
{
  auto group = place_group::by_locality_domains();

  const auto x_h = random_vector(static_cast<size_t>(m.cols), 11);
  double* d_x    = device_upload(x_h);

  // Handles are lazy: nothing in the cache before the first product.
  for (size_t i = 0; i < group.size(); i++)
  {
    EXPECT(!group.has_lib_state<reserved::cusparse_place_handle>(i));
  }

  // Two matrices over ONE group: after both ran, each place has exactly ONE
  // handle and both matrices went through it.
  sharded_csr<double> A1(group, m.rows, m.cols, m.offsets.data(), m.colinds.data(), m.values.data());
  sharded_csr<double> A2(group, m.rows, m.cols, m.offsets.data(), m.colinds.data(), m.values.data());
  auto y1 = A1.make_row_partitioned();
  auto y2 = A2.make_row_partitioned();

  spmv(group, A1, d_x, y1);
  y1.sync();

  ::std::vector<cusparseHandle_t> after_first(group.size());
  for (size_t i = 0; i < group.size(); i++)
  {
    EXPECT(group.has_lib_state<reserved::cusparse_place_handle>(i));
    after_first[i] = place_handle(group, i);
    EXPECT(after_first[i] != nullptr);
  }

  spmv(group, A2, d_x, y2);
  y2.sync();

  // The second matrix created NO new handles: same per-place handle objects.
  for (size_t i = 0; i < group.size(); i++)
  {
    EXPECT(place_handle(group, i) == after_first[i]);
  }

  // Distinct places hold distinct handles.
  for (size_t i = 1; i < group.size(); i++)
  {
    EXPECT(after_first[i] != after_first[0]);
  }

  // Both matrices produce correct results through the shared handles.
  const auto ref = host_spmv(m, x_h);
  ::std::vector<double> got(static_cast<size_t>(m.rows));
  y1.copy_to_host(got.data());
  expect_close(got, ref);
  y2.copy_to_host(got.data());
  expect_close(got, ref);

  // A SECOND group over the same places is a distinct resource scope: its
  // per-place handles are different objects.
  {
    place_group other = place_group::by_locality_domains();
    run_spmv_checked(other, m, x_h, d_x);
    for (size_t i = 0; i < other.size(); i++)
    {
      EXPECT(place_handle(other, i) != after_first[i]);
    }
  }

  cuda_safe_call(cudaFree(d_x));
}

void test_container_death_leaves_handle_alive(const host_csr& m)
{
  auto group = place_group::by_locality_domains();

  const auto x_h = random_vector(static_cast<size_t>(m.cols), 21);
  double* d_x    = device_upload(x_h);
  const auto ref = host_spmv(m, x_h);

  // First container builds the handles...
  ::std::vector<cusparseHandle_t> handles(group.size());
  {
    sharded_csr<double> A1(group, m.rows, m.cols, m.offsets.data(), m.colinds.data(), m.values.data());
    auto y1 = A1.make_row_partitioned();
    spmv(group, A1, d_x, y1);
    y1.sync();
    for (size_t i = 0; i < group.size(); i++)
    {
      handles[i] = place_handle(group, i);
    }
  } // ...and dies here, taking its descriptors/plans/workspaces with it.

  // The handles survive the container: still cached, same objects...
  for (size_t i = 0; i < group.size(); i++)
  {
    EXPECT(group.has_lib_state<reserved::cusparse_place_handle>(i));
    EXPECT(place_handle(group, i) == handles[i]);
  }

  // ...and a second container built AFTER the first died computes correct
  // results through those same handles (which also exercises them: a
  // destroyed handle would fail these cuSPARSE calls).
  sharded_csr<double> A2(group, m.rows, m.cols, m.offsets.data(), m.colinds.data(), m.values.data());
  auto y2 = A2.make_row_partitioned();
  spmv(group, A2, d_x, y2);
  y2.sync();
  ::std::vector<double> got(static_cast<size_t>(m.rows));
  y2.copy_to_host(got.data());
  expect_close(got, ref);
  for (size_t i = 0; i < group.size(); i++)
  {
    EXPECT(place_handle(group, i) == handles[i]);
  }

  cuda_safe_call(cudaFree(d_x));
}

void test_spmm_and_times_share_the_same_handle(const host_csr& m)
{
  // Every entry point of the sparse layer draws from the same per-place
  // cache: spmv, spmm, and the measured-rebalance timers.
  auto group = place_group::by_locality_domains();

  const ::std::int64_t n_cols = 8;
  const auto x_h              = random_vector(static_cast<size_t>(m.cols), 31);
  const auto B_h              = random_vector(static_cast<size_t>(m.cols * n_cols), 32);
  double* d_x                 = device_upload(x_h);
  double* d_B                 = device_upload(B_h);

  sharded_csr<double> A(group, m.rows, m.cols, m.offsets.data(), m.colinds.data(), m.values.data());
  auto y = A.make_row_partitioned();
  auto C = A.make_row_partitioned(n_cols);

  spmv(group, A, d_x, y);
  y.sync();
  ::std::vector<cusparseHandle_t> handles(group.size());
  for (size_t i = 0; i < group.size(); i++)
  {
    handles[i] = place_handle(group, i);
  }

  spmm(group, A, d_B, C, n_cols);
  C.sync();
  auto times = spmv_shard_times(group, A, d_x, y, 1.0, 0.0, /*warmup=*/1, /*iters=*/2);
  EXPECT(times.size() == group.size());

  for (size_t i = 0; i < group.size(); i++)
  {
    EXPECT(place_handle(group, i) == handles[i]);
  }

  cuda_safe_call(cudaFree(d_x));
  cuda_safe_call(cudaFree(d_B));
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  const auto m = make_csr(20011, 1024, 3);

  test_shared_within_group_distinct_across_groups(m);
  test_container_death_leaves_handle_alive(m);
  test_spmm_and_times_share_the_same_handle(m);

  return 0;
}
