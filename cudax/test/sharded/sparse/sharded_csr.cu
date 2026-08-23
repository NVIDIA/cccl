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
 * @brief `sharded_csr` container tests (no vendor library involved):
 *        construction against a host CSR reference (slice + rebase), default
 *        nnz-balanced and caller-supplied split boundaries, boundary
 *        validation, row-partitioned outputs (separate and contiguous),
 *        `lib_state` caching, and the `time_balanced_boundaries` host model.
 */

#include <cuda/experimental/sharded.cuh>

#include <cstdint>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
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

  ::std::int64_t nnz() const
  {
    return offsets.back();
  }
};

// Mixed row-length CSR: short rows, EMPTY rows, and a few long rows, with a
// deterministic pattern.
host_csr make_mixed_csr(::std::int64_t rows, ::std::int64_t cols, unsigned seed)
{
  host_csr m;
  m.rows = rows;
  m.cols = cols;
  m.offsets.reserve(static_cast<size_t>(rows) + 1);
  m.offsets.push_back(0);
  ::std::mt19937 rng(seed);
  ::std::uniform_int_distribution<int> col_dist(0, static_cast<int>(cols) - 1);
  ::std::uniform_real_distribution<double> val_dist(-1.0, 1.0);
  for (::std::int64_t r = 0; r < rows; r++)
  {
    int len = 0;
    switch (r % 7)
    {
      case 0:
        len = 0; // empty row
        break;
      case 1:
      case 2:
        len = 1 + static_cast<int>(r % 3);
        break;
      case 3:
        len = 24; // long row
        break;
      default:
        len = 4 + static_cast<int>(r % 5);
        break;
    }
    for (int k = 0; k < len; k++)
    {
      m.colinds.push_back(col_dist(rng));
      m.values.push_back(val_dist(rng));
    }
    m.offsets.push_back(m.offsets.back() + len);
  }
  return m;
}

// Copy one shard's arrays back to host and check the slice-and-rebase
// contract against the parent host CSR.
void check_shards_match_reference(const host_csr& m, const sharded_csr<double>& A)
{
  ::std::int64_t row_cursor = 0;
  ::std::int64_t nnz_total  = 0;
  for (size_t d = 0; d < A.num_shards(); d++)
  {
    const auto& sh = A.shard(d);
    EXPECT(sh.row_begin == row_cursor);
    EXPECT(sh.rows >= 1); // default and clamped splits keep shards non-empty
    EXPECT(sh.nnz_begin == m.offsets[sh.row_begin]);
    EXPECT(sh.nnz == m.offsets[sh.row_begin + sh.rows] - m.offsets[sh.row_begin]);
    row_cursor += sh.rows;
    nnz_total += sh.nnz;

    // Offsets: rows+1 entries, rebased to 0
    ::std::vector<int> off(static_cast<size_t>(sh.rows) + 1);
    cuda_safe_call(cudaMemcpy(off.data(), sh.offsets, off.size() * sizeof(int), cudaMemcpyDefault));
    EXPECT(off[0] == 0);
    for (::std::int64_t r = 0; r <= sh.rows; r++)
    {
      EXPECT(off[static_cast<size_t>(r)] == m.offsets[sh.row_begin + r] - m.offsets[sh.row_begin]);
    }

    // colinds/values: the parent nnz slice
    if (sh.nnz > 0)
    {
      ::std::vector<int> cols(static_cast<size_t>(sh.nnz));
      ::std::vector<double> vals(static_cast<size_t>(sh.nnz));
      cuda_safe_call(cudaMemcpy(cols.data(), sh.colinds, cols.size() * sizeof(int), cudaMemcpyDefault));
      cuda_safe_call(cudaMemcpy(vals.data(), sh.values, vals.size() * sizeof(double), cudaMemcpyDefault));
      for (::std::int64_t k = 0; k < sh.nnz; k++)
      {
        EXPECT(cols[static_cast<size_t>(k)] == m.colinds[static_cast<size_t>(sh.nnz_begin + k)]);
        EXPECT(vals[static_cast<size_t>(k)] == m.values[static_cast<size_t>(sh.nnz_begin + k)]);
      }
    }
  }
  EXPECT(row_cursor == m.rows);
  EXPECT(nnz_total == m.nnz());
}

void test_construction_default_split(place_group& group)
{
  const auto m = make_mixed_csr(10007, 512, 42);
  sharded_csr<double> A(group, m.rows, m.cols, m.offsets.data(), m.colinds.data(), m.values.data());

  EXPECT(A.num_rows() == m.rows);
  EXPECT(A.num_cols() == m.cols);
  EXPECT(A.nnz() == m.nnz());
  EXPECT(A.num_shards() == group.size());
  check_shards_match_reference(m, A);

  // Default split is nnz-balanced: every shard's nnz within one max-row-nnz
  // of the ideal share.
  int max_row_nnz = 0;
  for (::std::int64_t r = 0; r < m.rows; r++)
  {
    max_row_nnz = ::std::max(max_row_nnz, m.offsets[r + 1] - m.offsets[r]);
  }
  const double ideal = static_cast<double>(m.nnz()) / static_cast<double>(A.num_shards());
  for (size_t d = 0; d < A.num_shards(); d++)
  {
    EXPECT(::std::abs(static_cast<double>(A.shard(d).nnz) - ideal) <= static_cast<double>(max_row_nnz) + 1.0);
  }

  // interior_boundaries round-trips through the constructor.
  const auto bounds = A.interior_boundaries();
  EXPECT(bounds.size() == A.num_shards() - 1);
  sharded_csr<double> A2(group, m.rows, m.cols, m.offsets.data(), m.colinds.data(), m.values.data(), bounds);
  for (size_t d = 0; d < A.num_shards(); d++)
  {
    EXPECT(A2.shard(d).row_begin == A.shard(d).row_begin);
    EXPECT(A2.shard(d).rows == A.shard(d).rows);
  }
}

void test_explicit_boundaries(place_group& group)
{
  const auto m = make_mixed_csr(5000, 128, 7);

  // A deliberately skewed split: shard 0 gets 10% of the rows.
  ::std::vector<::std::int64_t> bounds;
  for (size_t d = 1; d < group.size(); d++)
  {
    bounds.push_back(static_cast<::std::int64_t>(d) * m.rows / (10 * static_cast<::std::int64_t>(group.size())));
  }
  // Boundaries must stay ascending and >= 1
  for (size_t d = 0; d < bounds.size(); d++)
  {
    bounds[d] = ::std::max<::std::int64_t>(bounds[d], static_cast<::std::int64_t>(d) + 1);
  }

  sharded_csr<double> A(group, m.rows, m.cols, m.offsets.data(), m.colinds.data(), m.values.data(), bounds);
  for (size_t d = 0; d + 1 < A.num_shards(); d++)
  {
    EXPECT(A.shard(d + 1).row_begin == bounds[d]);
  }
  check_shards_match_reference(m, A);
}

void test_boundary_validation(place_group& group)
{
  if (group.size() < 2)
  {
    return; // boundary counts are trivial with a single place
  }
  const auto m = make_mixed_csr(100, 32, 3);

  auto expect_throws = [&](::std::vector<::std::int64_t> bounds) {
    bool threw = false;
    try
    {
      sharded_csr<double> A(group, m.rows, m.cols, m.offsets.data(), m.colinds.data(), m.values.data(), mv(bounds));
    }
    catch (const ::std::invalid_argument&)
    {
      threw = true;
    }
    EXPECT(threw);
  };

  // Wrong count
  expect_throws(::std::vector<::std::int64_t>(group.size(), 10));
  // Out of range
  {
    ::std::vector<::std::int64_t> bad(group.size() - 1, 10);
    bad.back() = m.rows + 1;
    expect_throws(mv(bad));
  }
  // Descending (needs >= 3 places to have 2 interior boundaries)
  if (group.size() >= 3)
  {
    ::std::vector<::std::int64_t> bad(group.size() - 1, 50);
    bad[0] = 60;
    bad[1] = 40;
    expect_throws(mv(bad));
  }
}

void test_row_partitioned_outputs(place_group& group)
{
  const auto m = make_mixed_csr(4001, 64, 11);
  sharded_csr<double> A(group, m.rows, m.cols, m.offsets.data(), m.colinds.data(), m.values.data());

  const ::std::int64_t n_cols = 8;
  auto C                      = A.make_row_partitioned(n_cols);
  EXPECT(C.num_shards() == A.num_shards()); // every shard has rows >= 1
  EXPECT(C.size() == static_cast<size_t>(m.rows * n_cols));
  for (size_t d = 0; d < A.num_shards(); d++)
  {
    EXPECT(C.shard(d).size == static_cast<size_t>(A.shard(d).rows * n_cols));
    EXPECT(C.shard(d).place == A.shard(d).place);
    EXPECT(C.shard(d).global_offset == static_cast<size_t>(A.shard(d).row_begin * n_cols));
  }

  // Contiguous variant: one VA range, exact logical boundaries.
  auto Cc = A.make_row_partitioned(n_cols, /* contiguous */ true);
  EXPECT(Cc.is_contiguous());
  EXPECT(Cc.contiguous_data() != nullptr);
  for (size_t d = 0; d < Cc.num_shards(); d++)
  {
    EXPECT(Cc.shard(d).data == Cc.contiguous_data() + Cc.shard(d).global_offset);
  }
}

void test_lib_state(place_group& group)
{
  const auto m = make_mixed_csr(500, 16, 5);
  sharded_csr<double> A(group, m.rows, m.cols, m.offsets.data(), m.colinds.data(), m.values.data());

  struct fake_state
  {
    int value = 0;
  };

  int makes  = 0;
  auto maker = [&] {
    makes++;
    auto* s  = new fake_state();
    s->value = 42;
    return s;
  };

  auto& s1 = A.lib_state<fake_state>("op_a", maker);
  auto& s2 = A.lib_state<fake_state>("op_a", maker);
  EXPECT(&s1 == &s2); // created once, cached
  EXPECT(makes == 1);
  EXPECT(s1.value == 42);

  auto& s3 = A.lib_state<fake_state>("op_b", maker); // different key: new slot
  EXPECT(makes == 2);
  EXPECT(&s3 != &s1);
}

// The piecewise-rate model is pure host math: check it on synthetic
// measurements, independently of any device timing.
void test_time_balanced_boundaries_model()
{
  const ::std::int64_t rows = 1000;
  // Uniform 4 nnz per row: nnz-balanced == row-balanced.
  ::std::vector<int> offsets(static_cast<size_t>(rows) + 1);
  for (size_t r = 0; r < offsets.size(); r++)
  {
    offsets[r] = static_cast<int>(4 * r);
  }

  // Balanced times: the boundary must stay put.
  {
    auto b = sharded_csr<double>::time_balanced_boundaries(rows, offsets.data(), {500}, {2.0, 2.0});
    EXPECT(b.size() == 1UL);
    EXPECT(b[0] == 500);
  }

  // Shard 0 measured 3x slower per nnz: its row count must SHRINK, and with
  // piecewise-constant rates the predicted-equal point is exact:
  // t0(r) = 3u * r, t1(r) = u * (1000 - r); equal predicted time at the
  // target T/2 = (3*500u + 500u)/2 = 1000u => r = 1000/3 -> boundary 334.
  {
    auto b = sharded_csr<double>::time_balanced_boundaries(rows, offsets.data(), {500}, {3.0, 1.0});
    EXPECT(b.size() == 1UL);
    EXPECT(b[0] < 500); // direction: slower shard shrinks
    EXPECT(b[0] == 334);
  }

  // Mirror image: shard 1 slower, boundary must move up symmetrically.
  {
    auto b = sharded_csr<double>::time_balanced_boundaries(rows, offsets.data(), {500}, {1.0, 3.0});
    EXPECT(b[0] > 500);
    EXPECT(b[0] == 500 + 167); // t0 covers target 2/3 of its region: 667
  }

  // Monotonicity in the skew: a larger measured skew moves the boundary at
  // least as far.
  {
    ::std::int64_t prev = 500;
    for (double skew : {1.5, 2.0, 4.0, 8.0})
    {
      auto b = sharded_csr<double>::time_balanced_boundaries(rows, offsets.data(), {500}, {skew, 1.0});
      EXPECT(b[0] <= prev);
      prev = b[0];
    }
    EXPECT(prev < 500);
  }

  // Size mismatch throws.
  {
    bool threw = false;
    try
    {
      sharded_csr<double>::time_balanced_boundaries(rows, offsets.data(), {250, 500}, {1.0, 1.0});
    }
    catch (const ::std::invalid_argument&)
    {
      threw = true;
    }
    EXPECT(threw);
  }
}
} // namespace

void test_from_device_and_contiguous(place_group& group)
{
  const auto m = make_mixed_csr(10007, 512, 43);

  // Upload the CSR; from_device must reproduce the host-built container.
  int* d_off    = nullptr;
  int* d_col    = nullptr;
  double* d_val = nullptr;
  cuda_safe_call(cudaMalloc(&d_off, m.offsets.size() * sizeof(int)));
  cuda_safe_call(cudaMalloc(&d_col, m.colinds.size() * sizeof(int)));
  cuda_safe_call(cudaMalloc(&d_val, m.values.size() * sizeof(double)));
  cuda_safe_call(cudaMemcpy(d_off, m.offsets.data(), m.offsets.size() * sizeof(int), cudaMemcpyDefault));
  cuda_safe_call(cudaMemcpy(d_col, m.colinds.data(), m.colinds.size() * sizeof(int), cudaMemcpyDefault));
  cuda_safe_call(cudaMemcpy(d_val, m.values.data(), m.values.size() * sizeof(double), cudaMemcpyDefault));

  auto A = sharded_csr<double>::from_device(group, m.rows, m.cols, d_off, d_col, d_val);
  EXPECT(A.num_rows() == m.rows);
  EXPECT(A.nnz() == m.nnz());
  EXPECT(!A.values_contiguous());
  EXPECT(A.contiguous_values() == nullptr);
  check_shards_match_reference(m, A);

  // Same split as the host-built container (same offsets => same boundaries).
  sharded_csr<double> H(group, m.rows, m.cols, m.offsets.data(), m.colinds.data(), m.values.data());
  for (size_t d = 0; d < A.num_shards(); d++)
  {
    EXPECT(A.shard(d).row_begin == H.shard(d).row_begin);
    EXPECT(A.shard(d).rows == H.shard(d).rows);
  }

  // Contiguous internals: values/colinds are exact nnz-slices, so the
  // one-VA-range backing reads back as the parent arrays through the base
  // pointers -- the seam that lets unmodified writers mutate the values.
  auto C = sharded_csr<double>::from_device(
    group, m.rows, m.cols, d_off, d_col, d_val, /*row_boundaries=*/{}, /*contiguous=*/true);
  EXPECT(C.values_contiguous());
  EXPECT(C.contiguous_values() != nullptr);
  EXPECT(C.contiguous_colinds() != nullptr);
  check_shards_match_reference(m, C);
  {
    ::std::vector<double> vals(static_cast<size_t>(m.nnz()));
    ::std::vector<int> cols(static_cast<size_t>(m.nnz()));
    cuda_safe_call(cudaMemcpy(vals.data(), C.contiguous_values(), vals.size() * sizeof(double), cudaMemcpyDefault));
    cuda_safe_call(cudaMemcpy(cols.data(), C.contiguous_colinds(), cols.size() * sizeof(int), cudaMemcpyDefault));
    EXPECT(::std::memcmp(vals.data(), m.values.data(), vals.size() * sizeof(double)) == 0);
    EXPECT(::std::memcmp(cols.data(), m.colinds.data(), cols.size() * sizeof(int)) == 0);
    // Shard views are exact offsets into the base pointers.
    for (size_t d = 0; d < C.num_shards(); d++)
    {
      const auto& sh = C.shard(d);
      if (sh.nnz > 0)
      {
        EXPECT(sh.values == C.contiguous_values() + sh.nnz_begin);
        EXPECT(sh.colinds == C.contiguous_colinds() + sh.nnz_begin);
      }
    }
  }

  cuda_safe_call(cudaFree(d_off));
  cuda_safe_call(cudaFree(d_col));
  cuda_safe_call(cudaFree(d_val));
}

// ---------------------------------------------------------------------------
// fork_from / join_into: ordering declarations bridging a caller stream and
// the matrix's per-shard streams. Producer on the caller stream scales the
// values in place -> fork_from -> per-shard consumers on the shard streams
// sum each shard's values -> join_into -> readback on the caller stream. The
// ONLY host synchronization is the final caller-stream sync.
// ---------------------------------------------------------------------------

__global__ void spin_kernel(long long cycles)
{
  const long long start = clock64();
  while (clock64() - start < cycles)
  {
  }
}

__global__ void scale_values_kernel(double* values, ::std::int64_t n, double factor)
{
  const ::std::int64_t i = blockIdx.x * static_cast<::std::int64_t>(blockDim.x) + threadIdx.x;
  if (i < n)
  {
    values[i] *= factor;
  }
}

// Single-thread sum: the test matrices keep per-shard nnz small.
__global__ void shard_sum_kernel(const double* values, ::std::int64_t n, double* out)
{
  double acc = 0.0;
  for (::std::int64_t i = 0; i < n; i++)
  {
    acc += values[i];
  }
  *out = acc;
}

void test_fork_join(place_group& group)
{
  const auto m = make_mixed_csr(4001, 256, 7);
  sharded_csr<double> A(group, m.rows, m.cols, m.offsets.data(), m.colinds.data(), m.values.data());

  const double factor = 3.0;

  // Host reference: per-shard sums of the SCALED values.
  ::std::vector<double> expected(A.num_shards(), 0.0);
  for (size_t d = 0; d < A.num_shards(); d++)
  {
    const auto& sh = A.shard(d);
    for (::std::int64_t k = 0; k < sh.nnz; k++)
    {
      expected[d] += factor * m.values[static_cast<size_t>(sh.nnz_begin + k)];
    }
  }

  cudaStream_t caller = nullptr;
  cuda_safe_call(cudaStreamCreate(&caller));
  double* d_sums = nullptr;
  cuda_safe_call(cudaMalloc(&d_sums, A.num_shards() * sizeof(double)));

  // Producer on the caller stream: scale every shard's values in place.
  spin_kernel<<<1, 1, 0, caller>>>(20000000);
  for (size_t d = 0; d < A.num_shards(); d++)
  {
    auto& sh = A.shard(d);
    if (sh.nnz > 0)
    {
      const unsigned int blocks = static_cast<unsigned int>((sh.nnz + 255) / 256);
      scale_values_kernel<<<blocks, 256, 0, caller>>>(sh.values, sh.nnz, factor);
    }
  }

  A.fork_from(caller); // shard streams now depend on the producer

  for (size_t d = 0; d < A.num_shards(); d++)
  {
    const auto& sh = A.shard(d);
    shard_sum_kernel<<<1, 1, 0, sh.stream>>>(sh.values, sh.nnz, d_sums + d);
  }

  A.join_into(caller); // the caller stream now depends on every consumer

  ::std::vector<double> sums(A.num_shards(), 0.0);
  cuda_safe_call(cudaMemcpyAsync(sums.data(), d_sums, sums.size() * sizeof(double), cudaMemcpyDefault, caller));
  cuda_safe_call(cudaStreamSynchronize(caller)); // the only host sync

  for (size_t d = 0; d < A.num_shards(); d++)
  {
    EXPECT(::std::abs(sums[d] - expected[d]) <= 1e-11 * (1.0 + ::std::abs(expected[d])));
  }

  cuda_safe_call(cudaFree(d_sums));
  cuda_safe_call(cudaStreamDestroy(caller));
}

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group::by_locality_domains();

  test_construction_default_split(group);
  test_explicit_boundaries(group);
  test_boundary_validation(group);
  test_row_partitioned_outputs(group);
  test_lib_state(group);
  test_from_device_and_contiguous(group);
  test_time_balanced_boundaries_model();
  test_fork_join(group);

  return 0;
}
