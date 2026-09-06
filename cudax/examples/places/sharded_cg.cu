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
 * @brief Conjugate gradient over sharded arrays: a closed-source library
 *        CONSUMING the sharded structures inside an otherwise open pipeline.
 *
 * The point of this example is composition, not a faster SpMV. One placed
 * data layout flows through the whole iteration with no resharding and no
 * copies:
 *
 *   q = A p          — cuSPARSE tier (sparse.cuh, opt-in): one SM-confined
 *                      vendor call per shard over the row-partitioned CSR
 *   dot products     — generic tier: zip_transform + reduce_into
 *   axpy updates     — generic tier: zip_transform
 *
 * The vendor product itself is placement-NEUTRAL at full power (measured);
 * that is fine — and is the enabler: because the closed call accepts the
 * sharded structures where they live, every open-tier stage around it keeps
 * domain-local data, and the locality pays where it pays (power-constrained
 * operation, parts where cross-domain bandwidth is relatively scarcer,
 * downstream consumers) without this code changing.
 *
 * One structural detail carries the composition: the row-split SpMV needs
 * its INPUT vector whole at every place (each shard's rows gather arbitrary
 * columns), while the open tier wants the same vector sharded. The
 * CONTIGUOUS grade bridges the two: `allocate_contiguous` backs the shards
 * with one VMM range, so `p` is simultaneously per-place shard views (for
 * transforms) and a single flat pointer (for the vendor call) — zero copies
 * either way.
 *
 * Problem: 7-point Poisson stencil on a g^3 grid (SPD), b = A * ones, CG
 * from x0 = 0; converged x must recover ones.
 *
 * This example is written for CLARITY, not solver throughput: scalar
 * recurrences run on the host (one sync per dot), and each dot is a
 * two-pass zip_transform + sum. A production spelling would keep the
 * scalars on the device (reduce_into is async and capture-compatible),
 * fuse the dots, and replay iterations as a CUDA graph — none of which
 * changes the point demonstrated here: one placed layout, open and closed
 * tiers composing on it in place.
 */

#include <cuda/experimental/sharded.cuh>

#include <cmath>
#include <cstdio>
#include <vector>

#include <cuda/experimental/__sharded/sparse.cuh> // opt-in vendor tier

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{

// 7-point Poisson on a g^3 grid: A(i,i) = 6, A(i,j) = -1 for the 6 axis
// neighbors. SPD, uniform ~7 nnz/row, so an even row split is also
// nnz-balanced. deg(i) = number of in-bounds neighbors of cell i.
struct poisson
{
  int g;
  __host__ __device__ int deg(int64_t i) const
  {
    const int x = static_cast<int>(i % g), y = static_cast<int>((i / g) % g), z = static_cast<int>(i / (int64_t(g) * g));
    return (x > 0) + (x + 1 < g) + (y > 0) + (y + 1 < g) + (z > 0) + (z + 1 < g);
  }
  // Writes row i's segment (ascending columns) at colinds/values[offs[i]].
  __device__ void write_row(int64_t i, const int* offs, int* colinds, double* values) const
  {
    const int64_t gg = int64_t(g) * g;
    int o            = offs[i];
    auto push        = [&](int64_t col, double v) {
      colinds[o] = static_cast<int>(col);
      values[o]  = v;
      o++;
    };
    const int x = static_cast<int>(i % g), y = static_cast<int>((i / g) % g), z = static_cast<int>(i / gg);
    if (z > 0)
    {
      push(i - gg, -1.0);
    }
    if (y > 0)
    {
      push(i - g, -1.0);
    }
    if (x > 0)
    {
      push(i - 1, -1.0);
    }
    push(i, 6.0);
    if (x + 1 < g)
    {
      push(i + 1, -1.0);
    }
    if (y + 1 < g)
    {
      push(i + g, -1.0);
    }
    if (z + 1 < g)
    {
      push(i + gg, -1.0);
    }
  }
};

// dot(a, b) through the generic tier: multiply into per-place scratch, then
// the tier's host-returning sum (synchronous contract: one sync per dot).
double dot(const sharded_array<double>& a, const sharded_array<double>& b, sharded_array<double>& scratch)
{
  zip_transform(
    scratch,
    [] __device__(double x, double y) {
      return x * y;
    },
    a,
    b);
  return sum(scratch);
}

} // namespace

int main()
{
  place_group group{make_locality_domain_grid()};
  const size_t nd = group.size();
  ::std::printf("CG over %zu place(s)\n", nd);

  if (!contiguous_backing_supported())
  {
    ::std::printf("contiguous backing not supported on this device: skipping\n");
    return 0;
  }

  const int g     = 96; // 884k rows, ~6.1M nnz
  const int64_t n = int64_t(g) * g * g;
  const poisson P{g};
  // Face cells miss one neighbor each: nnz is analytic, nothing to count.
  const int64_t nnz = 7 * n - 6 * int64_t(g) * g;

  // Even row split shared by the matrix and every vector (uniform stencil:
  // even rows == nnz-balanced). The boundaries passed to the CSR are the
  // interior cut points; the same sizes shape the vectors.
  ::std::vector<::std::int64_t> bounds;
  ::std::vector<size_t> sizes;
  for (size_t d = 0; d < nd; d++)
  {
    const int64_t begin = static_cast<int64_t>(n * d / nd);
    const int64_t end   = static_cast<int64_t>(n * (d + 1) / nd);
    sizes.push_back(static_cast<size_t>(end - begin));
    if (d > 0)
    {
      bounds.push_back(begin);
    }
  }

  // DEVICE-NATIVE assembly, all through the generic tier — the matrix never
  // exists on the host: per-row counts by tabulate (data[i] = f(global i),
  // with a zero tail slot), then one exclusive scan makes them offsets — its
  // construction puts total nnz in the tail; here it is analytic anyway.
  auto offsets = sharded_array<int>::allocate_contiguous(group, static_cast<size_t>(n) + 1);
  tabulate(offsets, [P, n] __device__(size_t i) {
    return static_cast<int64_t>(i) < n ? 1 + P.deg(static_cast<int64_t>(i)) : 0;
  });
  exclusive_scan(offsets, ::cuda::std::plus<>{}, 0);

  // Column indices and values: a for_each over the row space (spelled as a
  // tabulate over global row ids) scatters each row's segment through the
  // contiguous views.
  auto colinds = sharded_array<int>::allocate_contiguous(group, static_cast<size_t>(nnz));
  auto values  = sharded_array<double>::allocate_contiguous(group, static_cast<size_t>(nnz));
  {
    auto rows = sharded_array<int>::allocate(group, sizes);
    offsets.sync(); // the scatter below reads offsets across lane boundaries
    int* d_ci        = colinds.contiguous_data();
    double* d_v      = values.contiguous_data();
    const int* d_off = offsets.contiguous_data();
    tabulate(rows, [P, d_off, d_ci, d_v] __device__(size_t i) {
      P.write_row(static_cast<int64_t>(i), d_off, d_ci, d_v);
      return static_cast<int>(i);
    });
    rows.sync();
  }

  // The operator, row-partitioned over the group; the vendor-state scope
  // (handles + per-shard plans) is caller-held, like every tier resource.
  auto A = sharded_csr<double>::from_device(
    group, n, n, offsets.contiguous_data(), colinds.contiguous_data(), values.contiguous_data(), bounds);
  cusparse_handles handles(group);
  spmv_plan<double> plan(handles, A);

  // Iteration state. p is CONTIGUOUS-backed: shard views for the open tier,
  // one flat pointer for the closed one.
  auto x       = sharded_array<double>::allocate(group, sizes);
  auto r       = sharded_array<double>::allocate(group, sizes);
  auto q       = sharded_array<double>::allocate(group, sizes);
  auto scratch = sharded_array<double>::allocate(group, sizes);
  auto p       = sharded_array<double>::allocate_contiguous(group, static_cast<size_t>(n));
  // Seam stream for the one cross-lane coupling in the loop (see below).
  const cudaStream_t seam = group.get_stream(0, 0);

  // b = A * ones is the row-sum: analytically 6 - deg(i) for this stencil,
  // tabulated in place — the host never sees b either. Start from x0 = 0,
  // so r0 = b and p0 = r0.
  fill(x, 0.0);
  for (auto* v : {&r, &p})
  {
    tabulate(*v, [P] __device__(size_t i) {
      return 6.0 - P.deg(static_cast<int64_t>(i));
    });
  }

  double rr = dot(r, r, scratch);
  const double rr0 = rr;
  ::std::printf("initial |r|^2 = %.6e\n", rr0);

  const double rel_tol = 1e-14; // on |r|^2
  const int max_iters  = 400;
  int it               = 0;
  for (; it < max_iters && rr > rel_tol * rr0; it++)
  {
    // Closed tier: the vendor call consumes the sharded operator and the
    // contiguous view of p, and writes the row-partitioned q — in place,
    // where every operand lives.
    //
    // Composition contract: the generic-tier convenience calls in this loop
    // are SYNCHRONOUS (empty call environment), so they order one another
    // through the host. spmv is the one lane-asynchronous call (it enqueues
    // on the MATRIX's shard streams and records no edges), so the CONSUMERS
    // of q own the coupling — expressed as device-side event edges (host
    // synchronization is unnecessary): the matrix's lanes join a seam
    // stream, and the lanes that will read q (the dot's scratch, and r's
    // update) fork from it. The fully asynchronous spelling (stream-carrying
    // call environments + lane_wait edges + graph replay) is the production
    // path, deliberately not taken.
    spmv(plan, p.contiguous_data(), q);
    A.join_into(seam);
    scratch.fork_from(seam);
    r.fork_from(seam);

    const double pq    = dot(p, q, scratch);
    const double alpha = rr / pq;

    // Open tier: the same shards, no movement.
    zip_transform(
      x,
      [alpha] __device__(double xv, double pv) {
        return xv + alpha * pv;
      },
      x,
      p);
    zip_transform(
      r,
      [alpha] __device__(double rv, double qv) {
        return rv - alpha * qv;
      },
      r,
      q);

    const double rr_new = dot(r, r, scratch);
    const double beta   = rr_new / rr;
    rr                  = rr_new;
    zip_transform(
      p,
      [beta] __device__(double rv, double pv) {
        return rv + beta * pv;
      },
      r,
      p);

    if (it % 25 == 0)
    {
      ::std::printf("  iter %3d: |r|^2 = %.6e\n", it, rr);
    }
  }
  ::std::printf("converged in %d iterations: |r|^2 = %.6e (rel %.3e)\n", it, rr, rr / rr0);

  // x must recover ones: max |x_i - 1| through the generic tier (x is not
  // needed past this point, so the transform runs in place).
  transform(x, [] __device__(double xv) {
    return fabs(xv - 1.0);
  });
  const double max_err = ::cuda::experimental::sharded::max(x);
  ::std::printf("max |x - 1| = %.3e\n", max_err);

  // The error bound carries the conditioning of the operator
  // (kappa ~ (g/pi)^2 ~ 1e3 here), so it is checked looser than the residual.
  const bool pass = (rr <= rel_tol * rr0) && (max_err < 1e-6 * 1e3);
  ::std::printf("%s\n", pass ? "PASS" : "FAIL");
  return pass ? 0 : 1;
}
