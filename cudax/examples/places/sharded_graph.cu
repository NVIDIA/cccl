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
 * @brief Composing sharded algorithms over a partitioned graph: a
 *        library-oriented data structure expressed as a set of sharded
 *        views, with graph primitives written as compositions of the
 *        generic sharded algorithms.
 *
 * The structure is a vertex-range-partitioned CSR — the layout multi-GPU
 * graph libraries use (cuGraph's vertex partitioning has this shape): each
 * shard owns a contiguous vertex interval, that interval's adjacency as a
 * LOCAL (rebased) row-offsets array, global column indices, and edge
 * values. The graph type below is deliberately just a struct of components
 * that each model `sharded_view` — the concepts describe the components,
 * and the composite needs no concept of its own.
 *
 * Two spellings do the heavy lifting:
 *
 * - SHIFTED-ALIAS VIEWS: a CSR row-offsets buffer has n+1 entries for n
 *   vertices, so it cannot be co-partitioned with vertex-space vectors.
 *   Building TWO views over the SAME buffer — `lo = offsets[0..n)`,
 *   `hi = offsets[1..n+1)` — yields views that are each co-partitioned
 *   with every vertex vector: offset-structured computations become plain
 *   `zip_transform`s, and the pair is exactly `segmented_reduce`'s
 *   segments description.
 * - CONTIGUOUS VERTEX VECTORS: gathering `x[col[e]]` reads across shards;
 *   `allocate_contiguous` gives the vertex vector one base pointer, which
 *   is what makes the gather a plain device-side load within one process.
 *   (Cross-process distribution is a separate layer above this one.)
 *
 * Graph primitives as compositions:
 *  1. vertex degrees            = zip_transform(deg, hi, lo)
 *  2. per-vertex neighbor
 *     reduce (SpMV-shaped)      = zip_transform (edge-space gather*weight)
 *                                 then segmented_reduce (edges -> vertices)
 *  3. frontier size             = count_if over a vertex property
 *  4. frontier contents (RAGGED) = out-of-place copy_if of the vertex ids
 *                                 into an owning array whose per-shard sizes
 *                                 become the data-dependent frontier sizes
 */

#include <cuda/experimental/sharded.cuh>

#include <cmath>
#include <cstdio>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
struct degree_op
{
  __device__ int operator()(int hi, int lo) const
  {
    return hi - lo;
  }
};

struct gather_multiply_op
{
  const float* x_base; // contiguous vertex-space base
  __device__ float operator()(int col, float w) const
  {
    return x_base[col] * w;
  }
};

struct sum_op
{
  __host__ __device__ float operator()(float a, float b) const
  {
    return a + b;
  }
};

struct deg_at_least_3
{
  __device__ bool operator()(int d) const
  {
    return d >= 3;
  }
};

struct frontier_pred // applied to vertex IDS; degree gathered via the contiguous base
{
  const int* deg_base;
  __device__ bool operator()(int v) const
  {
    return deg_base[v] >= 3;
  }
};

// A graph is a struct of components that each model `sharded_view`; the
// composite itself needs no concept.
struct sharded_csr_graph
{
  basic_sharded_view<const int> offsets_lo; // vertex space: offsets[0..n)
  basic_sharded_view<const int> offsets_hi; // vertex space: offsets[1..n+1)
  basic_sharded_view<const int> col_indices; // edge space (global vertex ids)
  basic_sharded_view<const float> values; // edge space
  ::std::size_t num_vertices = 0;
  ::std::size_t num_edges    = 0;
};
static_assert(sharded_view<decltype(sharded_csr_graph{}.offsets_lo)>, "components model the concept");
static_assert(sharded_view<decltype(sharded_csr_graph{}.values)>, "components model the concept");
} // namespace

int main()
{
  cuda_safe_call(cuInit(0));
  cuda_safe_call(cudaSetDevice(0));

  auto group            = place_group{make_locality_domain_grid()};
  const ::std::size_t P = group.size();
  auto envs             = group.envs(0);
  ::std::printf("place_group with %zu place(s)\n", P);

  // -------------------------------------------------------------------------
  // Host graph: V vertices; v -> (v+1)%V, (v+3)%V, and (v even) -> v/2.
  // Weight of (v,u) = 1 + (v+u) % 5.
  // -------------------------------------------------------------------------
  const ::std::size_t V = 10007; // deliberately not divisible by P
  ::std::vector<::std::vector<int>> adj(V);
  auto weight = [](::std::size_t v, int u) {
    return static_cast<float>(1 + (v + static_cast<::std::size_t>(u)) % 5);
  };
  for (::std::size_t v = 0; v < V; v++)
  {
    adj[v].push_back(static_cast<int>((v + 1) % V));
    adj[v].push_back(static_cast<int>((v + 3) % V));
    if (v % 2 == 0)
    {
      adj[v].push_back(static_cast<int>(v / 2));
    }
  }

  // Contiguous vertex interval per shard.
  ::std::vector<::std::size_t> v_begin(P + 1, 0);
  for (::std::size_t g = 0; g < P; g++)
  {
    v_begin[g + 1] = v_begin[g] + V / P + (g < V % P ? 1 : 0);
  }

  // Per-shard local CSR (rebased offsets, global column ids) on the host.
  ::std::vector<::std::vector<int>> h_off(P), h_col(P);
  ::std::vector<::std::vector<float>> h_val(P);
  for (::std::size_t g = 0; g < P; g++)
  {
    h_off[g].push_back(0);
    for (::std::size_t v = v_begin[g]; v < v_begin[g + 1]; v++)
    {
      for (int u : adj[v])
      {
        h_col[g].push_back(u);
        h_val[g].push_back(weight(v, u));
      }
      h_off[g].push_back(static_cast<int>(h_col[g].size()));
    }
  }
  ::std::size_t E = 0;
  for (::std::size_t g = 0; g < P; g++)
  {
    E += h_col[g].size();
  }

  // -------------------------------------------------------------------------
  // Device buffers through the environments' memory resources: the binding
  // tier does the placement, nothing here names a device.
  // -------------------------------------------------------------------------
  ::std::vector<int*> d_off(P), d_col(P);
  ::std::vector<float*> d_val(P);
  for (::std::size_t g = 0; g < P; g++)
  {
    const auto& env = envs[g];
    const auto strm = ::cuda::get_stream(env);
    stream_scope sc(strm.get());
    auto mr  = ::cuda::mr::get_memory_resource(env);
    d_off[g] = static_cast<int*>(mr.allocate(strm, h_off[g].size() * sizeof(int), 256));
    d_col[g] = static_cast<int*>(mr.allocate(strm, (h_col[g].empty() ? 1 : h_col[g].size()) * sizeof(int), 256));
    d_val[g] = static_cast<float*>(mr.allocate(strm, (h_val[g].empty() ? 1 : h_val[g].size()) * sizeof(float), 256));
    cuda_safe_call(
      cudaMemcpyAsync(d_off[g], h_off[g].data(), h_off[g].size() * sizeof(int), cudaMemcpyHostToDevice, strm.get()));
    cuda_safe_call(
      cudaMemcpyAsync(d_col[g], h_col[g].data(), h_col[g].size() * sizeof(int), cudaMemcpyHostToDevice, strm.get()));
    cuda_safe_call(
      cudaMemcpyAsync(d_val[g], h_val[g].data(), h_val[g].size() * sizeof(float), cudaMemcpyHostToDevice, strm.get()));
    cuda_safe_call(cudaStreamSynchronize(strm.get()));
  }

  // -------------------------------------------------------------------------
  // The composite: shifted-alias offset views + edge views.
  // -------------------------------------------------------------------------
  sharded_csr_graph graph;
  {
    ::std::vector<cuda::std::span<const int>> lo(P), hi(P), col(P);
    ::std::vector<cuda::std::span<const float>> val(P);
    for (::std::size_t g = 0; g < P; g++)
    {
      const ::std::size_t nv = v_begin[g + 1] - v_begin[g];
      lo[g]                  = {d_off[g], nv};
      hi[g]                  = {d_off[g] + 1, nv}; // the shifted alias
      col[g]                 = {d_col[g], h_col[g].size()};
      val[g]                 = {d_val[g], h_val[g].size()};
    }
    graph.offsets_lo   = make_sharded_view(lo);
    graph.offsets_hi   = make_sharded_view(hi);
    graph.col_indices  = make_sharded_view(col);
    graph.values       = make_sharded_view(val);
    graph.num_vertices = V;
    graph.num_edges    = E;
  }

  // Vertex- and edge-space property vectors, co-partitioned with the graph.
  ::std::vector<::std::size_t> vsizes(P), esizes(P);
  for (::std::size_t g = 0; g < P; g++)
  {
    vsizes[g] = v_begin[g + 1] - v_begin[g];
    esizes[g] = h_col[g].size();
  }
  auto deg = sharded_array<int>::allocate_contiguous(group, V, 0); // contiguous: the frontier predicate gathers by
                                                                   // vertex id
  auto y = sharded_array<float>::allocate(group, vsizes, 0);
  auto z = sharded_array<float>::allocate(group, esizes, 0);
  auto x = sharded_array<float>::allocate_contiguous(group, V, 0);

  ::std::vector<float> h_x(V);
  for (::std::size_t v = 0; v < V; v++)
  {
    h_x[v] = 0.25f + static_cast<float>(v % 7);
  }
  x.copy_from_host(h_x.data());

  bool ok = true;

  // =========================================================================
  // 1. Vertex degrees: the CSR off-by-one absorbed by the shifted aliases.
  // =========================================================================
  zip_transform(deg, envs, degree_op{}, default_call_env{}, graph.offsets_hi, graph.offsets_lo);
  {
    ::std::vector<int> h_deg(V);
    deg.copy_to_host(h_deg.data());
    for (::std::size_t v = 0; v < V; v++)
    {
      ok = ok && (h_deg[v] == static_cast<int>(adj[v].size()));
    }
  }
  ::std::printf("degrees as zip_transform over shifted aliases: %s\n", ok ? "OK" : "MISMATCH");

  // =========================================================================
  // 2. Per-vertex neighbor reduce: y[v] = sum over e in [lo[v],hi[v]) of
  //    x[col[e]] * w[e] — an edge-space gather+multiply, then the
  //    edges-to-vertices segmented reduction.
  // =========================================================================
  const float* x_base = static_cast<const float*>(x.shard(0).data);
  zip_transform(z, envs, gather_multiply_op{x_base}, default_call_env{}, graph.col_indices, graph.values);
  segmented_reduce(z, envs, graph.offsets_lo, graph.offsets_hi, y, sum_op{}, 0.0f);
  {
    ::std::vector<float> h_y(V);
    y.copy_to_host(h_y.data());
    for (::std::size_t v = 0; v < V; v++)
    {
      float ref = 0.0f;
      for (int u : adj[v])
      {
        ref += h_x[static_cast<::std::size_t>(u)] * weight(v, u);
      }
      ok = ok && (::std::abs(h_y[v] - ref) <= 1e-4f * (1.0f + ::std::abs(ref)));
    }
  }
  ::std::printf("neighbor reduce as gather zip_transform + segmented_reduce: %s\n", ok ? "OK" : "MISMATCH");

  // =========================================================================
  // 3. Frontier size: count_if over a vertex property.
  // =========================================================================
  const long long frontier = count_if(deg, envs, deg_at_least_3{});
  long long ref_frontier   = 0;
  for (::std::size_t v = 0; v < V; v++)
  {
    ref_frontier += (adj[v].size() >= 3) ? 1 : 0;
  }
  ok = ok && (frontier == ref_frontier);
  ::std::printf("frontier size as count_if: %s (|frontier| = %lld)\n", ok ? "OK" : "MISMATCH", frontier);

  // =========================================================================
  // 4. Frontier contents: a RAGGED result. Vertex ids pass through an
  //    out-of-place copy_if into an owning array; each shard's size becomes
  //    its data-dependent frontier count, committed atomically (offsets
  //    re-tile, the structure stays valid). The source ids are untouched.
  // =========================================================================
  auto ids = sharded_array<int>::allocate(group, vsizes, 0);
  iota(ids, 0); // ids[v] = v (global vertex id)
  auto frontier_ids   = sharded_array<int>::allocate(group, vsizes, 0); // capacity = worst case
  const int* deg_base = static_cast<const int*>(deg.shard(0).data); // contiguous base

  const size_t f_kept = copy_if(ids, frontier_ids, frontier_pred{deg_base});
  ok                  = ok && (f_kept == static_cast<size_t>(ref_frontier));
  {
    ::std::vector<int> h_f(f_kept);
    frontier_ids.copy_to_host(h_f.data());
    size_t k = 0;
    for (::std::size_t g = 0; g < P; g++)
    {
      for (::std::size_t v = v_begin[g]; v < v_begin[g + 1]; v++)
      {
        if (adj[v].size() >= 3)
        {
          ok = ok && (k < f_kept) && (h_f[k++] == static_cast<int>(v));
        }
      }
    }
    ok = ok && (k == f_kept);
  }
  ::std::printf("frontier contents as ragged copy_if: %s (%zu ids, per-shard sizes data-dependent)\n",
                ok ? "OK" : "MISMATCH",
                f_kept);

  for (::std::size_t g = 0; g < P; g++)
  {
    const auto& env = envs[g];
    const auto strm = ::cuda::get_stream(env);
    auto mr         = ::cuda::mr::get_memory_resource(env);
    mr.deallocate(strm, d_off[g], h_off[g].size() * sizeof(int), 256);
    mr.deallocate(strm, d_col[g], (h_col[g].empty() ? 1 : h_col[g].size()) * sizeof(int), 256);
    mr.deallocate(strm, d_val[g], (h_val[g].empty() ? 1 : h_val[g].size()) * sizeof(float), 256);
    cuda_safe_call(cudaStreamSynchronize(strm.get()));
  }

  if (!ok)
  {
    ::std::printf("FAILED\n");
    return 1;
  }
  ::std::printf("PASSED (V=%zu, E=%zu, P=%zu)\n", V, E, P);
  return 0;
}
