//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Computes the PageRank for vertices within a graph
 *
 */

#include <cuda/experimental/stf.cuh>

#include <vector>

using namespace cuda::experimental::stf;

/**
 * @brief Calculates the PageRank for a given vertex.
 *
 * @param idx               The index of the vertex for which PageRank is being calculated.
 * @param loffsets         Slice containing the offset vector of the CSR representation.
 * @param lnonzeros        Slice containing the non-zero elements (neighbors) vector of the CSR representation.
 * @param lpage_rank       Slice containing current PageRank values for each vertex.
 * @param lnew_page_rank   Slice containing where new PageRank values will be stored.
 * @param lpersonalization Slice containing the personalization vector for each vertex.
 */
__device__ void calculating_pagerank(
  int idx,
  const slice<const int>& loffsets,
  const slice<const int>& lnonzeros,
  const slice<const float>& lpage_rank,
  slice<float>& lnew_page_rank,
  const slice<const float>& lpersonalization)
{
  float rank_sum = 0.0;
  for (int i = loffsets[idx]; i < loffsets[idx + 1]; i++)
  {
    int neighbor   = lnonzeros[i];
    int out_degree = loffsets[neighbor + 1] - loffsets[neighbor];
    rank_sum += lpage_rank[neighbor] / out_degree;
  }
  lnew_page_rank[idx] = 0.85 * rank_sum + (1.0 - 0.85) * lpersonalization[idx];
}

/**
 * @brief Computes PageRank using the power iteration method
 *
 * @param ctx               The CUDASTF context
 * @param loffsets         Logical data for CSR offset vector
 * @param lnonzeros        Logical data for CSR non-zero elements vector
 * @param lpage_rank       Logical data for current PageRank values
 * @param lpersonalization Logical data for personalization vector
 * @param num_vertices     Number of vertices in the graph
 * @param NITER            Maximum number of iterations
 * @param tolerance        Convergence tolerance
 */
void compute_pagerank(
  stackable_ctx& ctx,
  stackable_logical_data<slice<int>>& loffsets,
  stackable_logical_data<slice<int>>& lnonzeros,
  stackable_logical_data<slice<float>>& lpage_rank,
  stackable_logical_data<slice<float>>& lpersonalization,
  int num_vertices,
  int NITER,
  float tolerance)
{
  // Create local temporary buffer and convergence tracking
  auto lnew_page_rank = ctx.logical_data(lpage_rank.shape());
  auto lmax_diff      = ctx.logical_data(shape_of<scalar_view<float>>());
  auto liter          = ctx.logical_data(shape_of<scalar_view<int>>());

  // Initialize iteration counter
  ctx.parallel_for(box(1), liter.write())->*[] __device__(size_t, auto iter) {
    *iter = 0;
  };

  {
    auto while_guard = ctx.while_graph_scope();

    // Calculate Current Iteration PageRank
    ctx.parallel_for(
      box(num_vertices),
      loffsets.read(),
      lnonzeros.read(),
      lpage_rank.rw(),
      lnew_page_rank.write(),
      lpersonalization.read(),
      lmax_diff.reduce(reducer::maxval<float>{}))
        ->*
      [] __device__(
        size_t idx,
        auto loffsets,
        auto lnonzeros,
        auto lpage_rank,
        auto lnew_page_rank,
        auto lpersonalization,
        auto& max_diff) {
        calculating_pagerank(idx, loffsets, lnonzeros, lpage_rank, lnew_page_rank, lpersonalization);
        max_diff = ::std::max(max_diff, lnew_page_rank[idx] - lpage_rank[idx]);
      };

    // Update PageRank Values
    ctx.parallel_for(lpage_rank.shape(), lpage_rank.write(), lnew_page_rank.read())
        ->*[] __device__(size_t i, auto page_rank, auto new_page_rank) {
              page_rank(i) = new_page_rank(i);
            };

    while_guard.update_cond(lmax_diff.read(), liter.rw())->*[NITER, tolerance] __device__(auto max_diff, auto iter) {
      bool converged   = (*max_diff < tolerance);
      bool max_reached = ((*iter)++ >= NITER); // Maximum iteration limit
      return !converged && !max_reached; // Continue if not converged and under limit
    };
  }
}

int main()
{
#if _CCCL_CTK_BELOW(12, 4)
  fprintf(stderr, "Waiving example: while_graph_scope is only available since CUDA 12.4.\n");
  return 0;
#else
  stackable_ctx ctx;

  // row offsets in CSR format
  std::vector<int> offsets = {0, 4, 11, 12, 14, 15, 16, 18, 19, 20};
  // edges in CSR format
  std::vector<int> nonzeros = {1, 2, 3, 6, 0, 3, 4, 5, 6, 7, 8, 0, 0, 1, 1, 1, 0, 1, 1, 1};

  int num_vertices        = offsets.size() - 1;
  float init_rank         = 1.0f / num_vertices;
  float tolerance         = 1e-6f;
  int NITER               = 100;
  int num_personalization = 4;

  ::std::vector<stackable_logical_data<slice<float>>> lpage_rank_slices;
  for (int i = 0; i < num_personalization; i++)
  {
    lpage_rank_slices.push_back(ctx.logical_data(shape_of<slice<float>>(num_vertices)));
  }

  auto loffsets  = ctx.logical_data(&offsets[0], offsets.size());
  auto lnonzeros = ctx.logical_data(&nonzeros[0], nonzeros.size());

  loffsets.set_read_only();
  lnonzeros.set_read_only();

  {
    auto scope = ctx.graph_scope();
    for (int p = 0; p < num_personalization; p++)
    {
      // Initialize PageRank values to uniform distribution
      ctx.parallel_for(lpage_rank_slices[p].shape(), lpage_rank_slices[p].write())
          ->*[init_rank] __device__(size_t i, auto page_rank) {
                page_rank(i) = init_rank;
              };

      // Create personalization vector (uniform for this example)
      auto lpersonalization = ctx.logical_data(shape_of<slice<float>>(num_vertices));
      ctx.parallel_for(lpersonalization.shape(), lpersonalization.write())
          ->*[init_rank] __device__(size_t i, auto lpersonalization) {
                lpersonalization(i) = init_rank;
              };

      compute_pagerank(ctx, loffsets, lnonzeros, lpage_rank_slices[p], lpersonalization, num_vertices, NITER, tolerance);
    }
  }

  for (int p = 0; p < num_personalization; p++)
  {
    ctx.host_launch(lpage_rank_slices[p].read())->*[p, num_vertices] __host__(slice<const float> page_rank) {
      double sum_pageranks = 0.0;
      for (int64_t i = 0; i < num_vertices; i++)
      {
        sum_pageranks += page_rank[i];
      }
      printf("Page rank answer for personalization %d is %s.\n",
             p,
             abs(sum_pageranks - 1.0) < 0.001 ? "correct" : "not correct");

      // Print first few results for verification
      printf("Personalization %d - First 5 vertices: ", p);
      for (size_t i = 0; i < std::min(5UL, page_rank.size()); ++i)
      {
        printf("%.6f ", page_rank[i]);
      }
      printf("\n");
    };
  }

  ctx.finalize();

  return 0;
#endif // !_CCCL_CTK_BELOW(12, 4)
}
