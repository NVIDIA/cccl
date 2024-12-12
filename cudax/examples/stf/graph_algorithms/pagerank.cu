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
 * @param init_rank         The initial PageRank value to be used in the calculation.
 */
__device__ void calculating_pagerank(
  int idx,
  const slice<const int>& loffsets,
  const slice<const int>& lnonzeros,
  const slice<const float>& lpage_rank,
  slice<float>& lnew_page_rank,
  float init_rank)
{
  float rank_sum = 0.0;
  for (int i = loffsets[idx]; i < loffsets[idx + 1]; i++)
  {
    int neighbor   = lnonzeros[i];
    int out_degree = loffsets[neighbor + 1] - loffsets[neighbor];
    rank_sum += lpage_rank[neighbor] / out_degree;
  }
  lnew_page_rank[idx] = 0.85 * rank_sum + (1.0 - 0.85) * init_rank;
}

int main()
{
  stream_ctx ctx;

  // row offsets in CSR format
  std::vector<int> offsets = {0, 4, 11, 12, 14, 15, 16, 18, 19, 20};
  // edges in CSR format
  std::vector<int> nonzeros = {1, 2, 3, 6, 0, 3, 4, 5, 6, 7, 8, 0, 0, 1, 1, 1, 0, 1, 1, 1};

  int num_vertices = offsets.size() - 1;
  float init_rank  = 1.0f / num_vertices;
  float tolerance  = 1e-6f;
  int NITER        = 100;

  // output pageranks for each vertex
  std::vector<float> page_rank(num_vertices, init_rank);
  std::vector<float> new_page_rank(num_vertices);

  auto loffsets       = ctx.logical_data(&offsets[0], offsets.size());
  auto lnonzeros      = ctx.logical_data(&nonzeros[0], nonzeros.size());
  auto lpage_rank     = ctx.logical_data(&page_rank[0], page_rank.size());
  auto lnew_page_rank = ctx.logical_data(&new_page_rank[0], new_page_rank.size());
  auto lmax_diff      = ctx.logical_data(shape_of<scalar_view<float>>());

  for (int iter = 0; iter < NITER; ++iter)
  {
    // Calculate Current Iteration PageRank
    ctx.parallel_for(
      box(num_vertices),
      loffsets.read(),
      lnonzeros.read(),
      lpage_rank.rw(),
      lnew_page_rank.rw(),
      lmax_diff.reduce(reducer::maxval<float>{}))
        ->*[init_rank] __device__(
             size_t idx, auto loffsets, auto lnonzeros, auto lpage_rank, auto lnew_page_rank, auto& max_diff) {
              calculating_pagerank(idx, loffsets, lnonzeros, lpage_rank, lnew_page_rank, init_rank);
              max_diff = ::std::max(max_diff, lnew_page_rank[idx] - lpage_rank[idx]);
            };

    // Reduce Error and Check for Convergence
    bool converged = (ctx.wait(lmax_diff) < tolerance);
    if (converged)
    {
      break;
    }

    // Update New PageRank Values
    std::swap(lpage_rank, lnew_page_rank);
  }

  ctx.finalize();

  /*      CHECKING FOR ANSWER CORRECTNESS      */
  // sum of all page ranks should equal 1
  double sum_pageranks = 0.0;
  for (int64_t i = 0; i < num_vertices; i++)
  {
    sum_pageranks += page_rank[i];
  }
  printf("Page rank answer is %s.\n", abs(sum_pageranks - 1.0) < 0.001 ? "correct" : "not correct");

  printf("PageRank Results:\n");
  for (size_t i = 0; i < page_rank.size(); ++i)
  {
    printf("Vertex %zu: %f\n", i, page_rank[i]);
  }

  return 0;
}
