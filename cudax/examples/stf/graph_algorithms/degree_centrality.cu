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
 * @brief Computes the Degree Centrality for each vertex within a graph
 *
 */

#include <cuda/experimental/stf.cuh>

#include <vector>

using namespace cuda::experimental::stf;

/**
 * @brief Computes the Degree Centrality for each vertex.
 *
 * @param idx        The index of the vertex for which Degree Centrality is being calculated.
 * @param d_offsets  Slice containing the offset vector of the CSR representation.
 * @return           The degree of each vertex.
 */
__device__ int degree_centrality(int idx, slice<const int> loffsets)
{
  return loffsets[idx + 1] - loffsets[idx];
}

int main()
{
  stream_ctx ctx;

  // row offsets in CSR format
  std::vector<int> offsets = {0, 4, 11, 12, 14, 15, 16, 18, 19, 20};
  // edges in CSR format
  std::vector<int> nonzeros = {1, 2, 3, 6, 0, 3, 4, 5, 6, 7, 8, 0, 0, 1, 1, 1, 0, 1, 1, 1};
  // output degrees for each vertex
  int num_vertices = offsets.size() - 1;
  std::vector<int> degrees(num_vertices, 0);

  auto loffsets  = ctx.logical_data(&offsets[0], offsets.size());
  auto lnonzeros = ctx.logical_data(&nonzeros[0], nonzeros.size());
  auto ldegrees  = ctx.logical_data(&degrees[0], degrees.size());

  ctx.parallel_for(box(num_vertices), loffsets.read(), ldegrees.rw())
      ->*[] __device__(size_t idx, auto loffsets, auto ldegrees) {
            ldegrees[idx] = degree_centrality(idx, loffsets);
          };

  ctx.finalize();

  // for (int i = 0; i < num_vertices; ++i) {
  //     printf("Vertex %d: Degree Centrality = %d\n", i, degrees[i]);
  // }

  return 0;
}
