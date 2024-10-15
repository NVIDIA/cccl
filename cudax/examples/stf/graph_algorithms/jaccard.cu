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
 * @brief Computes the Jaccard Similarity for each vertex within a graph
 *
 */

#include <cuda/experimental/stf.cuh>

#include <vector>

using namespace cuda::experimental::stf;

// Performs Binary Search on a given array with start/end bounds and a lookup element
__device__ int binary_search(slice<const int> arr, int start, int end, int lookup)
{
  while (start <= end)
  {
    int mid = start + (end - start) / 2;
    if (arr[mid] == lookup)
    {
      return mid;
    }
    else if (arr[mid] < lookup)
    {
      start = mid + 1;
    }
    else
    {
      end = mid - 1;
    }
  }
  return -1;
}

/**
 * @brief Computes the intersection size of neighbors of two vertices.
 *
 * @param loffsets  Slice containing the offset vector of the CSR representation.
 * @param lnonzeros Slice containing the non-zero elements (neighbors) vector of the CSR representation.
 * @param u          Index of the first vertex.
 * @param v          Index of the second vertex.
 * @return The number of common neighbors (intersection size) of vertices u and v.
 */
__device__ int calculate_intersection_size(slice<const int> loffsets, slice<const int> lnonzeros, int u, int v)
{
  int count = 0;
  for (int i = loffsets[u]; i < loffsets[u + 1]; i++)
  {
    if (binary_search(lnonzeros, loffsets[v], loffsets[v + 1] - 1, lnonzeros[i]) != -1)
    {
      count++;
    }
  }
  return count;
}

/**
 * @brief Computes the union size of neighbors of two vertices.
 *
 * @param loffsets  Slice containing the offset vector of the CSR representation.
 * @param lnonzeros Slice containing the non-zero elements (neighbors) vector of the CSR representation.
 * @param u          Index of the first vertex.
 * @param v          Index of the second vertex.
 * @return The number of unique neighbors (union size) of vertices u and v.
 */
__device__ int calculate_union_size(slice<const int> loffsets, slice<const int> lnonzeros, int u, int v)
{
  int count = (loffsets[u + 1] - loffsets[u]) + (loffsets[v + 1] - loffsets[v]);
  for (int i = loffsets[u]; i < loffsets[u + 1]; i++)
  {
    if (binary_search(lnonzeros, loffsets[v], loffsets[v + 1] - 1, lnonzeros[i]) != -1)
    {
      count--;
    }
  }
  return count;
}

int main()
{
  stream_ctx ctx;

  // row offsets in CSR format
  std::vector<int> offsets = {0, 4, 11, 12, 14, 15, 16, 18, 19, 20};
  // edges in CSR format
  std::vector<int> nonzeros = {1, 2, 3, 6, 0, 3, 4, 5, 6, 7, 8, 0, 0, 1, 1, 1, 0, 1, 1, 1};
  // output jaccard similarities for each vertex
  int num_vertices = offsets.size() - 1;
  std::vector<float> jaccard_similarities(num_vertices * num_vertices, 0.0f);

  auto loffsets              = ctx.logical_data(&offsets[0], offsets.size());
  auto lnonzeros             = ctx.logical_data(&nonzeros[0], nonzeros.size());
  auto ljaccard_similarities = ctx.logical_data(&jaccard_similarities[0], jaccard_similarities.size());

  ctx.parallel_for(box(num_vertices), loffsets.read(), lnonzeros.read(), ljaccard_similarities.rw())
      ->*[] __device__(size_t idx, auto loffsets, auto lnonzeros, auto ljaccard_similarities) {
            for (int j = 0; j < loffsets.size() - 1; j++)
            {
              if (idx != j)
              {
                int intersection = calculate_intersection_size(loffsets, lnonzeros, idx, j);
                int uni          = calculate_union_size(loffsets, lnonzeros, idx, j);
                if (uni > 0)
                {
                  ljaccard_similarities[idx * (loffsets.size() - 1) + j] = static_cast<float>(intersection) / uni;
                }
              }
            }
          };

  ctx.finalize();

  for (int u = 0; u < num_vertices; u++)
  {
    for (int v = 0; v < num_vertices; v++)
    {
      if (u != v)
      {
        printf(
          "Jaccard similarity between vertex %d and vertex %d: %f\n", u, v, jaccard_similarities[u * num_vertices + v]);
      }
    }
  }

  return 0;
}
