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
 * @brief Computes the total number of triangles within a graph
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
 * @brief Computes the Triangle Counting for each vertex.
 *
 * @param idx        The index of the vertex for which Triangle Counting is being calculated.
 * @param loffsets  Slice containing the offset vector of the CSR representation.
 * @param lnonzeros Slice containing the non-zero elements (neighbors) vector of the CSR representation.
 * @return           The local triangle count for the vertex.
 */
__device__ unsigned long long int triangle_count(int idx, slice<const int> loffsets, slice<const int> lnonzeros)
{
  int lcount = 0;
  for (int i = loffsets[idx]; i < loffsets[idx + 1]; i++)
  {
    int v = lnonzeros[i];
    for (int j = loffsets[idx]; j < loffsets[idx + 1]; j++)
    {
      int w = lnonzeros[j];
      if (binary_search(lnonzeros, loffsets[v], loffsets[v + 1] - 1, w) != -1)
      {
        lcount++;
      }
    }
  }
  return lcount;
}

int main()
{
  stream_ctx ctx;

  // row offsets in CSR format
  std::vector<int> offsets = {0, 0, 1, 2, 4, 5, 6, 8, 9, 10};
  // edges in CSR format
  std::vector<int> nonzeros = {0, 0, 0, 1, 1, 1, 0, 1, 1, 1};

  int num_vertices                   = offsets.size() - 1;
  unsigned long long int total_count = 0;

  auto loffsets     = ctx.logical_data(&offsets[0], offsets.size());
  auto lnonzeros    = ctx.logical_data(&nonzeros[0], nonzeros.size());
  auto ltotal_count = ctx.logical_data(&total_count, {1});

  ctx.parallel_for(box(num_vertices), loffsets.read(), lnonzeros.read(), ltotal_count.rw())
      ->*[] __device__(size_t idx, auto loffsets, auto lnonzeros, auto ltotal_count) {
            unsigned long long int count = triangle_count(idx, loffsets, lnonzeros);
            atomicAdd(ltotal_count.data_handle(), count);
          };

  ctx.finalize();

  printf("Number of triangles: %lld\n", total_count);

  return 0;
}
