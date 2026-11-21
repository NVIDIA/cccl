// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/coop/tile.cuh>

#include <cstdlib>
#include <iostream>
#include <numeric>

#include <cuda_runtime.h>

namespace ct   = cuda::tiles;
namespace coop = cuda::coop::tile;

constexpr int tile_items = 256;
constexpr int tile_count = 2;
constexpr int size       = tile_count * tile_items;

__tile_global__ void block_scan(int* data)
{
  auto span        = ct::tensor_span{data, ct::shape<size>{}};
  auto tiles       = ct::partition_view{span, ct::shape<tile_items>{}};
  const auto index = ct::bid().x;
  const auto items = tiles.load(index);
  tiles.store(coop::exclusive_sum(items), index);
}

bool check(cudaError_t status)
{
  if (status != cudaSuccess)
  {
    std::cerr << cudaGetErrorString(status) << '\n';
    return false;
  }
  return true;
}

int main()
{
  int* data = nullptr;
  if (!check(cudaMallocManaged(&data, size * sizeof(int))))
  {
    return EXIT_FAILURE;
  }
  std::iota(data, data + size, 0);
  block_scan<<<tile_count, 1>>>(data);
  bool valid = check(cudaGetLastError()) && check(cudaDeviceSynchronize());
  if (valid)
  {
    for (int tile = 0; tile < tile_count; ++tile)
    {
      int expected = 0;
      for (int i = 0; i < tile_items; ++i)
      {
        const int index = tile * tile_items + i;
        if (data[index] != expected)
        {
          std::cerr << "Prefix at " << index << ": expected " << expected << ", got " << data[index] << '\n';
          valid = false;
          break;
        }
        expected += index;
      }
    }
  }
  if (valid)
  {
    std::cout << "PASS: both tiles matched all 256 prefix sums\n";
  }
  return check(cudaFree(data)) && valid ? EXIT_SUCCESS : EXIT_FAILURE;
}
