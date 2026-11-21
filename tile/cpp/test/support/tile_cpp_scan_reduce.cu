//===----------------------------------------------------------------------===//
//
// Part of the CUDA Core Compute Libraries, under the Apache License v2.0 with
// LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/coop/tile.cuh>

#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

namespace ct           = cuda::tiles;
namespace coop         = cuda::coop::tile;
constexpr size_t count = 256;

__tile_global__ void scan_reduce(const unsigned* input, unsigned* scanned, unsigned* summed, unsigned* reduced)
{
  auto values  = ct::partition_view{ct::tensor_span{input, ct::shape<count>{}}, ct::shape<count>{}}.load(0);
  auto scan    = coop::exclusive_scan(values, cuda::std::plus<>{}, 7u);
  auto sum     = coop::sum(values);
  auto minimum = coop::reduce(values, cuda::minimum<>{});
  ct::partition_view{ct::tensor_span{scanned, ct::shape<count>{}}, ct::shape<count>{}}.store(scan, 0);
  ct::partition_view{ct::tensor_span{summed, ct::shape<count>{}}, ct::shape<count>{}}.store(sum, 0);
  ct::partition_view{ct::tensor_span{reduced, ct::shape<count>{}}, ct::shape<count>{}}.store(minimum, 0);
}

bool check(cudaError_t status)
{
  if (status == cudaSuccess)
  {
    return true;
  }
  std::cerr << cudaGetErrorString(status) << '\n';
  return false;
}

int main()
{
  std::vector<unsigned> input(count), scanned(count), summed(count), reduced(count);
  unsigned total = 0;
  for (size_t index = 0; index < count; ++index)
  {
    input[index] = 0x7fffff80u + static_cast<unsigned>(index);
    total += input[index];
  }
  unsigned *a = nullptr, *b = nullptr, *c = nullptr, *d = nullptr;
  constexpr size_t bytes = count * sizeof(unsigned);
  if (!check(cudaMalloc(&a, bytes)) || !check(cudaMalloc(&b, bytes)) || !check(cudaMalloc(&c, bytes))
      || !check(cudaMalloc(&d, bytes)) || !check(cudaMemcpy(a, input.data(), bytes, cudaMemcpyHostToDevice)))
  {
    return EXIT_FAILURE;
  }
  scan_reduce<<<1, 1>>>(a, b, c, d);
  if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())
      || !check(cudaMemcpy(scanned.data(), b, bytes, cudaMemcpyDeviceToHost))
      || !check(cudaMemcpy(summed.data(), c, bytes, cudaMemcpyDeviceToHost))
      || !check(cudaMemcpy(reduced.data(), d, bytes, cudaMemcpyDeviceToHost)))
  {
    return EXIT_FAILURE;
  }
  unsigned prefix = 7;
  for (size_t index = 0; index < count; ++index)
  {
    if (scanned[index] != prefix || summed[index] != total || reduced[index] != 0x7fffff80u)
    {
      std::cerr << "scan/reduce mismatch at " << index << '\n';
      return EXIT_FAILURE;
    }
    prefix += input[index];
  }
  if (!check(cudaFree(a)) || !check(cudaFree(b)) || !check(cudaFree(c)) || !check(cudaFree(d)))
  {
    return EXIT_FAILURE;
  }
  std::cout << "PASS: unsigned scan, sum, and generic reduction\n";
}
