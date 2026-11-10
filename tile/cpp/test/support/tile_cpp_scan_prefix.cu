// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/coop/tile.cuh>

#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

namespace ct   = cuda::tiles;
namespace coop = cuda::coop::tile;

inline constexpr size_t cases = 8;

template <size_t N>
__tile_global__ void scan_prefix(int* input, const int* prefix, int scalar_seed, int* output)
{
  const auto values = ct::partition_view{ct::tensor_span{input, ct::shape<N>{}}, ct::shape<N>{}}.load(0);
  const auto seed   = ct::partition_view{ct::tensor_span{prefix, ct::shape<1>{}}, ct::shape<1>{}}.load(0);
  auto out          = ct::partition_view{ct::tensor_span{output, ct::shape<cases * N>{}}, ct::shape<N>{}};
  out.store(coop::exclusive_sum(values, scalar_seed), 0);
  out.store(coop::inclusive_sum(values, seed, coop::scan_policy<coop::scan_algorithm::raking_memoize>{}), 1);
  out.store(coop::inclusive_scan(values, cuda::maximum<>{}, scalar_seed), 2);
  out.store(coop::inclusive_scan(values, seed), 3);
  out.store(coop::scan<coop::scan_mode::inclusive>(values, cuda::std::bit_xor<>{}, seed), 4);
  out.store(
    coop::scan<coop::scan_mode::inclusive>(values, scalar_seed, coop::scan_policy<coop::scan_algorithm::warp_scans>{}),
    5);
  out.store(coop::scan(values, cuda::maximum<>{}, seed), 6);
  out.store(coop::scan(values, scalar_seed), 7);
  ct::partition_view{ct::tensor_span{input, ct::shape<N>{}}, ct::shape<N>{}}.store(values, 0);
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

template <size_t N>
bool run(int prefix)
{
  std::vector<int> input(N), output(cases * N), preserved(N);
  for (size_t index = 0; index < N; ++index)
  {
    input[index] = static_cast<int>((index * 7) % 11) - 5;
  }
  int *device_input = nullptr, *device_prefix = nullptr, *device_output = nullptr;
  if (!check(cudaMalloc(&device_input, N * sizeof(int))) || !check(cudaMalloc(&device_prefix, sizeof(int)))
      || !check(cudaMalloc(&device_output, cases * N * sizeof(int)))
      || !check(cudaMemcpy(device_input, input.data(), N * sizeof(int), cudaMemcpyHostToDevice))
      || !check(cudaMemcpy(device_prefix, &prefix, sizeof(int), cudaMemcpyHostToDevice)))
  {
    return false;
  }
  scan_prefix<N><<<1, 1>>>(device_input, device_prefix, prefix, device_output);
  if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())
      || !check(cudaMemcpy(output.data(), device_output, cases * N * sizeof(int), cudaMemcpyDeviceToHost))
      || !check(cudaMemcpy(preserved.data(), device_input, N * sizeof(int), cudaMemcpyDeviceToHost)))
  {
    return false;
  }
  int sum     = prefix;
  int maximum = prefix;
  int bits    = prefix;
  bool valid  = preserved == input;
  for (size_t index = 0; index < N; ++index)
  {
    const int before         = sum;
    const int before_maximum = maximum;
    sum += input[index];
    maximum = maximum < input[index] ? input[index] : maximum;
    bits ^= input[index];
    const int expected[cases] = {before, sum, maximum, sum, bits, sum, before_maximum, before};
    for (size_t operation = 0; operation < cases; ++operation)
    {
      if (output[operation * N + index] != expected[operation])
      {
        std::cerr << "scan prefix mismatch: extent " << N << ", prefix " << prefix << ", operation " << operation
                  << ", index " << index << '\n';
        valid = false;
      }
    }
  }
  return check(cudaFree(device_input)) && check(cudaFree(device_prefix)) && check(cudaFree(device_output)) && valid;
}

int main()
{
  if (!run<128>(-7) || !run<128>(19) || !run<512>(-7) || !run<512>(19))
  {
    return EXIT_FAILURE;
  }
  std::cout << "PASS: scalar and Tile scan prefixes at extents 128 and 512\n";
}
