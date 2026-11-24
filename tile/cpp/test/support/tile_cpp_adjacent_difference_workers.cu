// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/coop/tile.cuh>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

namespace ct   = cuda::tiles;
namespace coop = cuda::coop::tile;

template <typename T, size_t N>
__tile__ void adjacent_difference_values(const T* input, T* output)
{
  const auto values = ct::partition_view{ct::tensor_span{input, ct::shape<N>{}}, ct::shape<N>{}}.load(0);
  const auto left   = coop::adjacent_difference_subtract_left(values);
  const auto right  = coop::adjacent_difference_subtract_right(values);
  auto out          = ct::partition_view{ct::tensor_span{output, ct::shape<3 * N>{}}, ct::shape<N>{}};
  out.store(left, 0);
  out.store(right, 1);
  out.store(values, 2);
}

template <typename T, size_t N>
__tile_global__ void default_workers(const T* input, T* output)
{
  adjacent_difference_values<T, N>(input, output);
}

template <typename T, size_t N>
[[cutile::hint(0, num_worker_warps_per_cta = 4)]]
__tile_global__ void four_worker_warps(const T* input, T* output)
{
  adjacent_difference_values<T, N>(input, output);
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

template <typename T, size_t N, bool ExplicitWorkers>
bool run()
{
  std::vector<T> input(N), output(3 * N);
  for (size_t index = 0; index < N; ++index)
  {
    input[index] = static_cast<T>((index * 17) % 23) - T{11};
    if constexpr (sizeof(T) > 4)
    {
      input[index] += static_cast<T>(index % 5) * (T{1} << 35);
    }
  }
  T* device_input               = nullptr;
  T* device_output              = nullptr;
  constexpr size_t input_bytes  = N * sizeof(T);
  constexpr size_t output_bytes = 3 * input_bytes;
  if (!check(cudaMalloc(&device_input, input_bytes)) || !check(cudaMalloc(&device_output, output_bytes))
      || !check(cudaMemcpy(device_input, input.data(), input_bytes, cudaMemcpyHostToDevice)))
  {
    return false;
  }
  if constexpr (ExplicitWorkers)
  {
    four_worker_warps<T, N><<<1, 1>>>(device_input, device_output);
  }
  else
  {
    default_workers<T, N><<<1, 1>>>(device_input, device_output);
  }
  if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())
      || !check(cudaMemcpy(output.data(), device_output, output_bytes, cudaMemcpyDeviceToHost)))
  {
    return false;
  }
  bool valid = true;
  for (size_t index = 0; index < N; ++index)
  {
    const T left  = index == 0 ? input[index] : input[index] - input[index - 1];
    const T right = index + 1 == N ? input[index] : input[index] - input[index + 1];
    if (output[index] != left || output[N + index] != right || output[2 * N + index] != input[index])
    {
      std::cerr << "adjacent difference mismatch: extent " << N << ", element bytes " << sizeof(T)
                << ", explicit workers " << ExplicitWorkers << ", index " << index << '\n';
      valid = false;
    }
  }
  return check(cudaFree(device_input)) && check(cudaFree(device_output)) && valid;
}

int main()
{
  if (!run<std::int32_t, 32, false>() || !run<std::int32_t, 64, false>() || !run<std::int64_t, 32, false>()
      || !run<std::int64_t, 64, false>() || !run<std::int32_t, 32, true>() || !run<std::int32_t, 64, true>()
      || !run<std::int64_t, 32, true>() || !run<std::int64_t, 64, true>())
  {
    return EXIT_FAILURE;
  }
  std::cout << "PASS: small Adjacent Difference with default and four worker warps\n";
}
