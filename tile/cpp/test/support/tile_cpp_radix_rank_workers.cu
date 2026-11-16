// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/coop/tile.cuh>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <type_traits>
#include <vector>

#include <cuda_runtime.h>

namespace ct   = cuda::tiles;
namespace coop = cuda::coop::tile;

template <typename T, size_t N>
__tile__ void rank_values(const T* input, int* output, T* preserved, int begin_bit, int end_bit)
{
  const auto values     = ct::partition_view{ct::tensor_span{input, ct::shape<N>{}}, ct::shape<N>{}}.load(0);
  const auto ascending  = coop::radix_rank_keys(values, begin_bit, end_bit);
  const auto descending = coop::radix_rank_keys(values, begin_bit, end_bit, coop::radix_rank_policy<4, true>{});
  auto out              = ct::partition_view{ct::tensor_span{output, ct::shape<2 * N>{}}, ct::shape<N>{}};
  out.store(ascending, 0);
  out.store(descending, 1);
  ct::partition_view{ct::tensor_span{preserved, ct::shape<N>{}}, ct::shape<N>{}}.store(values, 0);
}

template <typename T, size_t N>
__tile_global__ void default_workers(const T* input, int* output, T* preserved, int begin_bit, int end_bit)
{
  rank_values<T, N>(input, output, preserved, begin_bit, end_bit);
}

template <typename T, size_t N>
[[cutile::hint(0, num_worker_warps_per_cta = 4)]]
__tile_global__ void four_worker_warps(const T* input, int* output, T* preserved, int begin_bit, int end_bit)
{
  rank_values<T, N>(input, output, preserved, begin_bit, end_bit);
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
bool run(int begin_bit, int end_bit)
{
  std::vector<T> input(N), preserved(N);
  std::vector<int> output(2 * N);
  for (size_t index = 0; index < N; ++index)
  {
    T magnitude = static_cast<T>((index * 13 + index / 5) % 128);
    if constexpr (sizeof(T) == 8)
    {
      magnitude += static_cast<T>((index * 5 + index / 3) % 8) << 59;
    }
    input[index] = index % 3 == 0 ? std::numeric_limits<T>::lowest() + magnitude : magnitude;
  }
  input[0] = input[4] = std::numeric_limits<T>::lowest();
  input[1] = input[5] = std::numeric_limits<T>::max();
  input[2]            = T{-1};
  input[3]            = T{0};

  T* device_input             = nullptr;
  T* device_preserved         = nullptr;
  int* device_output          = nullptr;
  constexpr size_t key_bytes  = N * sizeof(T);
  constexpr size_t rank_bytes = 2 * N * sizeof(int);
  if (!check(cudaMalloc(&device_input, key_bytes)) || !check(cudaMalloc(&device_preserved, key_bytes))
      || !check(cudaMalloc(&device_output, rank_bytes))
      || !check(cudaMemcpy(device_input, input.data(), key_bytes, cudaMemcpyHostToDevice)))
  {
    return false;
  }
  if constexpr (ExplicitWorkers)
  {
    four_worker_warps<T, N><<<1, 1>>>(device_input, device_output, device_preserved, begin_bit, end_bit);
  }
  else
  {
    default_workers<T, N><<<1, 1>>>(device_input, device_output, device_preserved, begin_bit, end_bit);
  }
  if (!check(cudaGetLastError()) || !check(cudaDeviceSynchronize())
      || !check(cudaMemcpy(output.data(), device_output, rank_bytes, cudaMemcpyDeviceToHost))
      || !check(cudaMemcpy(preserved.data(), device_preserved, key_bytes, cudaMemcpyDeviceToHost)))
  {
    return false;
  }

  using UnsignedT              = std::make_unsigned_t<T>;
  constexpr UnsignedT sign_bit = UnsignedT{1} << (sizeof(T) * 8 - 1);
  const auto mask              = (std::uint64_t{1} << (end_bit - begin_bit)) - 1;
  const auto digit             = [&](T key) {
    return ((static_cast<UnsignedT>(key) ^ sign_bit) >> begin_bit) & mask;
  };
  bool valid = true;
  for (size_t index = 0; index < N; ++index)
  {
    int ascending  = 0;
    int descending = 0;
    for (size_t other = 0; other < N; ++other)
    {
      const bool earlier_equal = digit(input[other]) == digit(input[index]) && other < index;
      ascending += digit(input[other]) < digit(input[index]) || earlier_equal;
      descending += digit(input[other]) > digit(input[index]) || earlier_equal;
    }
    if (output[index] != ascending || output[N + index] != descending || preserved[index] != input[index])
    {
      std::cerr << "radix rank mismatch: extent " << N << ", element bytes " << sizeof(T) << ", explicit workers "
                << ExplicitWorkers << ", bits [" << begin_bit << ", " << end_bit << "), index " << index << '\n';
      valid = false;
    }
  }
  return check(cudaFree(device_input)) && check(cudaFree(device_output)) && check(cudaFree(device_preserved)) && valid;
}

int main()
{
  if (!run<std::int8_t, 32, false>(7, 8) || !run<std::int64_t, 64, false>(60, 64) || !run<std::int64_t, 32, false>(0, 4)
      || !run<std::int64_t, 32, true>(60, 64) || !run<std::int8_t, 64, true>(7, 8)
      || !run<std::int64_t, 64, true>(0, 4))
  {
    return EXIT_FAILURE;
  }
  std::cout << "PASS: small Radix Rank with default and four worker warps\n";
}
