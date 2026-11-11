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

#include <algorithm>
#include <array>
#include <cstdlib>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

namespace ct = ::cuda::tiles;

inline constexpr size_t tile_size = 256;

__tile_global__ void radix_sort_kernel(
  const unsigned int* keys_input,
  const unsigned long long* values_input,
  unsigned int* full_keys_output,
  unsigned int* scalar_bits_output,
  unsigned int* tile_bits_output,
  unsigned int* pair_keys_output,
  unsigned long long* pair_values_output,
  int begin_bit,
  int end_bit)
{
  auto keys_span        = ct::tensor_span{keys_input, ct::shape<tile_size>{}};
  auto values_span      = ct::tensor_span{values_input, ct::shape<tile_size>{}};
  auto full_keys_span   = ct::tensor_span{full_keys_output, ct::shape<tile_size>{}};
  auto scalar_bits_span = ct::tensor_span{scalar_bits_output, ct::shape<tile_size>{}};
  auto tile_bits_span   = ct::tensor_span{tile_bits_output, ct::shape<tile_size>{}};
  auto pair_keys_span   = ct::tensor_span{pair_keys_output, ct::shape<tile_size>{}};
  auto pair_values_span = ct::tensor_span{pair_values_output, ct::shape<tile_size>{}};

  auto keys_view        = ct::partition_view{keys_span, ct::shape<tile_size>{}};
  auto values_view      = ct::partition_view{values_span, ct::shape<tile_size>{}};
  auto full_keys_view   = ct::partition_view{full_keys_span, ct::shape<tile_size>{}};
  auto scalar_bits_view = ct::partition_view{scalar_bits_span, ct::shape<tile_size>{}};
  auto tile_bits_view   = ct::partition_view{tile_bits_span, ct::shape<tile_size>{}};
  auto pair_keys_view   = ct::partition_view{pair_keys_span, ct::shape<tile_size>{}};
  auto pair_values_view = ct::partition_view{pair_values_span, ct::shape<tile_size>{}};

  auto keys       = keys_view.load(0);
  auto values     = values_view.load(0);
  auto begin_tile = ct::reshape(ct::tile{begin_bit}, ct::shape<1>{});
  auto end_tile   = ct::reshape(ct::tile{end_bit}, ct::shape<1>{});

  auto full_keys                = ::cuda::coop::tile::radix_sort_keys(keys);
  auto scalar_bits              = ::cuda::coop::tile::radix_sort_keys(keys, begin_bit, end_bit);
  auto tile_bits                = ::cuda::coop::tile::radix_sort_keys(keys, begin_tile, end_tile);
  auto [pair_keys, pair_values] = ::cuda::coop::tile::radix_sort_pairs(
    keys,
    values,
    ::cuda::std::greater<>{},
    begin_tile,
    32,
    ::cuda::coop::tile::radix_sort_policy<5,
                                          false,
                                          ::cuda::coop::tile::scan_algorithm::raking,
                                          ::cuda::coop::tile::radix_smem_config::eight_byte>{});

  full_keys_view.store(full_keys, 0);
  scalar_bits_view.store(scalar_bits, 0);
  tile_bits_view.store(tile_bits, 0);
  pair_keys_view.store(pair_keys, 0);
  pair_values_view.store(pair_values, 0);
}

__tile_global__ void floating_radix_sort_kernel(
  const float* keys_input, const unsigned long long* values_input, float* keys_output, unsigned long long* values_output)
{
  auto keys_span    = ct::tensor_span{keys_input, ct::shape<tile_size>{}};
  auto values_span  = ct::tensor_span{values_input, ct::shape<tile_size>{}};
  auto output_span  = ct::tensor_span{keys_output, ct::shape<tile_size>{}};
  auto payload_span = ct::tensor_span{values_output, ct::shape<tile_size>{}};
  auto keys_view    = ct::partition_view{keys_span, ct::shape<tile_size>{}};
  auto values_view  = ct::partition_view{values_span, ct::shape<tile_size>{}};
  auto output_view  = ct::partition_view{output_span, ct::shape<tile_size>{}};
  auto payload_view = ct::partition_view{payload_span, ct::shape<tile_size>{}};
  auto [keys, values] =
    ::cuda::coop::tile::radix_sort_pairs(keys_view.load(0), values_view.load(0), ::cuda::std::greater<>{});
  output_view.store(keys, 0);
  payload_view.store(values, 0);
}

int main()
{
  constexpr size_t key_bytes   = tile_size * sizeof(unsigned int);
  constexpr size_t value_bytes = tile_size * sizeof(unsigned long long);
  constexpr int begin_bit      = 0;
  constexpr int end_bit        = 8;

  std::vector<unsigned int> keys(tile_size);
  std::vector<unsigned long long> values(tile_size);
  for (size_t i = 0; i < tile_size; ++i)
  {
    keys[i]   = static_cast<unsigned int>((i * 73 + 19) % tile_size) | (static_cast<unsigned int>(i % 7) << 16);
    values[i] = static_cast<unsigned long long>(i);
  }

  std::vector<unsigned int> full_keys_output(tile_size);
  std::vector<unsigned int> scalar_bits_output(tile_size);
  std::vector<unsigned int> tile_bits_output(tile_size);
  std::vector<unsigned int> pair_keys_output(tile_size);
  std::vector<unsigned long long> pair_values_output(tile_size);

  unsigned int* device_keys                     = nullptr;
  unsigned long long* device_values             = nullptr;
  unsigned int* device_full_keys_output         = nullptr;
  unsigned int* device_scalar_bits_output       = nullptr;
  unsigned int* device_tile_bits_output         = nullptr;
  unsigned int* device_pair_keys_output         = nullptr;
  unsigned long long* device_pair_values_output = nullptr;

  if (cudaMalloc(&device_keys, key_bytes) != cudaSuccess || cudaMalloc(&device_values, value_bytes) != cudaSuccess
      || cudaMalloc(&device_full_keys_output, key_bytes) != cudaSuccess
      || cudaMalloc(&device_scalar_bits_output, key_bytes) != cudaSuccess
      || cudaMalloc(&device_tile_bits_output, key_bytes) != cudaSuccess
      || cudaMalloc(&device_pair_keys_output, key_bytes) != cudaSuccess
      || cudaMalloc(&device_pair_values_output, value_bytes) != cudaSuccess)
  {
    std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaGetLastError()) << '\n';
    return EXIT_FAILURE;
  }

  cudaMemcpy(device_keys, keys.data(), key_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_values, values.data(), value_bytes, cudaMemcpyHostToDevice);

  radix_sort_kernel<<<1, 1>>>(
    device_keys,
    device_values,
    device_full_keys_output,
    device_scalar_bits_output,
    device_tile_bits_output,
    device_pair_keys_output,
    device_pair_values_output,
    begin_bit,
    end_bit);
  if (const auto launch_status = cudaGetLastError(); launch_status != cudaSuccess)
  {
    std::cerr << "radix_sort_kernel launch failed: " << cudaGetErrorString(launch_status) << '\n';
    return EXIT_FAILURE;
  }
  if (const auto sync_status = cudaDeviceSynchronize(); sync_status != cudaSuccess)
  {
    std::cerr << "radix_sort_kernel execution failed: " << cudaGetErrorString(sync_status) << '\n';
    return EXIT_FAILURE;
  }

  cudaMemcpy(full_keys_output.data(), device_full_keys_output, key_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(scalar_bits_output.data(), device_scalar_bits_output, key_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(tile_bits_output.data(), device_tile_bits_output, key_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(pair_keys_output.data(), device_pair_keys_output, key_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(pair_values_output.data(), device_pair_values_output, value_bytes, cudaMemcpyDeviceToHost);

  auto expected_full = keys;
  std::stable_sort(expected_full.begin(), expected_full.end());
  auto expected_bits = keys;
  std::stable_sort(expected_bits.begin(), expected_bits.end(), [](unsigned int lhs, unsigned int rhs) {
    return (lhs & 0xffu) < (rhs & 0xffu);
  });
  auto expected_pair_order = values;
  std::stable_sort(
    expected_pair_order.begin(), expected_pair_order.end(), [&](unsigned long long lhs, unsigned long long rhs) {
      return keys[lhs] > keys[rhs];
    });

  for (size_t i = 0; i < tile_size; ++i)
  {
    const auto source_index = pair_values_output[i];
    if (full_keys_output[i] != expected_full[i] || scalar_bits_output[i] != expected_bits[i]
        || tile_bits_output[i] != expected_bits[i] || scalar_bits_output[i] != tile_bits_output[i]
        || source_index != expected_pair_order[i] || source_index >= tile_size
        || pair_keys_output[i] != keys[source_index])
    {
      std::cerr
        << "Radix Sort mismatch at " << i << ": full=" << full_keys_output[i] << " expected_full=" << expected_full[i]
        << " scalar_bits=" << scalar_bits_output[i] << " tile_bits=" << tile_bits_output[i] << " expected_bits="
        << expected_bits[i] << " pair_key=" << pair_keys_output[i] << " pair_source=" << source_index << '\n';
      return EXIT_FAILURE;
    }
  }

  // Copy raw float bits so NaN payloads and both zero signs survive unchanged.
  // CUB radix order puts negative NaNs before -inf and positive NaNs after
  // +inf. Both zero signs belong to one stable equivalence class.
  constexpr std::array<unsigned int, 8> patterns = {
    0xff800001u, 0xff800000u, 0xbf800000u, 0x80000000u, 0u, 0x3f800000u, 0x7f800000u, 0x7f800001u};
  constexpr std::array<int, 8> order = {0, 1, 2, 3, 3, 4, 5, 6};
  std::vector<int> ranks(tile_size);
  for (size_t i = 0; i < tile_size; ++i)
  {
    const auto pattern = (i * 5 + i / 8) % patterns.size();
    keys[i]            = patterns[pattern];
    ranks[i]           = order[pattern];
  }
  cudaMemcpy(device_keys, keys.data(), key_bytes, cudaMemcpyHostToDevice);
  floating_radix_sort_kernel<<<1, 1>>>(
    reinterpret_cast<const float*>(device_keys),
    device_values,
    reinterpret_cast<float*>(device_pair_keys_output),
    device_pair_values_output);
  if (const auto launch_status = cudaGetLastError(); launch_status != cudaSuccess)
  {
    std::cerr << "floating_radix_sort_kernel launch failed: " << cudaGetErrorString(launch_status) << '\n';
    return EXIT_FAILURE;
  }
  if (const auto sync_status = cudaDeviceSynchronize(); sync_status != cudaSuccess)
  {
    std::cerr << "floating_radix_sort_kernel execution failed: " << cudaGetErrorString(sync_status) << '\n';
    return EXIT_FAILURE;
  }
  cudaMemcpy(pair_keys_output.data(), device_pair_keys_output, key_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(pair_values_output.data(), device_pair_values_output, value_bytes, cudaMemcpyDeviceToHost);
  expected_pair_order = values;
  std::stable_sort(
    expected_pair_order.begin(), expected_pair_order.end(), [&](unsigned long long lhs, unsigned long long rhs) {
      return ranks[lhs] > ranks[rhs];
    });
  for (size_t i = 0; i < tile_size; ++i)
  {
    const auto expected_source = expected_pair_order[i];
    if (pair_values_output[i] != expected_source || pair_keys_output[i] != keys[expected_source])
    {
      std::cerr << "Floating Radix Sort lost key bits or stable pair order at " << i << '\n';
      return EXIT_FAILURE;
    }
  }

  cudaFree(device_pair_values_output);
  cudaFree(device_pair_keys_output);
  cudaFree(device_tile_bits_output);
  cudaFree(device_scalar_bits_output);
  cudaFree(device_full_keys_output);
  cudaFree(device_values);
  cudaFree(device_keys);

  std::cout << "PASS: C++ Tile API Radix Sort keys/pairs\n";
  return EXIT_SUCCESS;
}
