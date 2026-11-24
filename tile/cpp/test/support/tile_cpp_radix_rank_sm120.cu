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

namespace ct = ::cuda::tiles;

inline constexpr size_t tile_size = 256;

__tile_global__ void radix_rank_kernel(
  const unsigned int* keys_input, int* ascending_output, int* descending_output, int begin_bit, int end_bit)
{
  auto keys_span       = ct::tensor_span{keys_input, ct::shape<tile_size>{}};
  auto ascending_span  = ct::tensor_span{ascending_output, ct::shape<tile_size>{}};
  auto descending_span = ct::tensor_span{descending_output, ct::shape<tile_size>{}};

  auto keys_view       = ct::partition_view{keys_span, ct::shape<tile_size>{}};
  auto ascending_view  = ct::partition_view{ascending_span, ct::shape<tile_size>{}};
  auto descending_view = ct::partition_view{descending_span, ct::shape<tile_size>{}};

  auto keys       = keys_view.load(0);
  auto begin_tile = ct::reshape(ct::tile{begin_bit}, ct::shape<1>{});

  auto ascending =
    ::cuda::coop::tile::radix_rank_keys(keys, begin_tile, end_bit, ::cuda::coop::tile::radix_rank_policy<4>{});
  auto descending = ::cuda::coop::tile::radix_rank_keys(
    keys,
    begin_bit,
    end_bit,
    ::cuda::coop::tile::radix_rank_policy<4,
                                          true,
                                          false,
                                          ::cuda::coop::tile::scan_algorithm::raking,
                                          ::cuda::coop::tile::radix_smem_config::eight_byte>{});

  ascending_view.store(ascending, 0);
  descending_view.store(descending, 0);
}

__tile_global__ void
radix_rank_clamped_kernel(const signed char* keys_input, int* ascending_output, int* descending_output, int begin_bit)
{
  auto keys_span       = ct::tensor_span{keys_input, ct::shape<tile_size>{}};
  auto ascending_span  = ct::tensor_span{ascending_output, ct::shape<tile_size>{}};
  auto descending_span = ct::tensor_span{descending_output, ct::shape<tile_size>{}};

  auto keys_view       = ct::partition_view{keys_span, ct::shape<tile_size>{}};
  auto ascending_view  = ct::partition_view{ascending_span, ct::shape<tile_size>{}};
  auto descending_view = ct::partition_view{descending_span, ct::shape<tile_size>{}};

  auto keys      = keys_view.load(0);
  auto ascending = ::cuda::coop::tile::radix_rank_keys(keys, begin_bit);
  auto descending =
    ::cuda::coop::tile::radix_rank_keys(keys, begin_bit, ::cuda::coop::tile::radix_rank_policy<4, true>{});

  ascending_view.store(ascending, 0);
  descending_view.store(descending, 0);
}

int main()
{
  constexpr size_t key_bytes      = tile_size * sizeof(unsigned int);
  constexpr size_t edge_key_bytes = tile_size * sizeof(signed char);
  constexpr size_t rank_bytes     = tile_size * sizeof(int);
  constexpr int begin_bit         = 4;
  constexpr int end_bit           = 8;
  constexpr int edge_begin_bit    = 7;
  constexpr int digits            = 1 << (end_bit - begin_bit);

  std::vector<unsigned int> keys(tile_size);
  std::vector<signed char> edge_keys(tile_size);
  for (size_t i = 0; i < tile_size; ++i)
  {
    keys[i] = static_cast<unsigned int>(((i * 37 + i / 7) % digits) << begin_bit) | static_cast<unsigned int>(i & 0xfu);
    edge_keys[i] = static_cast<signed char>(static_cast<int>((i * 13 + i / 5) % 256) - 128);
  }

  std::vector<int> ascending_output(tile_size);
  std::vector<int> descending_output(tile_size);
  std::vector<int> edge_ascending_output(tile_size);
  std::vector<int> edge_descending_output(tile_size);

  unsigned int* device_keys     = nullptr;
  signed char* device_edge_keys = nullptr;
  int* device_ascending         = nullptr;
  int* device_descending        = nullptr;
  int* device_edge_ascending    = nullptr;
  int* device_edge_descending   = nullptr;

  if (cudaMalloc(&device_keys, key_bytes) != cudaSuccess || cudaMalloc(&device_edge_keys, edge_key_bytes) != cudaSuccess
      || cudaMalloc(&device_ascending, rank_bytes) != cudaSuccess
      || cudaMalloc(&device_descending, rank_bytes) != cudaSuccess
      || cudaMalloc(&device_edge_ascending, rank_bytes) != cudaSuccess
      || cudaMalloc(&device_edge_descending, rank_bytes) != cudaSuccess)
  {
    std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaGetLastError()) << '\n';
    return EXIT_FAILURE;
  }

  cudaMemcpy(device_keys, keys.data(), key_bytes, cudaMemcpyHostToDevice);
  cudaMemcpy(device_edge_keys, edge_keys.data(), edge_key_bytes, cudaMemcpyHostToDevice);

  radix_rank_kernel<<<1, 1>>>(device_keys, device_ascending, device_descending, begin_bit, end_bit);
  if (const auto launch_status = cudaGetLastError(); launch_status != cudaSuccess)
  {
    std::cerr << "radix_rank_kernel launch failed: " << cudaGetErrorString(launch_status) << '\n';
    return EXIT_FAILURE;
  }
  if (const auto sync_status = cudaDeviceSynchronize(); sync_status != cudaSuccess)
  {
    std::cerr << "radix_rank_kernel execution failed: " << cudaGetErrorString(sync_status) << '\n';
    return EXIT_FAILURE;
  }

  radix_rank_clamped_kernel<<<1, 1>>>(device_edge_keys, device_edge_ascending, device_edge_descending, edge_begin_bit);
  if (const auto launch_status = cudaGetLastError(); launch_status != cudaSuccess)
  {
    std::cerr << "radix_rank_clamped_kernel launch failed: " << cudaGetErrorString(launch_status) << '\n';
    return EXIT_FAILURE;
  }
  if (const auto sync_status = cudaDeviceSynchronize(); sync_status != cudaSuccess)
  {
    std::cerr << "radix_rank_clamped_kernel execution failed: " << cudaGetErrorString(sync_status) << '\n';
    return EXIT_FAILURE;
  }

  cudaMemcpy(ascending_output.data(), device_ascending, rank_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(descending_output.data(), device_descending, rank_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(edge_ascending_output.data(), device_edge_ascending, rank_bytes, cudaMemcpyDeviceToHost);
  cudaMemcpy(edge_descending_output.data(), device_edge_descending, rank_bytes, cudaMemcpyDeviceToHost);

  int counts[digits]{};
  for (const auto key : keys)
  {
    ++counts[(key >> begin_bit) & (digits - 1)];
  }

  int ascending_offsets[digits]{};
  int descending_offsets[digits]{};
  int running = 0;
  for (int digit = 0; digit < digits; ++digit)
  {
    ascending_offsets[digit] = running;
    running += counts[digit];
  }
  running = 0;
  for (int digit = digits - 1; digit >= 0; --digit)
  {
    descending_offsets[digit] = running;
    running += counts[digit];
  }

  int ascending_encounters[digits]{};
  int descending_encounters[digits]{};
  for (size_t i = 0; i < tile_size; ++i)
  {
    const auto digit               = static_cast<int>((keys[i] >> begin_bit) & (digits - 1));
    const auto expected_ascending  = ascending_offsets[digit] + ascending_encounters[digit]++;
    const auto expected_descending = descending_offsets[digit] + descending_encounters[digit]++;
    if (ascending_output[i] != expected_ascending || descending_output[i] != expected_descending)
    {
      std::cerr
        << "Radix Rank mismatch at " << i << ": digit=" << digit << " ascending=" << ascending_output[i]
        << " expected_ascending=" << expected_ascending << " descending=" << descending_output[i]
        << " expected_descending=" << expected_descending << '\n';
      return EXIT_FAILURE;
    }
  }

  int edge_counts[2]{};
  for (const auto key : edge_keys)
  {
    ++edge_counts[key < 0 ? 0 : 1];
  }
  int edge_ascending_encounters[2]{};
  int edge_descending_encounters[2]{};
  for (size_t i = 0; i < tile_size; ++i)
  {
    const auto digit               = (edge_keys[i] < 0 ? 0 : 1);
    const auto expected_ascending  = (digit == 0 ? 0 : edge_counts[0]) + edge_ascending_encounters[digit]++;
    const auto expected_descending = (digit == 1 ? 0 : edge_counts[1]) + edge_descending_encounters[digit]++;
    if (edge_ascending_output[i] != expected_ascending || edge_descending_output[i] != expected_descending)
    {
      std::cerr
        << "Clamped Radix Rank mismatch at " << i << ": digit=" << digit << " ascending=" << edge_ascending_output[i]
        << " expected_ascending=" << expected_ascending << " descending=" << edge_descending_output[i]
        << " expected_descending=" << expected_descending << '\n';
      return EXIT_FAILURE;
    }
  }

  cudaFree(device_edge_descending);
  cudaFree(device_edge_ascending);
  cudaFree(device_descending);
  cudaFree(device_ascending);
  cudaFree(device_edge_keys);
  cudaFree(device_keys);

  std::cout << "PASS: C++ Tile API Radix Rank\n";
  return EXIT_SUCCESS;
}
