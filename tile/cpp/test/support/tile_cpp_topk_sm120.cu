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
#include <cstddef>
#include <cstdlib>
#include <initializer_list>
#include <iostream>
#include <vector>

#include <cuda_runtime.h>

namespace ct   = ::cuda::tiles;
namespace coop = ::cuda::coop::tile;

inline constexpr size_t tile_size = 256;
inline constexpr size_t outputs   = 6;

template <size_t BlockThreads, bool Partial>
[[cutile::hint(0, num_worker_warps_per_cta = BlockThreads / 32)]]
__tile_global__ void topk_kernel(
  const unsigned int* input_keys,
  const unsigned int* input_values,
  unsigned int* output,
  int runtime_k,
  int runtime_num_valid,
  int runtime_begin_bit,
  int runtime_end_bit)
{
  auto keys_span       = ct::tensor_span{input_keys, ct::shape<tile_size>{}};
  auto values_span     = ct::tensor_span{input_values, ct::shape<tile_size>{}};
  auto max_keys_span   = ct::tensor_span{output, ct::shape<tile_size>{}};
  auto min_keys_span   = ct::tensor_span{output + tile_size, ct::shape<tile_size>{}};
  auto max_pair_k_span = ct::tensor_span{output + 2 * tile_size, ct::shape<tile_size>{}};
  auto max_pair_v_span = ct::tensor_span{output + 3 * tile_size, ct::shape<tile_size>{}};
  auto min_pair_k_span = ct::tensor_span{output + 4 * tile_size, ct::shape<tile_size>{}};
  auto min_pair_v_span = ct::tensor_span{output + 5 * tile_size, ct::shape<tile_size>{}};

  auto keys_view       = ct::partition_view{keys_span, ct::shape<tile_size>{}};
  auto values_view     = ct::partition_view{values_span, ct::shape<tile_size>{}};
  auto max_keys_view   = ct::partition_view{max_keys_span, ct::shape<tile_size>{}};
  auto min_keys_view   = ct::partition_view{min_keys_span, ct::shape<tile_size>{}};
  auto max_pair_k_view = ct::partition_view{max_pair_k_span, ct::shape<tile_size>{}};
  auto max_pair_v_view = ct::partition_view{max_pair_v_span, ct::shape<tile_size>{}};
  auto min_pair_k_view = ct::partition_view{min_pair_k_span, ct::shape<tile_size>{}};
  auto min_pair_v_view = ct::partition_view{min_pair_v_span, ct::shape<tile_size>{}};

  auto keys   = keys_view.load(0);
  auto values = values_view.load(0);
  if constexpr (Partial)
  {
    auto max_keys = coop::topk_max_keys(
      keys, runtime_k, runtime_num_valid, runtime_begin_bit, runtime_end_bit, coop::topk_policy<>{});
    auto min_keys = coop::topk_min_keys(
      keys, runtime_k, runtime_num_valid, runtime_begin_bit, runtime_end_bit, coop::topk_policy<>{});
    auto [max_pair_keys, max_pair_values] = coop::topk_max_pairs(
      keys, values, runtime_k, runtime_num_valid, runtime_begin_bit, runtime_end_bit, coop::topk_policy<>{});
    auto [min_pair_keys, min_pair_values] = coop::topk_min_pairs(
      keys, values, runtime_k, runtime_num_valid, runtime_begin_bit, runtime_end_bit, coop::topk_policy<>{});
    max_keys_view.store(max_keys, 0);
    min_keys_view.store(min_keys, 0);
    max_pair_k_view.store(max_pair_keys, 0);
    max_pair_v_view.store(max_pair_values, 0);
    min_pair_k_view.store(min_pair_keys, 0);
    min_pair_v_view.store(min_pair_values, 0);
  }
  else
  {
    auto max_keys = coop::topk_max_keys(keys, runtime_k, runtime_begin_bit, runtime_end_bit, coop::topk_policy<>{});
    auto min_keys = coop::topk_min_keys(keys, runtime_k, runtime_begin_bit, runtime_end_bit, coop::topk_policy<>{});
    auto [max_pair_keys, max_pair_values] =
      coop::topk_max_pairs(keys, values, runtime_k, runtime_begin_bit, runtime_end_bit, coop::topk_policy<>{});
    auto [min_pair_keys, min_pair_values] =
      coop::topk_min_pairs(keys, values, runtime_k, runtime_begin_bit, runtime_end_bit, coop::topk_policy<>{});
    max_keys_view.store(max_keys, 0);
    min_keys_view.store(min_keys, 0);
    max_pair_k_view.store(max_pair_keys, 0);
    max_pair_v_view.store(max_pair_values, 0);
    min_pair_k_view.store(min_pair_keys, 0);
    min_pair_v_view.store(min_pair_values, 0);
  }
}

bool check_cuda(cudaError_t status, const char* operation)
{
  if (status == cudaSuccess)
  {
    return true;
  }
  std::cerr << operation << " failed: " << cudaGetErrorString(status) << '\n';
  return false;
}

bool validate_result(
  const std::vector<unsigned int>& keys,
  const std::vector<unsigned int>& values,
  const std::vector<unsigned int>& output,
  int runtime_k,
  int num_valid,
  int begin_bit,
  int end_bit)
{
  runtime_k       = std::min(runtime_k, num_valid);
  const auto mask = (1u << (end_bit - begin_bit)) - 1u;
  auto projected  = [&](unsigned int key) {
    return (key >> begin_bit) & mask;
  };
  std::vector<unsigned int> candidates(keys.begin(), keys.begin() + num_valid);
  std::sort(candidates.begin(), candidates.end(), [&](unsigned int a, unsigned int b) {
    return projected(a) < projected(b);
  });

  // Only min(requested k, valid count) entries are defined; ordering is unspecified.
  for (const size_t slot : {0u, 1u, 2u, 4u})
  {
    const bool descending = slot == 0 || slot == 2;
    auto first            = descending ? candidates.end() - runtime_k : candidates.begin();
    std::vector<unsigned int> expected(first, first + runtime_k);
    std::vector<unsigned int> actual(output.begin() + slot * tile_size, output.begin() + slot * tile_size + runtime_k);
    std::sort(expected.begin(), expected.end());
    std::sort(actual.begin(), actual.end());
    if (actual != expected)
    {
      std::cerr << "TopK selected-key set mismatch for output " << slot << " num_valid=" << num_valid << '\n';
      return false;
    }
    if (slot == 2 || slot == 4)
    {
      for (int item = 0; item < runtime_k; ++item)
      {
        const auto key   = output[slot * tile_size + item];
        const auto value = output[(slot + 1) * tile_size + item];
        const auto found = std::find(keys.begin(), keys.begin() + num_valid, key);
        if (found == keys.begin() + num_valid || values[static_cast<size_t>(found - keys.begin())] != value)
        {
          std::cerr << "TopK key/payload association mismatch for output " << slot << " item=" << item << '\n';
          return false;
        }
      }
    }
  }
  return true;
}

template <size_t BlockThreads, bool Partial>
bool run_case(
  const std::vector<unsigned int>& keys,
  const std::vector<unsigned int>& values,
  unsigned int* device_keys,
  unsigned int* device_values,
  unsigned int* device_output,
  int runtime_k,
  int num_valid,
  int begin_bit,
  int end_bit)
{
  topk_kernel<BlockThreads, Partial>
    <<<1, 1>>>(device_keys, device_values, device_output, runtime_k, num_valid, begin_bit, end_bit);
  if (!check_cuda(cudaGetLastError(), "topk_kernel launch") || !check_cuda(cudaDeviceSynchronize(), "topk_kernel"))
  {
    return false;
  }
  std::vector<unsigned int> output(outputs * tile_size);
  if (!check_cuda(cudaMemcpy(output.data(), device_output, output.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost),
                  "copy TopK result"))
  {
    return false;
  }
  if (!validate_result(keys, values, output, runtime_k, Partial ? num_valid : tile_size, begin_bit, end_bit))
  {
    std::cerr << "block_threads=" << BlockThreads << " partial=" << Partial << '\n';
    return false;
  }
  return true;
}

int main()
{
  constexpr int partial_valid = 157;
  constexpr size_t bytes      = tile_size * sizeof(unsigned int);
  std::vector<unsigned int> keys(tile_size);
  std::vector<unsigned int> values(tile_size);
  for (size_t i = 0; i < tile_size; ++i)
  {
    // Out-of-prefix keys would win both min and max selection if num_valid were ignored.
    const auto digit = i < partial_valid ? 1000u + static_cast<unsigned int>((i * 73 + 19) % tile_size)
                     : i % 2 == 0        ? static_cast<unsigned int>(i)
                                         : 60000u + static_cast<unsigned int>(i);
    keys[i]          = 0x80000000u | (digit << 4) | static_cast<unsigned int>(i % 16);
    values[i]        = 3u + static_cast<unsigned int>(17 * i);
  }

  unsigned int* device_keys   = nullptr;
  unsigned int* device_values = nullptr;
  unsigned int* device_output = nullptr;
  if (!check_cuda(cudaMalloc(&device_keys, bytes), "allocate keys")
      || !check_cuda(cudaMalloc(&device_values, bytes), "allocate values")
      || !check_cuda(cudaMalloc(&device_output, outputs * bytes), "allocate output")
      || !check_cuda(cudaMemcpy(device_keys, keys.data(), bytes, cudaMemcpyHostToDevice), "copy keys")
      || !check_cuda(cudaMemcpy(device_values, values.data(), bytes, cudaMemcpyHostToDevice), "copy values"))
  {
    return EXIT_FAILURE;
  }

  const bool passed =
    run_case<128, false>(keys, values, device_keys, device_values, device_output, 11, partial_valid, 4, 20)
    && run_case<128, true>(keys, values, device_keys, device_values, device_output, 11, partial_valid, 4, 20)
    && run_case<256, false>(keys, values, device_keys, device_values, device_output, 17, partial_valid, 4, 20)
    && run_case<256, true>(keys, values, device_keys, device_values, device_output, 17, partial_valid, 4, 20)
    && run_case<128, false>(keys, values, device_keys, device_values, device_output, 0, partial_valid, 4, 20)
    && run_case<128, true>(keys, values, device_keys, device_values, device_output, 0, 0, 4, 20)
    && run_case<128, true>(keys, values, device_keys, device_values, device_output, 5, 0, 4, 20)
    && run_case<128, true>(keys, values, device_keys, device_values, device_output, tile_size, 11, 4, 20)
    && run_case<256, true>(keys, values, device_keys, device_values, device_output, tile_size, tile_size, 4, 20);

  cudaFree(device_output);
  cudaFree(device_values);
  cudaFree(device_keys);
  if (!passed)
  {
    return EXIT_FAILURE;
  }
  std::cout << "PASS: C++ Tile API TopK keys/pairs full/partial 128/256\n";
  return EXIT_SUCCESS;
}
