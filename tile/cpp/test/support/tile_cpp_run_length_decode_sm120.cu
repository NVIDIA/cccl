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
#include <limits>
#include <vector>

#include <cuda_runtime.h>

namespace ct = ::cuda::tiles;

inline constexpr size_t run_count     = 128;
inline constexpr size_t decoded_count = 256;

__tile_global__ void run_length_decode_kernel(
  const int* run_values_input,
  const unsigned int* run_lengths_input,
  int* decoded_output,
  int* decoded_with_offsets_output,
  unsigned int* relative_offsets_output,
  unsigned int decoded_window_offset,
  unsigned int offsets_window_offset)
{
  auto run_values_span           = ct::tensor_span{run_values_input, ct::shape<run_count>{}};
  auto run_lengths_span          = ct::tensor_span{run_lengths_input, ct::shape<run_count>{}};
  auto decoded_span              = ct::tensor_span{decoded_output, ct::shape<decoded_count>{}};
  auto decoded_with_offsets_span = ct::tensor_span{decoded_with_offsets_output, ct::shape<decoded_count>{}};
  auto relative_offsets_span     = ct::tensor_span{relative_offsets_output, ct::shape<decoded_count>{}};

  auto run_values_view           = ct::partition_view{run_values_span, ct::shape<run_count>{}};
  auto run_lengths_view          = ct::partition_view{run_lengths_span, ct::shape<run_count>{}};
  auto decoded_view              = ct::partition_view{decoded_span, ct::shape<decoded_count>{}};
  auto decoded_with_offsets_view = ct::partition_view{decoded_with_offsets_span, ct::shape<decoded_count>{}};
  auto relative_offsets_view     = ct::partition_view{relative_offsets_span, ct::shape<decoded_count>{}};

  auto run_values  = run_values_view.load(0);
  auto run_lengths = run_lengths_view.load(0);
  auto decoded     = ::cuda::coop::tile::run_length_decode<decoded_count>(
    run_values, run_lengths, decoded_window_offset, ::cuda::coop::tile::run_length_decode_policy<>{});

  auto offsets_window_offset_tile               = ct::reshape(ct::tile{offsets_window_offset}, ct::shape<1>{});
  auto [decoded_with_offsets, relative_offsets] = ::cuda::coop::tile::run_length_decode_with_offsets<decoded_count>(
    run_values, run_lengths, offsets_window_offset_tile, ::cuda::coop::tile::run_length_decode_policy<>{});

  decoded_view.store(decoded, 0);
  decoded_with_offsets_view.store(decoded_with_offsets, 0);
  relative_offsets_view.store(relative_offsets, 0);
}

__tile_global__ void run_length_decode_u64_default_kernel(
  const int* run_values_input,
  const unsigned long long* run_lengths_input,
  int* decoded_tail_output,
  unsigned long long* relative_offsets_tail_output,
  int* decoded_high_output,
  unsigned long long* relative_offsets_high_output,
  unsigned long long tail_window_offset,
  unsigned long long high_window_offset)
{
  auto run_values_span            = ct::tensor_span{run_values_input, ct::shape<run_count>{}};
  auto run_lengths_span           = ct::tensor_span{run_lengths_input, ct::shape<run_count>{}};
  auto decoded_tail_span          = ct::tensor_span{decoded_tail_output, ct::shape<decoded_count>{}};
  auto relative_offsets_tail_span = ct::tensor_span{relative_offsets_tail_output, ct::shape<decoded_count>{}};
  auto decoded_high_span          = ct::tensor_span{decoded_high_output, ct::shape<decoded_count>{}};
  auto relative_offsets_high_span = ct::tensor_span{relative_offsets_high_output, ct::shape<decoded_count>{}};

  auto run_values_view            = ct::partition_view{run_values_span, ct::shape<run_count>{}};
  auto run_lengths_view           = ct::partition_view{run_lengths_span, ct::shape<run_count>{}};
  auto decoded_tail_view          = ct::partition_view{decoded_tail_span, ct::shape<decoded_count>{}};
  auto relative_offsets_tail_view = ct::partition_view{relative_offsets_tail_span, ct::shape<decoded_count>{}};
  auto decoded_high_view          = ct::partition_view{decoded_high_span, ct::shape<decoded_count>{}};
  auto relative_offsets_high_view = ct::partition_view{relative_offsets_high_span, ct::shape<decoded_count>{}};

  auto run_values  = run_values_view.load(0);
  auto run_lengths = run_lengths_view.load(0);

  auto [decoded_tail, relative_offsets_tail] =
    ::cuda::coop::tile::run_length_decode_with_offsets<decoded_count>(run_values, run_lengths, tail_window_offset);
  auto [decoded_high, relative_offsets_high] =
    ::cuda::coop::tile::run_length_decode_with_offsets<decoded_count>(run_values, run_lengths, high_window_offset);

  decoded_tail_view.store(decoded_tail, 0);
  relative_offsets_tail_view.store(relative_offsets_tail, 0);
  decoded_high_view.store(decoded_high, 0);
  relative_offsets_high_view.store(relative_offsets_high, 0);
}

bool check_cuda(cudaError_t status)
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
  constexpr unsigned int decoded_window_offset        = 7;
  constexpr unsigned int offsets_window_offset        = 191;
  constexpr unsigned long long u64_high_window_offset = 1ULL << 32;
  constexpr size_t nonzero_u64_run_count              = 96;
  constexpr size_t run_value_bytes                    = run_count * sizeof(int);
  constexpr size_t run_length_bytes                   = run_count * sizeof(unsigned int);
  constexpr size_t run_length_u64_bytes               = run_count * sizeof(unsigned long long);
  constexpr size_t decoded_value_bytes                = decoded_count * sizeof(int);
  constexpr size_t relative_offset_bytes              = decoded_count * sizeof(unsigned int);
  constexpr size_t relative_offset_u64_bytes          = decoded_count * sizeof(unsigned long long);

  std::vector<int> run_values(run_count);
  std::vector<unsigned int> run_lengths(run_count);
  std::vector<unsigned long long> run_lengths_u64(run_count, 0);
  unsigned int total_decoded_size           = 0;
  unsigned long long total_decoded_size_u64 = 0;
  for (size_t i = 0; i < run_count; ++i)
  {
    run_values[i]  = 1000 + static_cast<int>(i * 3);
    run_lengths[i] = static_cast<unsigned int>(i % 5 + 1);
    total_decoded_size += run_lengths[i];
    if (i < nonzero_u64_run_count)
    {
      run_lengths_u64[i] = static_cast<unsigned long long>(i % 5 + 1);
      total_decoded_size_u64 += run_lengths_u64[i];
    }
  }

  std::vector<int> decoded_output(decoded_count);
  std::vector<int> decoded_with_offsets_output(decoded_count);
  std::vector<unsigned int> relative_offsets_output(decoded_count);
  std::vector<int> decoded_u64_tail_output(decoded_count);
  std::vector<unsigned long long> relative_offsets_u64_tail_output(decoded_count);
  std::vector<int> decoded_u64_high_output(decoded_count);
  std::vector<unsigned long long> relative_offsets_u64_high_output(decoded_count);

  int* device_run_values                               = nullptr;
  unsigned int* device_run_lengths                     = nullptr;
  unsigned long long* device_run_lengths_u64           = nullptr;
  int* device_decoded                                  = nullptr;
  int* device_decoded_with_offsets                     = nullptr;
  unsigned int* device_relative_offsets                = nullptr;
  int* device_decoded_u64_tail                         = nullptr;
  unsigned long long* device_relative_offsets_u64_tail = nullptr;
  int* device_decoded_u64_high                         = nullptr;
  unsigned long long* device_relative_offsets_u64_high = nullptr;
  const auto relative_offset_invalid_sentinel          = (::std::numeric_limits<unsigned int>::max)();
  const auto relative_offset_u64_invalid_sentinel      = (::std::numeric_limits<unsigned long long>::max)();

  if (cudaMalloc(&device_run_values, run_value_bytes) != cudaSuccess
      || cudaMalloc(&device_run_lengths, run_length_bytes) != cudaSuccess
      || cudaMalloc(&device_run_lengths_u64, run_length_u64_bytes) != cudaSuccess
      || cudaMalloc(&device_decoded, decoded_value_bytes) != cudaSuccess
      || cudaMalloc(&device_decoded_with_offsets, decoded_value_bytes) != cudaSuccess
      || cudaMalloc(&device_relative_offsets, relative_offset_bytes) != cudaSuccess
      || cudaMalloc(&device_decoded_u64_tail, decoded_value_bytes) != cudaSuccess
      || cudaMalloc(&device_relative_offsets_u64_tail, relative_offset_u64_bytes) != cudaSuccess
      || cudaMalloc(&device_decoded_u64_high, decoded_value_bytes) != cudaSuccess
      || cudaMalloc(&device_relative_offsets_u64_high, relative_offset_u64_bytes) != cudaSuccess)
  {
    std::cerr << "cudaMalloc failed: " << cudaGetErrorString(cudaGetLastError()) << '\n';
    return EXIT_FAILURE;
  }

  if (!check_cuda(cudaMemcpy(device_run_values, run_values.data(), run_value_bytes, cudaMemcpyHostToDevice))
      || !check_cuda(cudaMemcpy(device_run_lengths, run_lengths.data(), run_length_bytes, cudaMemcpyHostToDevice))
      || !check_cuda(
        cudaMemcpy(device_run_lengths_u64, run_lengths_u64.data(), run_length_u64_bytes, cudaMemcpyHostToDevice)))
  {
    return EXIT_FAILURE;
  }

  run_length_decode_kernel<<<1, 1>>>(
    device_run_values,
    device_run_lengths,
    device_decoded,
    device_decoded_with_offsets,
    device_relative_offsets,
    decoded_window_offset,
    offsets_window_offset);
  if (const auto launch_status = cudaGetLastError(); launch_status != cudaSuccess)
  {
    std::cerr << "run_length_decode_kernel launch failed: " << cudaGetErrorString(launch_status) << '\n';
    return EXIT_FAILURE;
  }
  if (const auto sync_status = cudaDeviceSynchronize(); sync_status != cudaSuccess)
  {
    std::cerr << "run_length_decode_kernel execution failed: " << cudaGetErrorString(sync_status) << '\n';
    return EXIT_FAILURE;
  }

  if (total_decoded_size_u64 < 8)
  {
    std::cerr << "Run-Length Decode u64 fixture is too small for its tail window\n";
    return EXIT_FAILURE;
  }
  const auto u64_tail_window_offset = total_decoded_size_u64 - 8;
  run_length_decode_u64_default_kernel<<<1, 1>>>(
    device_run_values,
    device_run_lengths_u64,
    device_decoded_u64_tail,
    device_relative_offsets_u64_tail,
    device_decoded_u64_high,
    device_relative_offsets_u64_high,
    u64_tail_window_offset,
    u64_high_window_offset);
  if (const auto launch_status = cudaGetLastError(); launch_status != cudaSuccess)
  {
    std::cerr << "run_length_decode_u64_default_kernel launch failed: " << cudaGetErrorString(launch_status) << '\n';
    return EXIT_FAILURE;
  }
  if (const auto sync_status = cudaDeviceSynchronize(); sync_status != cudaSuccess)
  {
    std::cerr << "run_length_decode_u64_default_kernel execution failed: " << cudaGetErrorString(sync_status) << '\n';
    return EXIT_FAILURE;
  }

  if (!check_cuda(cudaMemcpy(decoded_output.data(), device_decoded, decoded_value_bytes, cudaMemcpyDeviceToHost))
      || !check_cuda(cudaMemcpy(
        decoded_with_offsets_output.data(), device_decoded_with_offsets, decoded_value_bytes, cudaMemcpyDeviceToHost))
      || !check_cuda(cudaMemcpy(
        relative_offsets_output.data(), device_relative_offsets, relative_offset_bytes, cudaMemcpyDeviceToHost))
      || !check_cuda(cudaMemcpy(
        decoded_u64_tail_output.data(), device_decoded_u64_tail, decoded_value_bytes, cudaMemcpyDeviceToHost))
      || !check_cuda(cudaMemcpy(relative_offsets_u64_tail_output.data(),
                                device_relative_offsets_u64_tail,
                                relative_offset_u64_bytes,
                                cudaMemcpyDeviceToHost))
      || !check_cuda(cudaMemcpy(
        decoded_u64_high_output.data(), device_decoded_u64_high, decoded_value_bytes, cudaMemcpyDeviceToHost))
      || !check_cuda(cudaMemcpy(relative_offsets_u64_high_output.data(),
                                device_relative_offsets_u64_high,
                                relative_offset_u64_bytes,
                                cudaMemcpyDeviceToHost)))
  {
    return EXIT_FAILURE;
  }

  auto validate_window =
    [&](unsigned int window_offset, const std::vector<int>& values, const std::vector<unsigned int>* relative_offsets) {
      for (size_t i = 0; i < decoded_count; ++i)
      {
        const auto decoded_index              = window_offset + static_cast<unsigned int>(i);
        int expected_value                    = 0;
        unsigned int expected_relative_offset = relative_offset_invalid_sentinel;
        unsigned int run_begin                = 0;
        for (size_t run = 0; run < run_count; ++run)
        {
          const unsigned int run_end = run_begin + run_lengths[run];
          if (decoded_index >= run_begin && decoded_index < run_end)
          {
            expected_value           = run_values[run];
            expected_relative_offset = decoded_index - run_begin;
            break;
          }
          run_begin = run_end;
        }

        if (values[i] != expected_value
            || (relative_offsets != nullptr && (*relative_offsets)[i] != expected_relative_offset))
        {
          std::cerr << "Run-Length Decode mismatch at output " << i << " for window offset " << window_offset
                    << ": value=" << values[i] << " expected_value=" << expected_value;
          if (relative_offsets != nullptr)
          {
            std::cerr << " relative_offset=" << (*relative_offsets)[i]
                      << " expected_relative_offset=" << expected_relative_offset;
          }
          std::cerr << '\n';
          return false;
        }
      }
      return true;
    };

  if (!validate_window(decoded_window_offset, decoded_output, nullptr)
      || !validate_window(offsets_window_offset, decoded_with_offsets_output, &relative_offsets_output))
  {
    return EXIT_FAILURE;
  }

  if (total_decoded_size <= offsets_window_offset || total_decoded_size >= offsets_window_offset + decoded_count)
  {
    std::cerr << "Run-Length Decode fixture did not exercise both valid and "
                 "out-of-range output lanes\n";
    return EXIT_FAILURE;
  }

  auto validate_u64_window =
    [&](unsigned long long window_offset,
        const std::vector<int>& values,
        const std::vector<unsigned long long>& relative_offsets) {
      for (size_t i = 0; i < decoded_count; ++i)
      {
        const auto decoded_index      = window_offset + static_cast<unsigned long long>(i);
        int expected_value            = 0;
        auto expected_relative_offset = relative_offset_u64_invalid_sentinel;
        unsigned long long run_begin  = 0;
        for (size_t run = 0; run < run_count; ++run)
        {
          const auto run_end = run_begin + run_lengths_u64[run];
          if (decoded_index >= run_begin && decoded_index < run_end)
          {
            expected_value           = run_values[run];
            expected_relative_offset = decoded_index - run_begin;
            break;
          }
          run_begin = run_end;
        }

        if (values[i] != expected_value || relative_offsets[i] != expected_relative_offset)
        {
          std::cerr
            << "Run-Length Decode u64 mismatch at output " << i << " for window offset " << window_offset << ": value="
            << values[i] << " expected_value=" << expected_value << " relative_offset=" << relative_offsets[i]
            << " expected_relative_offset=" << expected_relative_offset << '\n';
          return false;
        }
      }
      return true;
    };

  if (!validate_u64_window(u64_tail_window_offset, decoded_u64_tail_output, relative_offsets_u64_tail_output)
      || !validate_u64_window(u64_high_window_offset, decoded_u64_high_output, relative_offsets_u64_high_output))
  {
    return EXIT_FAILURE;
  }

  cudaFree(device_relative_offsets_u64_high);
  cudaFree(device_decoded_u64_high);
  cudaFree(device_relative_offsets_u64_tail);
  cudaFree(device_decoded_u64_tail);
  cudaFree(device_relative_offsets);
  cudaFree(device_decoded_with_offsets);
  cudaFree(device_decoded);
  cudaFree(device_run_lengths_u64);
  cudaFree(device_run_lengths);
  cudaFree(device_run_values);

  std::cout << "PASS: C++ Tile API Run-Length Decode\n";
  return EXIT_SUCCESS;
}
