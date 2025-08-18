// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/block/block_scan.cuh>

#include <cuda/functional>

#include <climits>

#include <c2h/catch2_test_helper.h>

template <cub::BlockScanAlgorithm Algorithm,
          int ItemsPerThread,
          int BlockDimX,
          int BlockDimY,
          int BlockDimZ,
          class T,
          class ActionT>
__global__ void block_scan_kernel(T* in, T* out, ActionT action, int valid_items)
{
  using block_scan_t = cub::BlockScan<T, BlockDimX, Algorithm, BlockDimY, BlockDimZ>;
  using storage_t    = typename block_scan_t::TempStorage;

  __shared__ storage_t storage;

  T thread_data[ItemsPerThread];

  const int tid           = static_cast<int>(cub::RowMajorTid(BlockDimX, BlockDimY, BlockDimZ));
  const int thread_offset = tid * ItemsPerThread;

  for (int item = 0; item < ItemsPerThread; item++)
  {
    const int idx     = thread_offset + item;
    thread_data[item] = in[idx];
  }
  __syncthreads();

  block_scan_t scan(storage);

  action(scan, thread_data, valid_items);

  for (int item = 0; item < ItemsPerThread; item++)
  {
    const int idx = thread_offset + item;
    out[idx]      = thread_data[item];
  }
}

template <cub::BlockScanAlgorithm Algorithm, int BlockDimX, int BlockDimY, int BlockDimZ, class T, class ActionT>
__global__ void block_scan_single_kernel(T* in, T* out, ActionT action, int valid_items)
{
  using block_scan_t = cub::BlockScan<T, BlockDimX, Algorithm, BlockDimY, BlockDimZ>;
  using storage_t    = typename block_scan_t::TempStorage;

  __shared__ storage_t storage;

  const int tid = static_cast<int>(cub::RowMajorTid(BlockDimX, BlockDimY, BlockDimZ));

  T thread_data = in[tid];

  block_scan_t scan(storage);

  action(scan, thread_data, valid_items);

  out[tid] = thread_data;
}

template <cub::BlockScanAlgorithm Algorithm,
          int ItemsPerThread,
          int BlockDimX,
          int BlockDimY,
          int BlockDimZ,
          class T,
          class ActionT>
void block_scan(c2h::device_vector<T>& in, c2h::device_vector<T>& out, ActionT action, int valid_items)
{
  dim3 block_dims(BlockDimX, BlockDimY, BlockDimZ);

  block_scan_kernel<Algorithm, ItemsPerThread, BlockDimX, BlockDimY, BlockDimZ, T, ActionT>
    <<<1, block_dims>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), action, valid_items);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

template <cub::BlockScanAlgorithm Algorithm, int BlockDimX, int BlockDimY, int BlockDimZ, class T, class ActionT>
void block_scan_single(c2h::device_vector<T>& in, c2h::device_vector<T>& out, ActionT action, int valid_items)
{
  dim3 block_dims(BlockDimX, BlockDimY, BlockDimZ);

  block_scan_single_kernel<Algorithm, BlockDimX, BlockDimY, BlockDimZ, T, ActionT>
    <<<1, block_dims>>>(thrust::raw_pointer_cast(in.data()), thrust::raw_pointer_cast(out.data()), action, valid_items);

  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());
}

enum class scan_mode
{
  exclusive,
  inclusive
};

template <class T, class ScanOpT>
T host_scan(scan_mode mode, c2h::host_vector<T>& result, ScanOpT scan_op, int valid_items, T initial_value = T{})
{
  if (result.empty())
  {
    return {};
  }

  T accumulator       = static_cast<T>(scan_op(initial_value, result[0]));
  T block_accumulator = result[0];

  if (mode == scan_mode::exclusive)
  {
    if (valid_items > 0)
    {
      result[0] = initial_value;
    }

    for (int i = 1; i < cuda::std::clamp(valid_items, 0, static_cast<int>(result.size())); i++)
    {
      T tmp             = result[i];
      result[i]         = accumulator;
      accumulator       = static_cast<T>(scan_op(accumulator, tmp));
      block_accumulator = static_cast<T>(scan_op(block_accumulator, tmp));
    }
  }
  else
  {
    if (valid_items > 0)
    {
      result[0] = accumulator;
    }

    for (int i = 1; i < cuda::std::clamp(valid_items, 0, static_cast<int>(result.size())); i++)
    {
      accumulator       = static_cast<T>(scan_op(accumulator, result[i]));
      block_accumulator = static_cast<T>(scan_op(block_accumulator, result[i]));
      result[i]         = accumulator;
    }
  }

  return block_accumulator;
}

template <class TestType>
struct params_t
{
  using type = typename c2h::get<0, TestType>;

  static constexpr int block_dim_x      = c2h::get<1, TestType>::value;
  static constexpr int block_dim_y      = c2h::get<2, TestType>::value;
  static constexpr int block_dim_z      = block_dim_y;
  static constexpr int threads_in_block = block_dim_x * block_dim_y * block_dim_z;
  static constexpr int items_per_thread = c2h::get<3, TestType>::value;
  static constexpr int tile_size        = items_per_thread * threads_in_block;

  static constexpr cub::BlockScanAlgorithm algo = c2h::get<4, TestType>::value;
  static constexpr scan_mode mode               = c2h::get<5, TestType>::value;
};
