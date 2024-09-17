/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

//! @file
//! cub::DeviceScan provides device-wide, parallel operations for computing a prefix scan across a sequence of data
//! items residing within device-accessible memory.

#pragma once

#include <cub/config.cuh>

#include "device_launch_parameters.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/choose_offset.cuh>
#include <cub/detail/nvtx.cuh>
#include <cub/device/dispatch/dispatch_scan.cuh>
#include <cub/device/dispatch/dispatch_scan_by_key.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_deprecated.cuh>

#define elements_per_thread 16

CUB_NAMESPACE_BEGIN

template <typename IterBegin, typename IterEnd, typename Pred>
__global__ void find_if(IterBegin begin, IterEnd end, Pred pred, int* result, std::size_t num_items)
{
  // int elements_per_thread = 32;
  auto tile_size = blockDim.x * elements_per_thread;
  __shared__ int sresult;
  __shared__ int block_result;

  if (threadIdx.x == 0)
  {
    block_result = num_items;
  }

  for (int tile_offset = blockIdx.x * tile_size; tile_offset < num_items; tile_offset += tile_size * gridDim.x)
  {
    // Only one thread reads atomically and propagates it to the
    // the rest threads of the block through shared memory
    if (threadIdx.x == 0)
    {
      sresult = atomicAdd(result, 0);
    }
    __syncthreads();

    // early exit
    if (sresult < tile_offset)
    {
      return;
    }

    bool found = false;
    for (int i = 0; i < elements_per_thread; ++i)
    {
      auto index = tile_offset + threadIdx.x + i * blockDim.x;

      if (index < num_items)
      {
        if (pred(*(begin + index)))
        {
          found = true;
          atomicMin(&block_result, index);
          break;
        }
      }
    }
    if (syncthreads_or(found))
    {
      if (threadIdx.x == 0)
      {
        if (block_result < num_items)
        {
          atomicMin(result, block_result);
        }
      }
    }
  }
}

template <typename ValueType, typename OutputIteratorT>
__global__ void write_final_result_in_output_iterator_already(ValueType* d_temp_storage, OutputIteratorT d_out)
{
  *d_out = *d_temp_storage;
}

template <typename ValueType, typename NumItemsT>
__global__ void cuda_mem_set_async_dtemp_storage(ValueType* d_temp_storage, NumItemsT num_items)
{
  *d_temp_storage = num_items;
}

struct DeviceFind
{
  template <typename InputIteratorT, typename OutputIteratorT, typename ScanOpT, typename NumItemsT>
  CUB_RUNTIME_FUNCTION static void FindIf(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ScanOpT op,
    NumItemsT num_items,
    cudaStream_t stream = 0)
  {
    int block_threads = 128;
    // int items_per_thread = 2;
    int tile_size = block_threads * elements_per_thread;
    int num_tiles = static_cast<int>(cub::DivideAndRoundUp(num_items, tile_size));

    // Get device ordinal
    int device_ordinal;
    cudaError error = CubDebug(cudaGetDevice(&device_ordinal));
    if (cudaSuccess != error)
    {
      return;
    }

    // Get SM count
    int sm_count;
    error = CubDebug(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_ordinal));
    if (cudaSuccess != error)
    {
      return;
    }

    int find_if_sm_occupancy;
    error = CubDebug(
      cub::MaxSmOccupancy(find_if_sm_occupancy, find_if<InputIteratorT, InputIteratorT, ScanOpT>, block_threads));
    if (cudaSuccess != error)
    {
      return;
    }

    int findif_device_occupancy = find_if_sm_occupancy * sm_count;

    // Even-share work distribution
    int max_blocks = findif_device_occupancy; // no * CUB_SUBSCRIPTION_FACTOR(0) because max_blocks gets too big

    int findif_grid_size = CUB_MIN(num_tiles, max_blocks);

    // Temporary storage allocation requirements
    void* allocations[1]       = {};
    size_t allocation_sizes[1] = {sizeof(int)};

    // Alias the temporary allocations from the single storage blob (or
    // compute the necessary size of the blob)
    error = CubDebug(AliasTemporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes));
    if (cudaSuccess != error)
    {
      return;
    }

    int* int_temp_storage = static_cast<int*>(allocations[0]); // this shouldn't be just int

    if (d_temp_storage == nullptr)
    {
      return;
    }

    // use d_temp_storage as the intermediate device result
    // to read and write from. Then store the final result in the output iterator.
    cuda_mem_set_async_dtemp_storage<<<1, 1>>>(int_temp_storage, num_items);

    find_if<<<findif_grid_size, block_threads, 0, stream>>>(d_in, d_in + num_items, op, int_temp_storage, num_items);

    write_final_result_in_output_iterator_already<int><<<1, 1>>>(int_temp_storage, d_out);

    return;
  }
};

CUB_NAMESPACE_END
