/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <cub/warp/warp_scan.cuh>

#include <thrust/detail/raw_pointer_cast.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <cuda/std/numeric>

#include "catch2_test_helper.h"

constexpr int num_warps = 4;

// example-begin inclusive-warp-scan-init-value
__global__ void InclusiveScanKernel(int* output)
{
  // Specialize WarpScan for type int
  typedef cub::WarpScan<int> warp_scan_t;
  // Allocate WarpScan shared memory for 4 warps
  __shared__ typename warp_scan_t::TempStorage temp_storage[num_warps];

  int initial_value = 1;
  int thread_data   = threadIdx.x;

  // warp #0 input: { 0,  1,  2,  3,   4, ...,  31}
  // warp #1 input: {32, 33, 34, 35,  36, ...,  63}
  // warp #2 input: {64, 65, 66, 67,  68, ...,  95}
  // warp #4 input: {96, 97, 98, 99, 100, ..., 127}

  // Collectively compute the block-wide inclusive prefix max scan
  int warp_id = threadIdx.x / 32;
  warp_scan_t(temp_storage[warp_id]).InclusiveScan(thread_data, thread_data, initial_value, cub::Sum());

  // warp #0 output: { 1,  2,   4, ...,   497}
  // warp #1 output: {33,  66, 100, ..., 1024}
  // warp #2 output: {64, 129, 195, ..., 2976}
  // warp #3 output: {96, 193, 291, ..., 4032}

  // example-end inclusive-warp-scan-init-value
  output[threadIdx.x] = thread_data;
}

CUB_TEST("Block array-based inclusive scan works with initial value", "[scan][block]")
{
  thrust::device_vector<int> d_out(num_warps * 32);

  InclusiveScanKernel<<<1, num_warps * 32>>>(thrust::raw_pointer_cast(d_out.data()));
  REQUIRE(cudaSuccess == cudaPeekAtLastError());
  REQUIRE(cudaSuccess == cudaDeviceSynchronize());

  c2h::host_vector<int> expected(d_out.size());
  expected[0] = 1; // Initial value

  // Calculate the prefix sum with an additional +1 every 32 elements
  for (int i = 1; i < num_warps * 32; ++i)
  {
    if (i % 32 == 0)
    {
      expected[i] = i + 1; // Reset at the start of each warp
    }
    else
    {
      expected[i] = expected[i - 1] + i;
    }
  }

  REQUIRE(expected == d_out);
}
