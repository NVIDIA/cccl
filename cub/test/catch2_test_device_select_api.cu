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

#include <cub/device/device_select.cuh>

#include <thrust/device_vector.h>
#include <thrust/equal.h>

#include <cstddef>

#include "catch2_test_helper.h"

// example-begin segmented-select-iseven
struct is_even_t
{
  __host__ __device__ bool operator()(int flag) const
  {
    return !(flag % 2);
  }
};
// example-end segmented-select-iseven

CUB_TEST("cub::DeviceSelect::FlaggedIf works with int data elements", "[select][device]")
{
  // example-begin segmented-select-flaggedif
  constexpr int num_items            = 8;
  thrust::device_vector<int> d_in    = {0, 1, 2, 3, 4, 5, 6, 7};
  thrust::device_vector<int> d_flags = {8, 6, 7, 5, 3, 0, 9, 3};
  thrust::device_vector<int> d_out(num_items);
  thrust::device_vector<int> d_num_selected_out(num_items);
  is_even_t is_even{};

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::FlaggedIf(
    d_temp_storage,
    temp_storage_bytes,
    d_in.begin(),
    d_flags.begin(),
    d_out.begin(),
    d_num_selected_out.data(),
    num_items,
    is_even);

  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Run selection
  cub::DeviceSelect::FlaggedIf(
    d_temp_storage,
    temp_storage_bytes,
    d_in.begin(),
    d_flags.begin(),
    d_out.begin(),
    d_num_selected_out.data(),
    num_items,
    is_even);

  thrust::device_vector<int> expected{0, 1, 5};
  // example-end segmented-select-flaggedif

  REQUIRE(d_num_selected_out[0] == static_cast<int>(expected.size()));
  d_out.resize(d_num_selected_out[0]);
  REQUIRE(d_out == expected);
}

CUB_TEST("cub::DeviceSelect::FlaggedIf in-place works with int data elements", "[select][device]")
{
  // example-begin segmented-select-flaggedif-inplace
  constexpr int num_items            = 8;
  thrust::device_vector<int> d_data  = {0, 1, 2, 3, 4, 5, 6, 7};
  thrust::device_vector<int> d_flags = {8, 6, 7, 5, 3, 0, 9, 3};
  thrust::device_vector<int> d_num_selected_out(num_items);
  is_even_t is_even{};

  // Determine temporary device storage requirements
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  cub::DeviceSelect::FlaggedIf(
    d_temp_storage, temp_storage_bytes, d_data.begin(), d_flags.begin(), d_num_selected_out.data(), num_items, is_even);

  // Allocate temporary storage
  cudaMalloc(&d_temp_storage, temp_storage_bytes);

  // Run selection
  cub::DeviceSelect::FlaggedIf(
    d_temp_storage, temp_storage_bytes, d_data.begin(), d_flags.begin(), d_num_selected_out.data(), num_items, is_even);

  thrust::device_vector<int> expected{0, 1, 5};
  // example-end segmented-select-flaggedif-inplace

  REQUIRE(d_num_selected_out[0] == static_cast<int>(expected.size()));
  d_data.resize(d_num_selected_out[0]);
  REQUIRE(d_data == expected);
}
