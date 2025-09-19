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
#include <cub/config.cuh>

#include <cub/device/device_for.cuh>

#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>

#include <cuda/std/array>
#include <cuda/std/mdspan>
#include <cuda/std/span>

#include <c2h/catch2_test_helper.h>
#include <c2h/utility.h>
#include <catch2_test_launch_helper.h>

// %PARAM% TEST_LAUNCH lid 0:1:2

/***********************************************************************************************************************
 * Function Objects
 **********************************************************************************************************************/

struct incrementer_t
{
  int* d_counts;

  template <class OffsetT>
  __device__ void operator()(OffsetT i, OffsetT)
  {
    atomicAdd(d_counts + i, 1); // Check if `i` was served more than once
  }
};

/***********************************************************************************************************************
 * TEST CASES
 **********************************************************************************************************************/

C2H_TEST("DeviceFor::ForEachInLayout works with layout_right", "[ForEachInLayout]")
{
  constexpr int num_items = 1000;
  using offset_t = int;
  using ext_t = cuda::std::extents<offset_t, num_items>;
  c2h::device_vector<int> counts(num_items);
  int* d_counts = thrust::raw_pointer_cast(counts.data());
  REQUIRE(cudaSuccess == cub::DeviceFor::ForEachInLayout(::cuda::std::layout_right{}, ext_t{}, incrementer_t{d_counts}));

  const auto num_of_once_marked_items =
    static_cast<offset_t>(thrust::count(c2h::device_policy, counts.begin(), counts.end(), 1));
  REQUIRE(num_of_once_marked_items == num_items);
}

// TODO: Test with layout_left once the const-correctness bug in ForEachInLayout is fixed
// C2H_TEST("DeviceFor::ForEachInLayout works with layout_left", "[ForEachInLayout]")
// {
//   constexpr int num_items = 1000;
//   using offset_t = int;
//   using ext_t = cuda::std::extents<offset_t, num_items>;
//   c2h::device_vector<int> counts(num_items);
//   int* d_counts = thrust::raw_pointer_cast(counts.data());
//   REQUIRE(cudaSuccess == cub::DeviceFor::ForEachInLayout(::cuda::std::layout_left{}, ext_t{}, incrementer_t{d_counts}));
//
//   const auto num_of_once_marked_items =
//     static_cast<offset_t>(thrust::count(c2h::device_policy, counts.begin(), counts.end(), 1));
//   REQUIRE(num_of_once_marked_items == num_items);
// }