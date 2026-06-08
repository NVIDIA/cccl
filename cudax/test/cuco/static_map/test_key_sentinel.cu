//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Temporary nvcc workaround __host__ __device__ dtor conflict in cuda::buffer
#if defined(__CUDACC__)
#  pragma nv_diag_suppress 20011
#endif

#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>

#include <cuda/iterator>
#include <cuda/std/cstddef>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/static_map.cuh>

#include <testing.cuh>

namespace cudax = cuda::experimental;

constexpr int empty_key   = -1;
constexpr int empty_value = -1;

// Constructing a map with an erased-key sentinel must keep insert and contains correct (no key ever
// collides with the empty or erased sentinels).
C2H_TEST("static_map — empty and erased key sentinels", "[sentinel]")
{
  constexpr int erased_sentinel = -2;
  constexpr int num_keys        = 256;

  using probing        = cudax::cuco::linear_probing<1, cudax::cuco::hash<int>>;
  constexpr int bucket = 1;
  constexpr ::cuda::std::size_t capacity =
    cudax::cuco::make_valid_capacity<probing, bucket>(static_cast<::cuda::std::size_t>(num_keys * 2));
  using map_type = cudax::cuco::static_map<int, int, capacity>;

  map_type map{
    cudax::cuco::empty_key{empty_key}, cudax::cuco::empty_value{empty_value}, cudax::cuco::erased_key{erased_sentinel}};

  thrust::device_vector<::cuda::std::pair<int, int>> pairs(num_keys);
  thrust::transform(
    cuda::counting_iterator<int>{0}, cuda::counting_iterator<int>{num_keys}, pairs.begin(), [] __device__(int i) {
      return ::cuda::std::pair<int, int>{i, i};
    });
  map.insert(pairs.begin(), pairs.end());

  thrust::device_vector<int> keys(num_keys);
  thrust::sequence(keys.begin(), keys.end(), 0);
  thrust::device_vector<int> found(num_keys, 0);
  map.contains(keys.begin(), keys.end(), found.begin());
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  REQUIRE(thrust::all_of(found.begin(), found.end(), [] __device__(int v) {
    return v != 0;
  }));
}
