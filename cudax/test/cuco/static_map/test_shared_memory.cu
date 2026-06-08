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

// A static-capacity map. `_Capacity` must be a valid slot count, so it is computed from the probing
// scheme and bucket size (the map's defaults: linear probing, cg_size 1, bucket_size 1) rather than
// hand-written.
using default_probing               = cudax::cuco::linear_probing<1, cudax::cuco::hash<int>>;
inline constexpr int default_bucket = 1;
inline constexpr ::cuda::std::size_t static_capacity =
  cudax::cuco::make_valid_capacity<default_probing, default_bucket>(::cuda::std::size_t{512});
using static_map_512_type = cudax::cuco::static_map<int, int, static_capacity>;

// Demonstrates compile-time __shared__ sizing via ref_type::capacity_v.
__global__ void
insert_shmem_kernel(static_map_512_type::ref_type global_ref, const ::cuda::std::pair<int, int>* pairs, int n)
{
  using ref_t = static_map_512_type::ref_type;
  static_assert(ref_t::capacity_v != ::cuda::std::dynamic_extent,
                "capacity_v must be a compile-time constant for static extents");

  __shared__ ref_t::value_type smem[ref_t::capacity_v];

  const auto idx    = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  smem[threadIdx.x] = (idx < n) ? pairs[idx] : ref_t::value_type{};
  __syncthreads();
  if (idx < n)
  {
    global_ref.insert(smem[threadIdx.x]);
  }
}

C2H_TEST("static_map static extent — shared memory sizing via capacity_v", "[shmem][static]")
{
  constexpr int num_keys = 64;

  static_map_512_type map{cudax::cuco::empty_key{empty_key}, cudax::cuco::empty_value{empty_value}};

  thrust::device_vector<::cuda::std::pair<int, int>> pairs(num_keys);
  thrust::transform(
    cuda::counting_iterator<int>{0}, cuda::counting_iterator<int>{num_keys}, pairs.begin(), [] __device__(int i) {
      return ::cuda::std::pair<int, int>{i, i};
    });

  const int block_size = 128;
  const int grid_size  = (num_keys + block_size - 1) / block_size;

  insert_shmem_kernel<<<grid_size, block_size>>>(map.ref(), pairs.data().get(), num_keys);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  // Verify the insertions actually landed in the global map
  thrust::device_vector<int> keys(num_keys);
  thrust::sequence(keys.begin(), keys.end(), 0);
  thrust::device_vector<int> found(num_keys, 0);
  map.contains(keys.begin(), keys.end(), found.begin());
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  REQUIRE(thrust::all_of(found.begin(), found.end(), [] __device__(int v) {
    return v != 0;
  }));
}
