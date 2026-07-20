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

#include <thrust/execution_policy.h>
#include <thrust/logical.h>

#include <cuda/buffer>
#include <cuda/iterator>
#include <cuda/memory>
#include <cuda/memory_pool>
#include <cuda/std/cstddef>
#include <cuda/stream>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>

#include <testing.cuh>

namespace cudax = cuda::experimental;

constexpr int empty_key   = -1;
constexpr int empty_value = -1;

// A static-capacity map with cg_size 1 so the test can use scalar device inserts.
using probing               = cudax::cuco::linear_probing<1, cudax::cuco::hash<int>>;
inline constexpr int bucket = 1;
inline constexpr ::cuda::std::size_t static_capacity =
  cudax::cuco::make_valid_capacity<probing, bucket>(::cuda::std::size_t{512});
using fixed_capacity_map_512_type = cudax::cuco::
  fixed_capacity_map<int, int, static_capacity, ::cuda::thread_scope_device, ::cuda::std::equal_to<int>, probing>;

template <class Pair>
struct iota_pair
{
  __host__ __device__ Pair operator()(typename Pair::first_type key) const noexcept
  {
    return Pair{key, key};
  }
};

struct is_nonzero
{
  __device__ bool operator()(int v) const noexcept
  {
    return v != 0;
  }
};

// Demonstrates compile-time __shared__ sizing via ref_type::capacity_v.
template <class PairIt>
__global__ void insert_shmem_kernel(fixed_capacity_map_512_type::ref_type global_ref, PairIt pairs, int num_keys)
{
  using ref_t = fixed_capacity_map_512_type::ref_type;
  static_assert(ref_t::capacity_v != ::cuda::std::dynamic_extent,
                "capacity_v must be a compile-time constant for static extents");

  __shared__ ::cuda::__uninitialized_array<ref_t::value_type, ref_t::capacity_v> smem;

  const auto idx    = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  smem[threadIdx.x] = (idx < num_keys) ? pairs[idx] : ref_t::value_type{};
  __syncthreads();
  if (idx < num_keys)
  {
    global_ref.insert(smem[threadIdx.x]);
  }
}

C2H_TEST("fixed_capacity_map static extent — shared memory sizing via capacity_v", "[shmem][static]")
{
  constexpr int num_keys = 64;

  ::cuda::stream stream{::cuda::device_ref{0}};
  auto mr = ::cuda::device_default_memory_pool(::cuda::device_ref{0});

  fixed_capacity_map_512_type map{stream, mr, cudax::cuco::empty_key{empty_key}, cudax::cuco::empty_value{empty_value}};

  const int block_size = 128;
  const int grid_size  = (num_keys + block_size - 1) / block_size;

  insert_shmem_kernel<<<grid_size, block_size, 0, stream.get()>>>(
    map.ref(),
    cuda::transform_iterator(cuda::counting_iterator<int>{0}, iota_pair<fixed_capacity_map_512_type::value_type>{}),
    num_keys);
  REQUIRE(cudaGetLastError() == cudaSuccess);

  // Verify the insertions actually landed in the global map
  auto found = ::cuda::make_buffer<int>(stream, mr, num_keys, 0);
  map.contains(stream, cuda::counting_iterator<int>{0}, cuda::counting_iterator<int>{num_keys}, found.begin());
  REQUIRE(::thrust::all_of(::thrust::cuda::par.on(stream.get()), found.data(), found.data() + num_keys, is_nonzero{}));
}
