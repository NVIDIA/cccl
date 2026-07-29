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
#include <cuda/memory_pool>
#include <cuda/std/cstddef>
#include <cuda/stream>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>

#include <testing.cuh>

namespace cudax = cuda::experimental;

constexpr int empty_key   = -1;
constexpr int empty_value = -1;

// Constructing a map with an erased-key sentinel must keep insert and contains correct (no key ever
// collides with the empty or erased sentinels).
template <class _Pair>
struct iota_pair
{
  __host__ __device__ _Pair operator()(typename _Pair::first_type __i) const noexcept
  {
    return _Pair{__i, __i};
  }
};

struct is_nonzero
{
  __device__ bool operator()(int v) const noexcept
  {
    return v != 0;
  }
};

C2H_TEST("fixed_capacity_map — empty and erased key sentinels", "[sentinel]")
{
  constexpr int erased_sentinel = -2;
  constexpr int num_keys        = 256;

  using probing                         = cudax::cuco::linear_probing<1, cudax::cuco::hash<int>>;
  [[maybe_unused]] constexpr int bucket = 1;
  [[maybe_unused]] constexpr ::cuda::std::size_t capacity =
    cudax::cuco::make_valid_capacity<probing, bucket>(::cuda::std::size_t{num_keys} * 2);
  using map_type = cudax::cuco::fixed_capacity_map<int, int, capacity>;

  ::cuda::stream stream{::cuda::device_ref{0}};
  auto mr = ::cuda::device_default_memory_pool(::cuda::device_ref{0});

  map_type map{stream,
               mr,
               cudax::cuco::empty_key{empty_key},
               cudax::cuco::empty_value{empty_value},
               cudax::cuco::erased_key{erased_sentinel}};

  auto __pairs = cuda::transform_iterator(cuda::counting_iterator<int>{0}, iota_pair<::cuda::std::pair<int, int>>{});
  map.insert(stream, __pairs, __pairs + num_keys);

  auto found = ::cuda::make_buffer<int>(stream, mr, num_keys, 0);
  map.contains(stream, cuda::counting_iterator<int>{0}, cuda::counting_iterator<int>{num_keys}, found.begin());
  REQUIRE(::thrust::all_of(::thrust::cuda::par.on(stream.get()), found.data(), found.data() + num_keys, is_nonzero{}));
}
