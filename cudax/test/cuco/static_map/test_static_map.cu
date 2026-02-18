//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/sequence.h>
#include <thrust/tuple.h>

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/tuple>

#include <cuda/experimental/__cuco/static_map.cuh>

#include <testing.cuh>

namespace cudax = cuda::experimental;

using size_type = ::cuda::std::int32_t;

constexpr size_type empty_key   = -1;
constexpr size_type empty_value = -1;

using map_type = cudax::cuco::static_map<size_type, size_type>;

__global__ void contains_kernel(map_type::ref_type ref, const size_type* keys, int* results, size_type n)
{
  const auto idx = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    results[idx] = ref.contains(keys[idx]) ? 1 : 0;
  }
}

__global__ void find_kernel(map_type::ref_type ref, const size_type* keys, size_type* results, size_type n)
{
  const auto idx = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    auto it      = ref.find(keys[idx]);
    results[idx] = (it == ref.end()) ? empty_value : it->second;
  }
}

__global__ void
insert_or_assign_kernel(map_type::ref_type ref, const ::cuda::std::pair<size_type, size_type>* pairs, size_type n)
{
  const auto idx = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    ref.insert_or_assign(pairs[idx]);
  }
}

struct add_op
{
  __device__ void operator()(size_type& existing, size_type value) const
  {
    ::cuda::atomic_ref<size_type, ::cuda::thread_scope_device> atomic{existing};
    atomic.fetch_add(value, ::cuda::memory_order_relaxed);
  }
};

__global__ void insert_or_apply_kernel(
  map_type::ref_type ref, const ::cuda::std::pair<size_type, size_type>* pairs, size_type n, add_op op)
{
  const auto idx = static_cast<size_type>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    ref.insert_or_apply(pairs[idx], op);
  }
}

C2H_TEST("static_map device ref APIs", "[container]")
{
  constexpr size_type num_keys = 128;

  map_type map{
    static_cast<::cuda::std::size_t>(num_keys * 2), map_type::empty_key{empty_key}, map_type::empty_value{empty_value}};

  SECTION("insert_or_assign and contains")
  {
    map.clear();

    thrust::device_vector<size_type> keys(num_keys);
    thrust::device_vector<size_type> values(num_keys);
    thrust::sequence(keys.begin(), keys.end(), size_type{0});
    thrust::sequence(values.begin(), values.end(), size_type{0});

    thrust::device_vector<::cuda::std::pair<size_type, size_type>> pairs(num_keys);
    thrust::transform(
      thrust::counting_iterator<size_type>{0},
      thrust::counting_iterator<size_type>{num_keys},
      pairs.begin(),
      [] __device__(size_type i) {
        return ::cuda::std::pair<size_type, size_type>{i, i};
      });

    auto ref = map.ref();

    const int block_size = 128;
    const int grid_size  = (num_keys + block_size - 1) / block_size;

    insert_or_assign_kernel<<<grid_size, block_size>>>(ref, pairs.data().get(), num_keys);

    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    thrust::device_vector<int> contains_results(num_keys, 0);
    contains_kernel<<<grid_size, block_size>>>(ref, keys.data().get(), contains_results.data().get(), num_keys);

    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    const auto contains_ok = thrust::all_of(contains_results.begin(), contains_results.end(), [] __device__(int value) {
      return value == 1;
    });
    REQUIRE(contains_ok);

    thrust::device_vector<size_type> found_values(num_keys, empty_value);
    find_kernel<<<grid_size, block_size>>>(ref, keys.data().get(), found_values.data().get(), num_keys);

    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    const auto zipped  = thrust::make_zip_iterator(found_values.begin(), values.begin());
    const auto find_ok = thrust::all_of(zipped, zipped + num_keys, [] __device__(const auto& p) {
      return thrust::get<0>(p) == thrust::get<1>(p);
    });
    REQUIRE(find_ok);
  }
}
