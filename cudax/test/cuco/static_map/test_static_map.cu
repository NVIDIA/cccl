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

#include <cuda/iterator>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/static_map.cuh>

#include <testing.cuh>

namespace cudax = cuda::experimental;

constexpr int empty_key   = -1;
constexpr int empty_value = -1;

// ---------------------------------------------------------------------------
// Default dynamic-extent map (backward-compatible form)
// ---------------------------------------------------------------------------
using map_type = cudax::cuco::static_map<int, int>;

// ---------------------------------------------------------------------------
// Static-capacity map. `_Capacity` must be a *valid* slot count, so it is computed from the probing
// scheme and bucket size first and only then used to name the map type — never hand-written as a
// guessed literal. These match this map's default configuration (linear probing, cg_size 1,
// bucket_size 1).
// ---------------------------------------------------------------------------
using default_probing               = cudax::cuco::linear_probing<1, cudax::cuco::hash<int>>;
inline constexpr int default_bucket = 1;

inline constexpr ::cuda::std::size_t static_capacity =
  cudax::cuco::next_valid_capacity<default_probing, default_bucket>(::cuda::std::size_t{512});
using static_map_512_type = cudax::cuco::static_map<int, int, static_capacity>;

// ---------------------------------------------------------------------------
// Device kernels (all operate through the ref type)
// ---------------------------------------------------------------------------

__global__ void contains_kernel(map_type::ref_type ref, const int* keys, int* results, int n)
{
  const auto idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    results[idx] = ref.contains(keys[idx]) ? 1 : 0;
  }
}

template <class _RefType>
__global__ void contains_ref_kernel(_RefType ref, const int* keys, int* results, int n)
{
  const auto idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    results[idx] = ref.contains(keys[idx]) ? 1 : 0;
  }
}

__global__ void contains_static_kernel(static_map_512_type::ref_type ref, const int* keys, int* results, int n)
{
  const auto idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    results[idx] = ref.contains(keys[idx]) ? 1 : 0;
  }
}

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

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

C2H_TEST("static_map device ref APIs", "[container]")
{
  constexpr int num_keys = 128;

  map_type map{static_cast<::cuda::std::size_t>(num_keys * 2),
               cudax::cuco::empty_key{empty_key},
               cudax::cuco::empty_value{empty_value}};

  SECTION("insert and contains")
  {
    map.clear();

    thrust::device_vector<int> keys(num_keys);
    thrust::sequence(keys.begin(), keys.end(), int{0});

    thrust::device_vector<::cuda::std::pair<int, int>> pairs(num_keys);
    thrust::transform(
      cuda::counting_iterator<int>{0}, cuda::counting_iterator<int>{num_keys}, pairs.begin(), [] __device__(int i) {
        return ::cuda::std::pair<int, int>{i, i};
      });

    map.insert(pairs.begin(), pairs.end());

    auto ref = map.ref();

    const int block_size = 128;
    const int grid_size  = (num_keys + block_size - 1) / block_size;

    thrust::device_vector<int> contains_results(num_keys, 0);
    contains_kernel<<<grid_size, block_size>>>(ref, keys.data().get(), contains_results.data().get(), num_keys);

    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    const auto contains_ok = thrust::all_of(contains_results.begin(), contains_results.end(), [] __device__(int value) {
      return value == 1;
    });
    REQUIRE(contains_ok);
  }
}

C2H_TEST("static_map dynamic capacity — valid capacity and capacity()", "[extent][dynamic]")
{
  constexpr ::cuda::std::size_t requested = 1000;
  using dyn_map_t                         = cudax::cuco::static_map<int, int>;

  const auto actual_cap =
    cudax::cuco::next_valid_capacity<dyn_map_t::probing_scheme_type, dyn_map_t::bucket_size>(requested);
  REQUIRE(actual_cap >= requested);

  static_assert(dyn_map_t::capacity_v == ::cuda::std::dynamic_extent,
                "capacity_v must be dynamic_extent for dynamic-capacity maps");
  static_assert(dyn_map_t::ref_type::capacity_v == ::cuda::std::dynamic_extent,
                "ref capacity_v must be dynamic_extent for dynamic maps");

  dyn_map_t map{requested, cudax::cuco::empty_key{empty_key}, cudax::cuco::empty_value{empty_value}};
  REQUIRE(map.capacity() == actual_cap);
}

C2H_TEST("static_map static capacity — valid capacity and capacity_v", "[extent][static]")
{
  // Double hashing rounds a requested slot count up to a prime-cycle capacity, so the valid capacity
  // must be computed from the probing scheme and bucket size before it can name a static map type.
  using probing        = cudax::cuco::double_hashing<1, cudax::cuco::hash<int>>;
  constexpr int bucket = 1;

  constexpr ::cuda::std::size_t requested = 1000;
  constexpr auto valid                    = cudax::cuco::next_valid_capacity<probing, bucket>(requested);
  static_assert(valid > requested, "1000 is not a valid double-hashing capacity; it rounds up");

  using smap_t =
    cudax::cuco::static_map<int, int, valid, ::cuda::thread_scope_device, ::cuda::std::equal_to<int>, probing, 1>;
  static_assert(smap_t::capacity_v == valid, "the map type carries the valid capacity, not the request");
  static_assert(smap_t::ref_type::capacity_v == valid, "the ref carries the same valid capacity");

  smap_t map{cudax::cuco::empty_key{empty_key}, cudax::cuco::empty_value{empty_value}};
  REQUIRE(map.capacity() == valid);
}

C2H_TEST("static_map static extent — device insert and contains", "[extent][static]")
{
  constexpr int num_keys = 64;
  static_assert(static_map_512_type::capacity_v >= 512, "map must have at least 512 slots");

  static_map_512_type map{cudax::cuco::empty_key{empty_key}, cudax::cuco::empty_value{empty_value}};

  thrust::device_vector<::cuda::std::pair<int, int>> pairs(num_keys);
  thrust::transform(
    cuda::counting_iterator<int>{0}, cuda::counting_iterator<int>{num_keys}, pairs.begin(), [] __device__(int i) {
      return ::cuda::std::pair<int, int>{i, i * 10};
    });

  map.insert(pairs.begin(), pairs.end());

  thrust::device_vector<int> keys(num_keys);
  thrust::sequence(keys.begin(), keys.end(), int{0});
  thrust::device_vector<int> results(num_keys, 0);
  auto ref = map.ref();

  const int block_size = 128;
  const int grid_size  = (num_keys + block_size - 1) / block_size;
  contains_static_kernel<<<grid_size, block_size>>>(ref, keys.data().get(), results.data().get(), num_keys);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const auto all_found = thrust::all_of(results.begin(), results.end(), [] __device__(int v) {
    return v == 1;
  });
  REQUIRE(all_found);
}

// ---------------------------------------------------------------------------
// Test: static-extent map — compile-time shared-memory sizing kernel
// ---------------------------------------------------------------------------
C2H_TEST("static_map static extent — shared memory sizing via capacity_v", "[extent][static][shmem]")
{
  constexpr int num_keys = 64;

  static_map_512_type map{cudax::cuco::empty_key{empty_key}, cudax::cuco::empty_value{empty_value}};

  thrust::device_vector<::cuda::std::pair<int, int>> pairs(num_keys);
  thrust::transform(
    cuda::counting_iterator<int>{0}, cuda::counting_iterator<int>{num_keys}, pairs.begin(), [] __device__(int i) {
      return ::cuda::std::pair<int, int>{i, i};
    });

  auto ref = map.ref();

  const int block_size = 128;
  const int grid_size  = (num_keys + block_size - 1) / block_size;

  // This kernel uses ref_type::capacity_v to size __shared__ memory
  insert_shmem_kernel<<<grid_size, block_size>>>(ref, pairs.data().get(), num_keys);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  // Verify the insertions actually landed in the global map
  thrust::device_vector<int> keys(num_keys);
  thrust::sequence(keys.begin(), keys.end(), int{0});
  thrust::device_vector<int> results(num_keys, 0);

  contains_static_kernel<<<grid_size, block_size>>>(ref, keys.data().get(), results.data().get(), num_keys);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const auto all_found = thrust::all_of(results.begin(), results.end(), [] __device__(int v) {
    return v == 1;
  });
  REQUIRE(all_found);
}

// ---------------------------------------------------------------------------
// Test: static-extent map constructed with an erased-key sentinel
// ---------------------------------------------------------------------------
C2H_TEST("static_map static extent — erased_key constructor", "[extent][static][erase]")
{
  constexpr int erased_sentinel = -2;
  constexpr int num_keys        = 32;

  // Compute the valid capacity before naming the map type (the default linear scheme here).
  constexpr ::cuda::std::size_t erase_capacity =
    cudax::cuco::next_valid_capacity<default_probing, default_bucket>(::cuda::std::size_t{256});
  using smap_erase_t = cudax::cuco::static_map<int, int, erase_capacity>;

  smap_erase_t map{
    cudax::cuco::empty_key{empty_key}, cudax::cuco::empty_value{empty_value}, cudax::cuco::erased_key{erased_sentinel}};

  thrust::device_vector<::cuda::std::pair<int, int>> pairs(num_keys);
  thrust::transform(
    cuda::counting_iterator<int>{0}, cuda::counting_iterator<int>{num_keys}, pairs.begin(), [] __device__(int i) {
      return ::cuda::std::pair<int, int>{i, i};
    });

  map.insert(pairs.begin(), pairs.end());

  auto ref = map.ref();

  thrust::device_vector<int> all_keys(num_keys);
  thrust::sequence(all_keys.begin(), all_keys.end(), int{0});
  thrust::device_vector<int> results(num_keys, 0);

  contains_ref_kernel<smap_erase_t::ref_type>
    <<<(num_keys + 127) / 128, 128>>>(ref, all_keys.data().get(), results.data().get(), num_keys);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const auto all_found = thrust::all_of(results.begin(), results.end(), [] __device__(int v) {
    return v == 1;
  });
  REQUIRE(all_found);
}

// ---------------------------------------------------------------------------
// Test: load-factor constructor (dynamic extent only)
// ---------------------------------------------------------------------------
C2H_TEST("static_map dynamic extent — load factor constructor", "[extent][dynamic][load_factor]")
{
  constexpr int num_elements   = 500;
  constexpr double load_factor = 0.5;

  map_type map{static_cast<::cuda::std::size_t>(num_elements),
               load_factor,
               cudax::cuco::empty_key{empty_key},
               cudax::cuco::empty_value{empty_value}};

  // With load_factor=0.5 and 500 elements, capacity should be >= 1000
  REQUIRE(map.capacity() >= static_cast<::cuda::std::size_t>(num_elements / load_factor));
}

// ---------------------------------------------------------------------------
// Test: dynamic capacity constructor with runtime valid-capacity rounding
// ---------------------------------------------------------------------------
C2H_TEST("static_map dynamic capacity — runtime valid capacity", "[extent][dynamic]")
{
  const ::cuda::std::size_t capacity = 2048;

  map_type map{capacity, cudax::cuco::empty_key{empty_key}, cudax::cuco::empty_value{empty_value}};

  const auto expected_cap =
    cudax::cuco::next_valid_capacity<map_type::probing_scheme_type, map_type::bucket_size>(capacity);
  REQUIRE(map.capacity() == expected_cap);
  REQUIRE(map.capacity() >= capacity);
}

// ---------------------------------------------------------------------------
// Test: valid_capacity descriptor and capacity factories
// ---------------------------------------------------------------------------
C2H_TEST("static_map capacity descriptor and factories", "[capacity]")
{
  using probing        = cudax::cuco::double_hashing<1, cudax::cuco::hash<int>>;
  constexpr int bucket = 1;

  static_assert(cudax::cuco::is_double_hashing_v<probing>, "scheme is double hashing");

  // next_valid_capacity rounds up and is idempotent; is_valid_capacity is derived from it
  constexpr auto valid = cudax::cuco::next_valid_capacity<probing, bucket>(::cuda::std::size_t{1000});
  static_assert(valid >= 1000, "rounds up");
  static_assert(cudax::cuco::is_valid_capacity<probing, bucket>(valid), "result is valid");
  static_assert(cudax::cuco::next_valid_capacity<probing, bucket>(valid) == valid, "idempotent");

  // canonical types: equal-rounding requests share one descriptor type
  using a = cudax::cuco::valid_capacity_for_t<probing, bucket, 1000>;
  using b = cudax::cuco::valid_capacity_for_t<probing, bucket, 1008>;
  static_assert(::cuda::std::is_same_v<a, b>, "equal-rounding requests are the same type");
  static_assert(a{}.capacity() == valid, "static descriptor capacity folds to the valid value");
  static_assert(a::bucket_size == bucket && a::cg_size == 1, "descriptor exposes bucket and cg size");

  // runtime factory + operator%
  const auto cap = cudax::cuco::make_valid_capacity<probing, bucket>(::cuda::std::size_t{1000});
  REQUIRE(cap.capacity() == valid);
  REQUIRE(cap.num_buckets() == valid); // bucket_size == 1
  REQUIRE((valid + 3) % cap == ::cuda::std::size_t{3});
}
