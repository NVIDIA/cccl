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
#include <cuda/std/tuple>

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
// Compile-time (static) capacity map: 512 requested slots.
// The actual capacity is static_map_512_type::capacity_v.
// For linear_probing<1> with stride=1 this equals 512.
// ---------------------------------------------------------------------------
using static_map_512_type = cudax::cuco::static_map<int, int, 512>;

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

__global__ void find_kernel(map_type::ref_type ref, const int* keys, int* results, int n)
{
  const auto idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    auto it      = ref.find(keys[idx]);
    results[idx] = (it == ref.end()) ? empty_value : it->second;
  }
}

__global__ void insert_or_assign_kernel(map_type::ref_type ref, const ::cuda::std::pair<int, int>* pairs, int n)
{
  const auto idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    ref.insert_or_assign(pairs[idx]);
  }
}

struct add_op
{
  __device__ void operator()(int& existing, int value) const
  {
    ::cuda::atomic_ref<int, ::cuda::thread_scope_device> atomic{existing};
    atomic.fetch_add(value, ::cuda::memory_order_relaxed);
  }
};

__global__ void
insert_or_apply_kernel(map_type::ref_type ref, const ::cuda::std::pair<int, int>* pairs, int n, add_op op)
{
  const auto idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx < n)
  {
    ref.insert_or_apply(pairs[idx], op);
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

  map_type map{
    static_cast<::cuda::std::size_t>(num_keys * 2), map_type::empty_key{empty_key}, map_type::empty_value{empty_value}};

  SECTION("insert_or_assign and contains")
  {
    map.clear();

    thrust::device_vector<int> keys(num_keys);
    thrust::device_vector<int> values(num_keys);
    thrust::sequence(keys.begin(), keys.end(), int{0});
    thrust::sequence(values.begin(), values.end(), int{0});

    thrust::device_vector<::cuda::std::pair<int, int>> pairs(num_keys);
    thrust::transform(
      cuda::counting_iterator<int>{0}, cuda::counting_iterator<int>{num_keys}, pairs.begin(), [] __device__(int i) {
        return ::cuda::std::pair<int, int>{i, i};
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

    thrust::device_vector<int> found_values(num_keys, empty_value);
    find_kernel<<<grid_size, block_size>>>(ref, keys.data().get(), found_values.data().get(), num_keys);

    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

    const auto zipped  = cuda::make_zip_iterator(cuda::std::tuple{found_values.begin(), values.begin()});
    const auto find_ok = thrust::all_of(zipped, zipped + num_keys, [] __device__(const auto& p) {
      return cuda::std::get<0>(p) == cuda::std::get<1>(p);
    });
    REQUIRE(find_ok);
  }
}

C2H_TEST("static_map dynamic capacity — compute_capacity and capacity()", "[extent][dynamic]")
{
  constexpr ::cuda::std::size_t requested = 1000;
  using dyn_map_t                         = cudax::cuco::static_map<int, int>;

  const auto actual_cap = dyn_map_t::compute_capacity(requested);
  REQUIRE(actual_cap >= requested);

  static_assert(dyn_map_t::capacity_v == ::cuda::std::dynamic_extent,
                "capacity_v must be dynamic_extent for dynamic-capacity maps");
  static_assert(dyn_map_t::ref_type::capacity_v == ::cuda::std::dynamic_extent,
                "ref capacity_v must be dynamic_extent for dynamic maps");

  dyn_map_t map{requested, dyn_map_t::empty_key{empty_key}, dyn_map_t::empty_value{empty_value}};
  REQUIRE(map.capacity() == actual_cap);
}

C2H_TEST("static_map static capacity — compute_capacity and capacity_v", "[extent][static]")
{
  constexpr ::cuda::std::size_t N = 512;
  using smap_t                    = cudax::cuco::static_map<int, int, N>;

  // compute_capacity is callable at compile time
  constexpr auto computed = smap_t::template compute_capacity<N>();
  static_assert(computed >= N, "compute_capacity() result must be >= N at compile time");
  static_assert(smap_t::capacity_v != ::cuda::std::dynamic_extent,
                "capacity_v must carry a compile-time value for static capacity");
  static_assert(computed == smap_t::capacity_v,
                "compile-time compute_capacity() and capacity_v must encode the same slot count");
  static_assert(smap_t::ref_type::capacity_v == smap_t::capacity_v,
                "ref::capacity_v must equal the compile-time adjusted slot count");

  smap_t map{smap_t::empty_key{empty_key}, smap_t::empty_value{empty_value}};
  REQUIRE(map.capacity() == smap_t::capacity_v);
}

C2H_TEST("static_map static extent — device insert and contains", "[extent][static]")
{
  constexpr int num_keys = 64;
  static_assert(static_map_512_type::capacity_v >= 512, "map must have at least 512 slots");

  static_map_512_type map{static_map_512_type::empty_key{empty_key}, static_map_512_type::empty_value{empty_value}};

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

  static_map_512_type map{static_map_512_type::empty_key{empty_key}, static_map_512_type::empty_value{empty_value}};

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
// Test: static-extent map with erasure support
// ---------------------------------------------------------------------------
C2H_TEST("static_map static extent — erasure support", "[extent][static][erase]")
{
  constexpr int erased_sentinel = -2;
  constexpr int num_keys        = 32;

  using smap_erase_t = cudax::cuco::static_map<int, int, 256>;

  smap_erase_t map{smap_erase_t::empty_key{empty_key},
                   smap_erase_t::empty_value{empty_value},
                   smap_erase_t::erased_key{erased_sentinel}};

  thrust::device_vector<::cuda::std::pair<int, int>> pairs(num_keys);
  thrust::transform(
    cuda::counting_iterator<int>{0}, cuda::counting_iterator<int>{num_keys}, pairs.begin(), [] __device__(int i) {
      return ::cuda::std::pair<int, int>{i, i};
    });

  // Insert all keys
  map.insert(pairs.begin(), pairs.end());
  REQUIRE(map.size() == static_cast<::cuda::std::size_t>(num_keys));

  // Erase even keys
  thrust::device_vector<int> even_keys(num_keys / 2);
  thrust::transform(
    cuda::counting_iterator<int>{0},
    cuda::counting_iterator<int>{num_keys / 2},
    even_keys.begin(),
    [] __device__(int i) {
      return i * 2;
    });

  map.erase(even_keys.begin(), even_keys.end());

  // Verify odd keys still present, even keys gone
  auto ref = map.ref();

  thrust::device_vector<int> all_keys(num_keys);
  thrust::sequence(all_keys.begin(), all_keys.end(), int{0});
  thrust::device_vector<int> results(num_keys, 0);

  contains_ref_kernel<smap_erase_t::ref_type>
    <<<(num_keys + 127) / 128, 128>>>(ref, all_keys.data().get(), results.data().get(), num_keys);
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  const auto correct = thrust::all_of(
    cuda::make_zip_iterator(cuda::std::tuple{results.begin(), cuda::counting_iterator<int>{0}}),
    cuda::make_zip_iterator(cuda::std::tuple{results.end(), cuda::counting_iterator<int>{num_keys}}),
    [] __device__(const cuda::std::tuple<int, int>& t) {
      const auto found = cuda::std::get<0>(t);
      const auto key   = cuda::std::get<1>(t);
      return (key % 2 == 0) ? (found == 0) : (found == 1); // even erased, odd present
    });
  REQUIRE(correct);
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
               map_type::empty_key{empty_key},
               map_type::empty_value{empty_value}};

  // With load_factor=0.5 and 500 elements, capacity should be >= 1000
  REQUIRE(map.capacity() >= static_cast<::cuda::std::size_t>(num_elements / load_factor));
}

// ---------------------------------------------------------------------------
// Test: dynamic capacity constructor with runtime compute_capacity overload
// ---------------------------------------------------------------------------
C2H_TEST("static_map dynamic capacity — runtime compute_capacity", "[extent][dynamic]")
{
  const ::cuda::std::size_t capacity = 2048;

  map_type map{capacity, map_type::empty_key{empty_key}, map_type::empty_value{empty_value}};

  const auto expected_cap = map_type::compute_capacity(capacity);
  REQUIRE(map.capacity() == expected_cap);
  REQUIRE(map.capacity() >= capacity);
}
