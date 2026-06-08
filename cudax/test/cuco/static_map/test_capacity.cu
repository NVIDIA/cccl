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

#include <cuda/std/cstddef>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/static_map.cuh>

#include <testing.cuh>

namespace cudax = cuda::experimental;

constexpr int empty_key   = -1;
constexpr int empty_value = -1;

C2H_TEST("static_map dynamic capacity — capacity() reflects the valid capacity", "[capacity][dynamic]")
{
  constexpr ::cuda::std::size_t requested = 1000;
  using dyn_map_t                         = cudax::cuco::static_map<int, int>;

  static_assert(dyn_map_t::capacity_v == ::cuda::std::dynamic_extent,
                "capacity_v must be dynamic_extent for dynamic-capacity maps");
  static_assert(dyn_map_t::ref_type::capacity_v == ::cuda::std::dynamic_extent,
                "ref capacity_v must be dynamic_extent for dynamic maps");

  const auto valid =
    cudax::cuco::make_valid_capacity<dyn_map_t::probing_scheme_type, dyn_map_t::bucket_size>(requested);

  dyn_map_t map{requested, cudax::cuco::empty_key{empty_key}, cudax::cuco::empty_value{empty_value}};
  REQUIRE(map.capacity() == valid);
  REQUIRE(map.capacity() >= requested);
}

C2H_TEST("static_map static capacity — valid capacity and capacity_v", "[capacity][static]")
{
  // Double hashing rounds a requested slot count up to a prime-cycle capacity, so the valid capacity
  // must be computed from the probing scheme and bucket size before it can name a static map type.
  using probing                         = cudax::cuco::double_hashing<1, cudax::cuco::hash<int>>;
  [[maybe_unused]] constexpr int bucket = 1;

  constexpr ::cuda::std::size_t requested = 1000;
  constexpr auto valid                    = cudax::cuco::make_valid_capacity<probing, bucket>(requested);
  static_assert(valid > requested, "1000 is not a valid double-hashing capacity; it rounds up");

  using smap_t =
    cudax::cuco::static_map<int, int, valid, ::cuda::thread_scope_device, ::cuda::std::equal_to<int>, probing, 1>;
  static_assert(smap_t::capacity_v == valid, "the map type carries the valid capacity, not the request");
  static_assert(smap_t::ref_type::capacity_v == valid, "the ref carries the same valid capacity");

  smap_t map{cudax::cuco::empty_key{empty_key}, cudax::cuco::empty_value{empty_value}};
  REQUIRE(map.capacity() == valid);
}

C2H_TEST("static_map dynamic extent — load factor constructor", "[capacity][dynamic][load_factor]")
{
  constexpr int num_elements   = 500;
  constexpr double load_factor = 0.5;

  cudax::cuco::static_map<int, int> map{
    static_cast<::cuda::std::size_t>(num_elements),
    load_factor,
    cudax::cuco::empty_key{empty_key},
    cudax::cuco::empty_value{empty_value}};

  // With load_factor = 0.5 and 500 elements, capacity should be >= 1000
  REQUIRE(map.capacity() >= static_cast<::cuda::std::size_t>(num_elements / load_factor));
}
