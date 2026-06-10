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
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/logical.h>

#include <cuda/iterator>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/static_map.cuh>

#include <testing.cuh>

namespace cudax = cuda::experimental;

// Matrix dimensions, mirroring the cuCollections static_map tests: key/value type x probing scheme x
// cooperative-group size x bucket size. `C2H_TEST` forms the cartesian product of these lists.
template <int _N>
using _int_c = ::cuda::std::integral_constant<int, _N>;

using key_types     = c2h::type_list<::cuda::std::int32_t, ::cuda::std::int64_t>;
using cg_sizes      = c2h::type_list<_int_c<1>, _int_c<2>>;
using bucket_sizes  = c2h::type_list<_int_c<1>, _int_c<2>>;
using probing_kinds = c2h::type_list<_int_c<0>, _int_c<1>>; // 0 = linear probing, 1 = double hashing

template <class _Pair>
struct iota_pair
{
  __host__ __device__ _Pair operator()(typename _Pair::first_type __i) const noexcept
  {
    return _Pair{__i, __i};
  }
};

C2H_TEST("static_map insert and contains", "[container]", key_types, cg_sizes, bucket_sizes, probing_kinds)
{
  using key_type                             = c2h::get<0, TestType>;
  [[maybe_unused]] constexpr int cg_size     = c2h::get<1, TestType>::value;
  [[maybe_unused]] constexpr int bucket_size = c2h::get<2, TestType>::value;
  [[maybe_unused]] constexpr int probing     = c2h::get<3, TestType>::value;

  using hasher = cudax::cuco::hash<key_type>;
  using probing_type =
    ::cuda::std::conditional_t<probing == 0,
                               cudax::cuco::linear_probing<cg_size, hasher>,
                               cudax::cuco::double_hashing<cg_size, hasher>>;
  using map_type =
    cudax::cuco::static_map<key_type,
                            key_type,
                            cudax::cuco::dynamic_extent,
                            ::cuda::thread_scope_device,
                            ::cuda::std::equal_to<key_type>,
                            probing_type,
                            bucket_size>;
  using value_type = typename map_type::value_type;

  constexpr int num_keys = 400;

  map_type map{static_cast<::cuda::std::size_t>(num_keys * 2),
               cudax::cuco::empty_key{key_type{-1}},
               cudax::cuco::empty_value{key_type{-1}}};

  auto __pairs = cuda::transform_iterator(cuda::counting_iterator<key_type>{0}, iota_pair<value_type>{});
  map.insert(__pairs, __pairs + num_keys);

  // Query present keys [0, num_keys) and absent keys [num_keys, 2 * num_keys)
  ::thrust::device_vector<int> found(2 * num_keys, 0);
  map.contains(cuda::counting_iterator<key_type>{0}, cuda::counting_iterator<key_type>{2 * num_keys}, found.begin());
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);

  ::thrust::host_vector<int> h_found(found);
  int mismatches = 0;
  for (int i = 0; i < 2 * num_keys; ++i)
  {
    const bool expected = i < num_keys; // present keys found, absent keys not
    mismatches += (static_cast<bool>(h_found[i]) != expected);
  }
  REQUIRE(mismatches == 0);

  // After clear the map is empty, so none of the previously inserted keys are found
  map.clear();
  ::thrust::fill(found.begin(), found.end(), 1);
  map.contains(cuda::counting_iterator<key_type>{0}, cuda::counting_iterator<key_type>{num_keys}, found.begin());
  REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
  REQUIRE(::thrust::none_of(found.begin(), found.begin() + num_keys, [] __device__(int v) {
    return v != 0;
  }));
}
