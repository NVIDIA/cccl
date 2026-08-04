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
#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>

#include <testing.cuh>

namespace cudax = cuda::experimental;

template <int _N>
using _int_c = ::cuda::std::integral_constant<int, _N>;

using key_types =
  c2h::type_list<::cuda::std::uint8_t, ::cuda::std::uint16_t, ::cuda::std::int32_t, ::cuda::std::int64_t>;
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

// Present keys [0, num_keys) are found, absent keys [num_keys, ...) are not
struct match_expected
{
  const int* found;
  int num_keys;

  __device__ bool operator()(int i) const noexcept
  {
    return static_cast<bool>(found[i]) == (i < num_keys);
  }
};

struct is_nonzero
{
  __device__ bool operator()(int v) const noexcept
  {
    return v != 0;
  }
};

C2H_TEST("fixed_capacity_map insert and contains", "[container]", key_types, cg_sizes, bucket_sizes, probing_kinds)
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
  using map_type = cudax::cuco::fixed_capacity_map<
    key_type,
    key_type,
    ::cuda::std::dynamic_extent,
    ::cuda::thread_scope_device,
    ::cuda::std::equal_to<key_type>,
    probing_type,
    bucket_size>;
  using value_type = typename map_type::value_type;

  constexpr int num_keys = (::cuda::std::numeric_limits<key_type>::max() > 800) ? 400 : 100;

  ::cuda::stream stream{::cuda::device_ref{0}};
  auto mr = ::cuda::device_default_memory_pool(::cuda::device_ref{0});

  map_type map{stream,
               mr,
               static_cast<::cuda::std::size_t>(num_keys * 2),
               cudax::cuco::empty_key{static_cast<key_type>(-1)},
               cudax::cuco::empty_value{static_cast<key_type>(-1)}};

  auto __pairs = cuda::transform_iterator(cuda::counting_iterator<key_type>{0}, iota_pair<value_type>{});
  map.insert(stream, __pairs, __pairs + num_keys);

  // Query present keys [0, num_keys) and absent keys [num_keys, 2 * num_keys)
  auto found = ::cuda::make_buffer<int>(stream, mr, 2 * num_keys, 0);
  map.contains(
    stream, cuda::counting_iterator<key_type>{0}, cuda::counting_iterator<key_type>{2 * num_keys}, found.begin());

  REQUIRE(::thrust::all_of(
    ::thrust::cuda::par.on(stream.get()),
    cuda::counting_iterator<int>{0},
    cuda::counting_iterator<int>{2 * num_keys},
    match_expected{found.data(), num_keys}));

  // After clear the map is empty, so none of the previously inserted keys are found
  map.clear(stream);
  auto cleared = ::cuda::make_buffer<int>(stream, mr, num_keys, 1);
  map.contains(
    stream, cuda::counting_iterator<key_type>{0}, cuda::counting_iterator<key_type>{num_keys}, cleared.begin());
  REQUIRE(
    ::thrust::none_of(::thrust::cuda::par.on(stream.get()), cleared.data(), cleared.data() + num_keys, is_nonzero{}));
}

template <class _Key, class _Tp>
using __map_of = cudax::cuco::fixed_capacity_map<
  _Key,
  _Tp,
  ::cuda::std::dynamic_extent,
  ::cuda::thread_scope_device,
  ::cuda::std::equal_to<_Key>,
  cudax::cuco::linear_probing<1, cudax::cuco::hash<_Key>>,
  1>;

C2H_TEST("fixed_capacity_map key and slot size constraint", "[container]")
{
  static_assert(sizeof(typename __map_of<::cuda::std::uint8_t, ::cuda::std::uint8_t>::value_type) == 2,
                "<uint8_t, uint8_t> is a valid 2-byte slot");
  static_assert(sizeof(typename __map_of<::cuda::std::uint16_t, ::cuda::std::uint16_t>::value_type) == 4,
                "<uint16_t, uint16_t> is a valid 4-byte slot");
  static_assert(sizeof(typename __map_of<::cuda::std::uint32_t, ::cuda::std::uint32_t>::value_type) == 8,
                "<uint32_t, uint32_t> is a valid 8-byte slot");
  static_assert(sizeof(typename __map_of<::cuda::std::uint8_t, ::cuda::std::uint32_t>::value_type) == 8,
                "a mismatched <uint8_t, uint32_t> slot is a valid 8-byte slot");
}
