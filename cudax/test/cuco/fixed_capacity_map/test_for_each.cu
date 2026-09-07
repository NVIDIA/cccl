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
#endif // defined(__CUDACC__)

#include <thrust/execution_policy.h>
#include <thrust/logical.h>

#include <cuda/atomic>
#include <cuda/buffer>
#include <cuda/functional>
#include <cuda/iterator>
#include <cuda/memory_pool>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/__cuco/capacity.cuh>
#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>

#include <cooperative_groups.h>
#include <testing.cuh>

namespace cudax = cuda::experimental;

template <int N>
using int_c = ::cuda::std::integral_constant<int, N>;

using key_types     = c2h::type_list<::cuda::std::int32_t, ::cuda::std::int64_t>;
using mapped_types  = c2h::type_list<::cuda::std::int32_t, ::cuda::std::int64_t>;
using cg_sizes      = c2h::type_list<int_c<1>, int_c<2>>;
using bucket_sizes  = c2h::type_list<int_c<1>, int_c<2>>;
using probing_kinds = c2h::type_list<int_c<0>, int_c<1>>; // 0 = linear probing, 1 = double hashing

// Payloads are offset from their key so a bug that passes the key instead of the whole slot is caught.
constexpr int payload_offset = 7;

template <class Pair>
struct iota_pair
{
  __host__ __device__ Pair operator()(typename Pair::first_type i) const noexcept
  {
    using mapped_type = typename Pair::second_type;
    return Pair{i, static_cast<mapped_type>(i) + static_cast<mapped_type>(payload_offset)};
  }
};

// Tags every matched slot, but only if the slot carries the payload that belongs to its key, so a
// callback invoked with a partial or wrong slot leaves the key unvisited.
template <class Value>
struct record_visit
{
  int* visits;

  __device__ void operator()(Value slot) const noexcept
  {
    using mapped_type = typename Value::second_type;
    if (slot.second != static_cast<mapped_type>(slot.first) + static_cast<mapped_type>(payload_offset))
    {
      return;
    }
    ::cuda::atomic_ref<int, ::cuda::thread_scope_device>{visits[static_cast<int>(slot.first)]}.fetch_add(
      1, ::cuda::memory_order_relaxed);
  }
};

// Keys in [0, num_keys) are present and must be visited exactly once; keys beyond that are absent
// and must never be visited.
struct visited_once_iff_present
{
  const int* visits;
  int num_keys;

  __device__ bool operator()(int i) const noexcept
  {
    return (i < num_keys) ? (visits[i] == 1) : (visits[i] == 0);
  }
};

struct is_zero
{
  __device__ bool operator()(int value) const noexcept
  {
    return value == 0;
  }
};

// Exercises the device-side `for_each` on the map ref: one cooperative group per query key.
template <class Ref, class KeyIt, class Value>
__global__ void for_each_ref_kernel(Ref ref, KeyIt keys, int num_keys, int* visits)
{
  const auto block     = ::cooperative_groups::this_thread_block();
  const auto thread_id = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);

  if constexpr (Ref::cg_size == 1)
  {
    if (thread_id < num_keys)
    {
      ref.for_each(*(keys + thread_id), record_visit<Value>{visits});
    }
  }
  else
  {
    const auto tile = ::cooperative_groups::tiled_partition<Ref::cg_size, ::cooperative_groups::thread_block>(block);
    const auto idx  = thread_id / Ref::cg_size;
    if (idx < num_keys)
    {
      ref.for_each(tile, *(keys + idx), record_visit<Value>{visits});
    }
  }
}

C2H_TEST("fixed_capacity_map for_each", "[container]", key_types, mapped_types, cg_sizes, bucket_sizes, probing_kinds)
{
  using key_type                             = c2h::get<0, TestType>;
  using mapped_type                          = c2h::get<1, TestType>;
  [[maybe_unused]] constexpr int cg_size     = c2h::get<2, TestType>::value;
  [[maybe_unused]] constexpr int bucket_size = c2h::get<3, TestType>::value;
  [[maybe_unused]] constexpr int probing     = c2h::get<4, TestType>::value;

  using hasher = cuda::hash<key_type>;
  using probing_type =
    ::cuda::std::conditional_t<probing == 0,
                               cudax::cuco::linear_probing<cg_size, hasher>,
                               cudax::cuco::double_hashing<cg_size, hasher>>;
  using map_type = cudax::cuco::fixed_capacity_map<
    key_type,
    mapped_type,
    ::cuda::std::dynamic_extent,
    ::cuda::thread_scope_device,
    ::cuda::std::equal_to<key_type>,
    probing_type,
    bucket_size>;
  using value_type = typename map_type::value_type;

  constexpr int num_keys             = 400;
  constexpr key_type key_sentinel    = key_type{-1};
  constexpr mapped_type val_sentinel = mapped_type{-1};

  ::cuda::stream stream{::cuda::device_ref{0}};
  auto mr = ::cuda::device_default_memory_pool(::cuda::device_ref{0});

  map_type map{stream,
               mr,
               static_cast<::cuda::std::size_t>(num_keys * 2),
               cudax::cuco::empty_key{key_sentinel},
               cudax::cuco::empty_value{val_sentinel}};

  auto pairs = cuda::transform_iterator(cuda::counting_iterator<key_type>{0}, iota_pair<value_type>{});
  map.insert(stream, pairs, pairs + num_keys);

  // Query present keys [0, num_keys) and absent keys [num_keys, 2 * num_keys)
  auto visits = ::cuda::make_buffer<int>(stream, mr, 2 * num_keys, 0);

  map.for_each(stream,
               cuda::counting_iterator<key_type>{0},
               cuda::counting_iterator<key_type>{2 * num_keys},
               record_visit<value_type>{visits.data()});

  REQUIRE(::thrust::all_of(
    ::thrust::cuda::par.on(stream.get()),
    cuda::counting_iterator<int>{0},
    cuda::counting_iterator<int>{2 * num_keys},
    visited_once_iff_present{visits.data(), num_keys}));

  // An empty query range is a no-op
  auto empty_visits = ::cuda::make_buffer<int>(stream, mr, num_keys, 0);
  map.for_each(stream,
               cuda::counting_iterator<key_type>{0},
               cuda::counting_iterator<key_type>{0},
               record_visit<value_type>{empty_visits.data()});

  REQUIRE(::thrust::all_of(
    ::thrust::cuda::par.on(stream.get()), empty_visits.data(), empty_visits.data() + num_keys, is_zero{}));

  // The async overload only enqueues the work; the caller synchronizes
  auto async_visits = ::cuda::make_buffer<int>(stream, mr, 2 * num_keys, 0);
  map.for_each_async(stream,
                     cuda::counting_iterator<key_type>{0},
                     cuda::counting_iterator<key_type>{2 * num_keys},
                     record_visit<value_type>{async_visits.data()});
  stream.sync();

  REQUIRE(::thrust::all_of(
    ::thrust::cuda::par.on(stream.get()),
    cuda::counting_iterator<int>{0},
    cuda::counting_iterator<int>{2 * num_keys},
    visited_once_iff_present{async_visits.data(), num_keys}));

  // The device-side ref API visits the same slots as the host API
  auto ref_visits = ::cuda::make_buffer<int>(stream, mr, 2 * num_keys, 0);
  {
    constexpr int block_size = 128;
    const auto num_threads   = 2 * num_keys * cg_size;
    const auto grid_size     = static_cast<unsigned>((num_threads + block_size - 1) / block_size);

    for_each_ref_kernel<typename map_type::ref_type, cuda::counting_iterator<key_type>, value_type>
      <<<grid_size, block_size, 0, stream.get()>>>(
        map.ref(), cuda::counting_iterator<key_type>{0}, 2 * num_keys, ref_visits.data());
    stream.sync();
  }

  REQUIRE(::thrust::all_of(
    ::thrust::cuda::par.on(stream.get()),
    cuda::counting_iterator<int>{0},
    cuda::counting_iterator<int>{2 * num_keys},
    visited_once_iff_present{ref_visits.data(), num_keys}));

  // After clear no key matches, so the callback is never invoked
  map.clear(stream);
  auto cleared_visits = ::cuda::make_buffer<int>(stream, mr, num_keys, 0);
  map.for_each(stream,
               cuda::counting_iterator<key_type>{0},
               cuda::counting_iterator<key_type>{num_keys},
               record_visit<value_type>{cleared_visits.data()});

  REQUIRE(::thrust::all_of(
    ::thrust::cuda::par.on(stream.get()), cleared_visits.data(), cleared_visits.data() + num_keys, is_zero{}));
}
