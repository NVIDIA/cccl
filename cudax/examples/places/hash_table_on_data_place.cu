//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Put a hash table on a specific data place.
 *
 * The smallest placement example: one `cuda::experimental::cuco::
 * fixed_capacity_map`, whose slots live on a `data_place` of our choosing —
 * here one locality domain of device 0 — built and probed from a stream of the
 * matching `exec_place`, so the table is written and read by the SMs of the
 * domain that owns its memory.
 *
 * The only line that places the table is the memory resource handed to the
 * map: `place_memory_resource{place}`. `fixed_capacity_map`'s `_MemoryResource`
 * template parameter accepts it as-is (the resource is `device_accessible`).
 * Change `place` to `data_place::device(0)`, to another domain, or to a
 * composite place spread over several domains, and nothing else changes.
 *
 * No sharded arrays, no `for_each_shard`: this is the rung below
 * `sharded_hash_per_domain.cu`, which builds one such table per domain.
 *
 * Data: key i is the odd number 2i+1 with payload i. Probe i asks for 2i+1 on
 * odd i (hit) and for 2i on even i (miss): exactly n/2 hits.
 */

#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/reduce.h>

#include <cuda/experimental/__cuco/fixed_capacity_map.cuh>
#include <cuda/experimental/places.cuh>

#include <cstddef>
#include <cstdio>

using namespace cuda::experimental::places;
namespace xcuco = cuda::experimental::cuco;

struct key_value_of
{
  __host__ __device__ ::cuda::std::pair<int, int> operator()(::std::size_t i) const
  {
    return {static_cast<int>(2 * i + 1), static_cast<int>(i)};
  }
};
struct probe_of
{
  __host__ __device__ int operator()(::std::size_t i) const
  {
    return static_cast<int>((i & 1) ? 2 * i + 1 : 2 * i);
  }
};

int main()
{
  // Pick the place: the last locality domain of device 0. On a device without
  // locality domains there is exactly one, standing for the whole device.
  const int domain       = static_cast<int>(locality_domain_count(0)) - 1;
  const data_place place = data_place::locality_domain(0, domain);
  const exec_place where = exec_place::locality_domain(0, domain);

  // A stream born in the execution place's context: kernels launched into it
  // run on that domain's SMs.
  exec_place_resources resources;
  const cudaStream_t stream = where.pick_stream(resources);
  const ::cuda::stream_ref s{stream};

  printf("table storage: %s   compute: %s\n", place.to_string().c_str(), where.to_string().c_str());

  // The table. The memory resource is the placement.
  using map_t = xcuco::fixed_capacity_map<
    int,
    int,
    ::cuda::std::dynamic_extent,
    ::cuda::thread_scope_device,
    ::cuda::std::equal_to<int>,
    xcuco::linear_probing<4, ::cuda::hash<int>>,
    1,
    place_memory_resource>;
  const ::std::size_t n = ::std::size_t{1} << 22;
  map_t table{
    s, place_memory_resource{place}, /*capacity*/ 2 * n, xcuco::empty_key<int>{-1}, xcuco::empty_value<int>{-1}};

  // Build: n rows straight from the index, no key array to allocate.
  auto rows = thrust::make_transform_iterator(thrust::make_counting_iterator(::std::size_t{0}), key_value_of{});
  table.insert_async(s, rows, rows + n);

  // Probe: the flags land on the same place as the table.
  auto probes = thrust::make_transform_iterator(thrust::make_counting_iterator(::std::size_t{0}), probe_of{});
  int* found  = static_cast<int*>(place.allocate(static_cast<::std::ptrdiff_t>(n * sizeof(int)), stream));
  table.contains_async(s, probes, probes + n, found);

  const long long hits = thrust::reduce(thrust::cuda::par.on(stream), found, found + n, 0LL);
  place.deallocate(found, n * sizeof(int), stream);
  cuda_safe_call(cudaStreamSynchronize(stream));

  printf(
    "capacity %zu, %zu rows, %lld hits (expected %zu)\n", static_cast<::std::size_t>(table.capacity()), n, hits, n / 2);
  if (hits != static_cast<long long>(n / 2))
  {
    printf("FAILED\n");
    return 1;
  }
  printf("PASSED\n");
  return 0;
}
