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
 * @brief Parity of the live sharded verbs (MGMN engines) against the legacy
 *        reference implementations of `__sharded/reference/`: `reduce`,
 *        `reduce_into`, `reduce_into_lanes`, the scans and the sums are
 *        compared BITWISE (bit patterns, never floating-point
 *        `==`) on the locality-domain group and on a single-place group, at
 *        small, non-divisible and large sizes (empty shards included), for
 *        `int` and `double`, with known-identity and custom operators and
 *        non-trivial initial values. Floating-point inputs are
 *        integer-valued, so every association of the fold is exact and the
 *        comparison is meaningful; the run-to-run determinism of the live
 *        verbs on fractional data is the subject of `bit_determinism.cu`.
 *        (The transforms are direct per-shard launches; their parity with
 *        the MGMN-engine reference is `engine_parity.cu`.)
 */

#include <cuda/experimental/__sharded/reference/legacy_reduce.cuh>
#include <cuda/experimental/__sharded/reference/legacy_scan.cuh>
#include <cuda/experimental/sharded.cuh>

#include <cstring>
#include <limits>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::exec_place;
using cuda::experimental::places::place_group;
namespace legacy = cuda::experimental::sharded::reserved::legacy;

namespace
{
// Custom operators: no `cuda::identity_element`, so the live reduce runs
// its lifted engine path and the scans take an explicit identity.
template <class T>
struct max_fn
{
  __host__ __device__ T operator()(T a, T b) const
  {
    return a > b ? a : b;
  }
};

template <class T>
struct sum_fn
{
  __host__ __device__ T operator()(T a, T b) const
  {
    return a + b;
  }
};

template <class T>
bool bits_equal(const ::std::vector<T>& a, const ::std::vector<T>& b)
{
  return a.size() == b.size() && (a.empty() || ::std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0);
}

template <class T>
bool bits_equal(const T& a, const T& b)
{
  return ::std::memcmp(&a, &b, sizeof(T)) == 0;
}

template <class T>
::std::vector<T> host_of(const sharded_array<T>& a)
{
  ::std::vector<T> h(a.size());
  a.copy_to_host(h.data());
  return h;
}

// Small-magnitude integer-valued inputs (exact under any association)
template <class T>
::std::vector<T> make_input(size_t n, int seed)
{
  ::std::vector<T> v(n);
  for (size_t i = 0; i < n; i++)
  {
    v[i] = static_cast<T>(static_cast<long long>((i * 7919 + static_cast<size_t>(seed) * 104729) % 101) - 50);
  }
  return v;
}

//! Live vs legacy for one (op, init, identity) triple on the same input.
template <class T, class Op>
void compare_reduce_family(place_group& group, size_t n, Op op, T init, cudaStream_t cs)
{
  const auto input = make_input<T>(n, 1);
  auto live        = sharded_array<T>::allocate(group, n);
  auto ref         = sharded_array<T>::allocate(group, n);
  live.copy_from_host(input.data());
  ref.copy_from_host(input.data());
  const auto envs_live = default_envs(live);
  const auto envs_ref  = default_envs(ref);
  const size_t P       = live.num_shards();
  const auto ce = ::cuda::std::execution::env{::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{cs}}};

  // reduce (self-bound and explicit environments)
  {
    const T a = reduce(live, op, init);
    const T b = legacy::reduce(ref, op, init);
    EXPECT(bits_equal(a, b));
    const T c = reduce(live, envs_live, op, init);
    EXPECT(bits_equal(c, b));
  }

  // reduce_into on the call stream (pinned output)
  {
    T* h_live;
    T* h_ref;
    cuda_safe_call(cudaMallocHost(&h_live, sizeof(T)));
    cuda_safe_call(cudaMallocHost(&h_ref, sizeof(T)));
    reduce_into(live, envs_live, h_live, op, init, ce);
    legacy::reduce_into(ref, envs_ref, h_ref, op, init, ce);
    cuda_safe_call(cudaStreamSynchronize(cs));
    EXPECT(bits_equal(*h_live, *h_ref));
    cuda_safe_call(cudaFreeHost(h_live));
    cuda_safe_call(cudaFreeHost(h_ref));
  }

  // reduce_into_lanes (pinned outputs, one per lane)
  if (P > 0)
  {
    T* h_live;
    T* h_ref;
    cuda_safe_call(cudaMallocHost(&h_live, P * sizeof(T)));
    cuda_safe_call(cudaMallocHost(&h_ref, P * sizeof(T)));
    reduce_into_lanes(live, envs_live, h_live, op, init);
    legacy::reduce_into_lanes(ref, envs_ref, h_ref, op, init);
    barrier(envs_live);
    barrier(envs_ref);
    for (size_t g = 0; g < P; g++)
    {
      EXPECT(bits_equal(h_live[g], h_ref[g]));
      EXPECT(bits_equal(h_live[g], h_live[0]));
    }
    cuda_safe_call(cudaFreeHost(h_live));
    cuda_safe_call(cudaFreeHost(h_ref));
  }
}

//! Live vs legacy scans for one operator with an explicit identity.
template <class T, class Op>
void compare_scans(place_group& group, size_t n, Op op, T init, T identity)
{
  const auto input = make_input<T>(n, 2);
  auto live        = sharded_array<T>::allocate(group, n);
  auto ref         = sharded_array<T>::allocate(group, n);

  live.copy_from_host(input.data());
  ref.copy_from_host(input.data());
  inclusive_scan(live, op, identity);
  legacy::inclusive_scan(ref, op, identity);
  EXPECT(bits_equal(host_of(live), host_of(ref)));

  live.copy_from_host(input.data());
  ref.copy_from_host(input.data());
  exclusive_scan(live, default_envs(live), op, init, identity);
  legacy::exclusive_scan(ref, default_envs(ref), op, init, identity);
  EXPECT(bits_equal(host_of(live), host_of(ref)));
}

//! Live vs legacy sums (known identity: the direct engine path).
template <class T>
void compare_sums(place_group& group, size_t n, T init)
{
  const auto input = make_input<T>(n, 3);
  auto live        = sharded_array<T>::allocate(group, n);
  auto ref         = sharded_array<T>::allocate(group, n);

  live.copy_from_host(input.data());
  ref.copy_from_host(input.data());
  inclusive_sum(live);
  legacy::inclusive_sum(ref);
  EXPECT(bits_equal(host_of(live), host_of(ref)));

  live.copy_from_host(input.data());
  ref.copy_from_host(input.data());
  exclusive_sum(live, init);
  legacy::exclusive_sum(ref, init);
  EXPECT(bits_equal(host_of(live), host_of(ref)));

  // The conveniences
  live.copy_from_host(input.data());
  ref.copy_from_host(input.data());
  EXPECT(bits_equal(sum(live), legacy::sum(ref)));
  EXPECT(bits_equal(min(live), legacy::min(ref)));
  EXPECT(bits_equal(max(live), legacy::max(ref)));
}

template <class T>
void run_type(place_group& group, size_t n, cudaStream_t cs)
{
  const T lowest = ::std::numeric_limits<T>::lowest();
  // Known identity (direct path), non-trivial init
  compare_reduce_family<T>(group, n, ::cuda::std::plus<T>{}, static_cast<T>(17), cs);
  compare_reduce_family<T>(group, n, ::cuda::maximum<T>{}, static_cast<T>(-3), cs);
  // Custom operators (lifted path)
  compare_reduce_family<T>(group, n, max_fn<T>{}, static_cast<T>(-3), cs);
  compare_reduce_family<T>(group, n, sum_fn<T>{}, static_cast<T>(17), cs);
  compare_reduce_family<T>(group, n, max_fn<T>{}, static_cast<T>(5000), cs); // init dominates

  compare_scans<T>(group, n, ::cuda::std::plus<T>{}, static_cast<T>(7), static_cast<T>(0));
  compare_scans<T>(group, n, max_fn<T>{}, static_cast<T>(-9), lowest);
  compare_scans<T>(group, n, sum_fn<T>{}, static_cast<T>(7), static_cast<T>(0));
  compare_sums<T>(group, n, static_cast<T>(11));
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto domains = place_group{exec_place::all_locality_domains()};
  auto single  = place_group{exec_place::device(0)};
  EXPECT(single.size() == 1);

  cudaStream_t cs;
  cuda_safe_call(cudaStreamCreate(&cs));

  for (place_group* group : {&domains, &single})
  {
    for (const size_t n :
         {size_t{0},
          size_t{1},
          size_t{2},
          size_t{3},
          size_t{7},
          size_t{1000},
          size_t{4097},
          size_t{65537},
          size_t{262147},
          (size_t{1} << 20) + 37})
    {
      run_type<int>(*group, n, cs);
      run_type<double>(*group, n, cs);
    }
  }

  cuda_safe_call(cudaStreamDestroy(cs));
  return 0;
}
