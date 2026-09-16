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
 * @brief Run-to-run bit determinism of the sharded reductions and scans on
 *        fractional floating-point data: two runs of every verb from the
 *        same input produce identical bit patterns (`memcmp`). The scans
 *        forward `determinism::run_to_run` to their per-shard CUB scans
 *        (whose environment default is `not_guaranteed`); the cross-shard
 *        folds run in fixed shard order.
 */

#include <cuda/experimental/sharded.cuh>

#include <cstring>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
template <class T>
bool bits_equal(const ::std::vector<T>& a, const ::std::vector<T>& b)
{
  return a.size() == b.size() && ::std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

template <class T>
::std::vector<T> host_of(const sharded_array<T>& a)
{
  ::std::vector<T> h(a.size());
  a.copy_to_host(h.data());
  return h;
}

// Fractional values of mixed magnitude and sign: no association is exact
template <class T>
::std::vector<T> make_input(size_t n)
{
  ::std::vector<T> v(n);
  for (size_t i = 0; i < n; i++)
  {
    v[i] = static_cast<T>((i * 7919) % 1013) / static_cast<T>(97) - static_cast<T>(5.2);
  }
  return v;
}

template <class T>
void test_type(place_group& group, size_t n, cudaStream_t cs)
{
  const auto input = make_input<T>(n);
  auto data        = sharded_array<T>::allocate(group, n);
  const auto envs  = default_envs(data);
  const size_t P   = data.num_shards();
  const auto ce = ::cuda::std::execution::env{::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{cs}}};

  // inclusive_sum, exclusive_sum: two runs from the same input
  data.copy_from_host(input.data());
  inclusive_sum(data);
  const auto inc_1 = host_of(data);
  data.copy_from_host(input.data());
  inclusive_sum(data);
  EXPECT(bits_equal(inc_1, host_of(data)));

  data.copy_from_host(input.data());
  exclusive_sum(data, envs, static_cast<T>(0.3), ce);
  cuda_safe_call(cudaStreamSynchronize(cs));
  barrier(envs);
  const auto exc_1 = host_of(data);
  data.copy_from_host(input.data());
  exclusive_sum(data, envs, static_cast<T>(0.3), ce);
  cuda_safe_call(cudaStreamSynchronize(cs));
  barrier(envs);
  EXPECT(bits_equal(exc_1, host_of(data)));

  // reduce, reduce_into, reduce_into_lanes over the prefix sums
  ::std::vector<T> r(4);
  r[0] = reduce(data, ::cuda::std::plus<T>{}, static_cast<T>(0.1));
  r[1] = reduce(data, ::cuda::std::plus<T>{}, static_cast<T>(0.1));
  EXPECT(::std::memcmp(&r[0], &r[1], sizeof(T)) == 0);

  T* h_out;
  cuda_safe_call(cudaMallocHost(&h_out, sizeof(T)));
  reduce_into(data, envs, h_out, ::cuda::std::plus<T>{}, static_cast<T>(0.1), ce);
  cuda_safe_call(cudaStreamSynchronize(cs));
  r[2] = *h_out;
  reduce_into(data, envs, h_out, ::cuda::std::plus<T>{}, static_cast<T>(0.1), ce);
  cuda_safe_call(cudaStreamSynchronize(cs));
  r[3] = *h_out;
  EXPECT(::std::memcmp(&r[2], &r[3], sizeof(T)) == 0);
  EXPECT(::std::memcmp(&r[0], &r[2], sizeof(T)) == 0); // the same fold, the same bits
  cuda_safe_call(cudaFreeHost(h_out));

  T* h_lanes;
  cuda_safe_call(cudaMallocHost(&h_lanes, P * sizeof(T)));
  reduce_into_lanes(data, envs, h_lanes, ::cuda::std::plus<T>{}, static_cast<T>(0.1));
  barrier(envs);
  ::std::vector<T> lanes_1(h_lanes, h_lanes + P);
  reduce_into_lanes(data, envs, h_lanes, ::cuda::std::plus<T>{}, static_cast<T>(0.1));
  barrier(envs);
  EXPECT(bits_equal(lanes_1, ::std::vector<T>(h_lanes, h_lanes + P)));
  for (size_t g = 0; g < P; g++)
  {
    EXPECT(::std::memcmp(&lanes_1[g], &r[0], sizeof(T)) == 0);
  }
  cuda_safe_call(cudaFreeHost(h_lanes));
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group{make_locality_domain_grid()};
  cudaStream_t cs;
  cuda_safe_call(cudaStreamCreate(&cs));

  for (const size_t n : {size_t{65537}, (size_t{1} << 20) + 37})
  {
    test_type<float>(group, n, cs);
    test_type<double>(group, n, cs);
  }

  cuda_safe_call(cudaStreamDestroy(cs));
  return 0;
}
