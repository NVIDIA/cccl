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
 * @brief Regression: a translation unit that includes `<cuda/std/execution>`
 *        ahead of the sharded headers and drives CUB's environment dispatch
 *        with a `place_memory_resource` — through the sharded scans in their
 *        synchronous and stream-bearing forms, and through the MGMN-engine
 *        reference transform (`reserved::mgmn_engine`) — must
 *        compile under relocatable device code. CUB reaches the resource
 *        from `CUB_RUNTIME_FUNCTION` (`__host__ __device__`) code there; a
 *        host-only `allocate`/`deallocate` was nvcc error #20011.
 */

#include <cuda/std/execution>

#include <cuda/experimental/__sharded/mgmn_transform.cuh>
#include <cuda/experimental/sharded.cuh>

#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;
namespace mgmn_engine = cuda::experimental::sharded::reserved::mgmn_engine;

namespace
{
struct twice_op
{
  __host__ __device__ long long operator()(long long x) const
  {
    return 2 * x;
  }
};
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group     = place_group{make_locality_domain_grid()};
  const size_t n = 65537;
  auto data      = sharded_array<long long>::allocate(group, n);
  iota(data, 0LL);

  cudaStream_t origin;
  cuda_safe_call(cudaStreamCreate(&origin));
  const auto ce =
    ::cuda::std::execution::env{::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{origin}}};
  const auto envs = default_envs(data);

  // The stream-bearing forms, lane-ordered behind a fork/join
  data.fork_from(origin);
  mgmn_engine::transform(data, envs, twice_op{}, ce);
  inclusive_sum(data, envs, ce);
  inclusive_scan(data, envs, ::cuda::std::plus<long long>{}, 0LL, ce);
  data.join_into(origin);
  cuda_safe_call(cudaStreamSynchronize(origin));

  // The synchronous form on the same array
  inclusive_sum(data);

  ::std::vector<long long> h(n);
  data.copy_to_host(h.data());
  // x_i = 2i; three inclusive prefix sums of x
  ::std::vector<long long> ref(n);
  for (size_t i = 0; i < n; i++)
  {
    ref[i] = 2 * static_cast<long long>(i);
  }
  for (int pass = 0; pass < 3; pass++)
  {
    for (size_t i = 1; i < n; i++)
    {
      ref[i] += ref[i - 1];
    }
  }
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(h[i] == ref[i]);
  }

  cuda_safe_call(cudaStreamDestroy(origin));
  return 0;
}
