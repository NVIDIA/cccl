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
 * @brief The concept-generic algorithm tier (pilot: transform, reduce)
 *        agrees with the container tier, supports explicit and derived
 *        (self-bound) environments, honors the asynchronous call-environment
 *        contract, and refuses cleanly under sync_policy::forbid.
 */

#include <cuda/experimental/sharded.cuh>

#include <stdexcept>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::place_group;

namespace
{
struct times2
{
  __host__ __device__ long long operator()(long long v) const
  {
    return v * 2;
  }
};
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));
  auto group     = place_group::by_locality_domains();
  const size_t n = 1000001;
  auto a         = sharded_array<long long>::allocate(group, n);
  auto b         = sharded_array<long long>::allocate(group, n);
  iota(group, a, 1LL);
  iota(group, b, 1LL);
  const long long tri = (long long) n * ((long long) n + 1) / 2;

  // Container-tier reference
  transform(group, a, times2{}); // a = 2..2n
  EXPECT(sum(group, a) == 2 * tri);

  // Generic tier, self-bound synchronous form
  transform(b, times2{}); // b = 2..2n
  EXPECT(reduce(b, ::cuda::std::plus<long long>{}, 0LL) == 2 * tri);

  // Generic tier, explicit environments
  auto envs = default_envs(b);
  transform(b, envs, times2{}); // b = 4..4n
  EXPECT(reduce(b, envs, ::cuda::std::plus<long long>{}, 0LL) == 4 * tri);

  // Asynchronous form: ordered against a caller stream, no host sync inside
  cudaStream_t call_stream;
  cuda_safe_call(cudaStreamCreate(&call_stream));
  const auto stream_prop = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{call_stream}};
  const auto call_env    = ::cuda::std::execution::env{stream_prop};
  transform(b, times2{}, call_env); // b = 8..8n, on call_stream's timeline
  cuda_safe_call(cudaStreamSynchronize(call_stream));
  EXPECT(reduce(b, ::cuda::std::plus<long long>{}, 0LL) == 8 * tri);

  // forbid policy: the synchronous form refuses cleanly, state stays valid
  const auto forbid_prop = ::cuda::std::execution::prop{get_sync_policy_t{}, sync_policy::forbid};
  const auto forbid_env  = ::cuda::std::execution::env{forbid_prop};
  bool threw             = false;
  try
  {
    (void) reduce(b, ::cuda::std::plus<long long>{}, 0LL, forbid_env);
  }
  catch (const ::std::runtime_error&)
  {
    threw = true;
  }
  EXPECT(threw);
  EXPECT(reduce(b, ::cuda::std::plus<long long>{}, 0LL) == 8 * tri);

  cuda_safe_call(cudaStreamDestroy(call_stream));
  return 0;
}
