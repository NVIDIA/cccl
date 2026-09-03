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

struct twice_index
{
  __host__ __device__ long long operator()(size_t i) const
  {
    return static_cast<long long>(2 * i);
  }
};

struct add_index
{
  __host__ __device__ void operator()(long long& e, size_t i) const
  {
    e += static_cast<long long>(i);
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

  // Elementwise family, generic self-bound forms
  const long long tri0 = (long long) (n - 1) * (long long) n / 2; // sum of 0..n-1
  fill(b, 7LL);
  EXPECT(reduce(b, ::cuda::std::plus<long long>{}, 0LL) == 7 * (long long) n);
  iota(b, 0LL);
  EXPECT(reduce(b, ::cuda::std::plus<long long>{}, 0LL) == tri0);
  tabulate(b, twice_index{});
  EXPECT(reduce(b, ::cuda::std::plus<long long>{}, 0LL) == 2 * tri0);
  for_each(b, add_index{}); // b[i] = 2i + i = 3i
  EXPECT(reduce(b, ::cuda::std::plus<long long>{}, 0LL) == 3 * tri0);

  // Elementwise family, generic explicit-envs forms
  fill(b, envs, 1LL);
  EXPECT(reduce(b, envs, ::cuda::std::plus<long long>{}, 0LL) == (long long) n);

  // Generic adjacent_difference: boundary crossing between shards must see
  // the predecessor's last element; iota -> minus gives all-1s except [0].
  iota(b, 5LL); // b = 5, 6, ..., 5 + n - 1
  adjacent_difference(b, envs, a, ::cuda::std::minus<long long>{});
  // a[0] = 5, a[i>0] = 1  =>  sum = 5 + (n - 1)
  EXPECT(reduce(a, ::cuda::std::plus<long long>{}, 0LL) == 5 + (long long) n - 1);
  // Self-bound overload, and a second op to vary the boundary value
  adjacent_difference(b, a, ::cuda::std::plus<long long>{});
  // a[0] = 5, a[i>0] = b[i] + b[i-1] = 2*(5+i) - 1 => sum = 5 + sum_{i=1..n-1}(2i + 9)
  {
    const long long m = (long long) n - 1;
    EXPECT(reduce(a, ::cuda::std::plus<long long>{}, 0LL) == 5 + m * (m + 1) + 9 * m);
  }
  // Aliasing refused
  {
    bool threw = false;
    try
    {
      adjacent_difference(b, envs, b, ::cuda::std::minus<long long>{});
    }
    catch (const ::std::invalid_argument&)
    {
      threw = true;
    }
    EXPECT(threw);
  }
  fill(b, envs, 1LL); // restore the state the zip block below expects

  // Out-of-place elementwise: zip_transform is the generic spelling (a plain
  // `transform(in, out, op)` overload would collide with cuda::std::transform
  // through ADL when cuda::std functors are involved — a recorded design
  // decision, not an omission)
  zip_transform(a, times2{}, b); // a[i] = 2 * b[i] = 2
  EXPECT(reduce(a, ::cuda::std::plus<long long>{}, 0LL) == 2 * (long long) n);
  zip_transform(a, ::cuda::std::plus<long long>{}, b, a); // a = b + a = 3
  EXPECT(reduce(a, ::cuda::std::plus<long long>{}, 0LL) == 3 * (long long) n);

  // Restore b for the arms below (they expect b == 8..8n trajectory continuity
  // is not required; recompute expectations locally instead)
  iota(group, b, 1LL);
  transform(b, times2{});
  transform(b, times2{});
  transform(b, times2{}); // b = 8..8n
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
