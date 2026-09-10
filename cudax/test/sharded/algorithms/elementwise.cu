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
 * @brief Correctness of the elementwise sharded algorithms (fill, sequence,
 *        iota, tabulate, generate, for_each, transform) against host
 *        references, over multiple places; and the composition contract of
 *        the asynchronous `zip_transform` (lane-ordered by default, sealed
 *        only under `composition::bracketed`).
 */

#include <cuda/stream>

#include <cuda/experimental/sharded.cuh>

#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::make_locality_domain_grid;
using cuda::experimental::places::place_group;

namespace
{
struct times_two_plus_index
{
  __host__ __device__ long long operator()(size_t i) const
  {
    return 2 * static_cast<long long>(i) + 7;
  }
};

struct negate_op
{
  __host__ __device__ long long operator()(long long x) const
  {
    return -x;
  }
};

struct saxpy_op
{
  __host__ __device__ long long operator()(long long x, long long y) const
  {
    return 3 * x + y;
  }
};

struct set_to_index
{
  __host__ __device__ void operator()(long long& v, size_t i) const
  {
    v += static_cast<long long>(i);
  }
};

struct const_gen
{
  __host__ __device__ long long operator()() const
  {
    return 42;
  }
};

void test_fill_and_sequence(place_group& group)
{
  const size_t n = 100003;
  auto data      = sharded_array<long long>::allocate(group, n);

  fill(data, 17LL);
  ::std::vector<long long> host(n);
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == 17LL);
  }

  sequence(data, default_envs(data), 5LL, 3LL); // 5, 8, 11, ...
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == 5LL + 3LL * static_cast<long long>(i));
  }

  iota(data, 100LL);
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == 100LL + static_cast<long long>(i));
  }
}

void test_tabulate_generate_for_each(place_group& group)
{
  const size_t n = 65537;
  auto data      = sharded_array<long long>::allocate(group, n);

  tabulate(data, times_two_plus_index{});
  ::std::vector<long long> host(n);
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == 2 * static_cast<long long>(i) + 7);
  }

  generate(data, const_gen{});
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == 42LL);
  }

  // for_each sees the GLOBAL index: 42 + i
  for_each(data, set_to_index{});
  data.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == 42LL + static_cast<long long>(i));
  }
}

void test_transform(place_group& group)
{
  const size_t n = 50000;
  auto a         = sharded_array<long long>::allocate(group, n);
  iota(a, 0LL);

  // In-place
  transform(a, negate_op{});
  ::std::vector<long long> host(n);
  a.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == -static_cast<long long>(i));
  }

  // Unary out-of-place
  auto b = sharded_array<long long>::allocate_like(a);
  zip_transform(b, negate_op{}, a);
  b.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == static_cast<long long>(i));
  }

  // Binary: c = 3*a + b = -3i + i = -2i
  auto c = sharded_array<long long>::allocate_like(a);
  zip_transform(c, saxpy_op{}, a, b);
  c.copy_to_host(host.data());
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(host[i] == -2 * static_cast<long long>(i));
  }

  // Incompatible layouts must throw
  auto other = sharded_array<long long>::allocate({{n / 2, data_place::device(0), exec_place::device(0), nullptr}});
  bool threw = false;
  try
  {
    zip_transform(other, negate_op{}, a);
  }
  catch (const ::std::invalid_argument&)
  {
    threw = true;
  }
  EXPECT(threw);
}

struct axpb_op
{
  __host__ __device__ double operator()(double x) const
  {
    return 1.0001 * x + 1.0;
  }
};

struct add_op
{
  __host__ __device__ double operator()(double x, double y) const
  {
    return x + y;
  }
};

// The composition contract of the asynchronous N-ary form. Lane-ordered
// (the default): a call enqueues each shard's work on its environment's
// stream and touches NOTHING else — in particular it puts no work and no
// dependency on the call stream, so an event recorded on the call stream
// right after the call completes immediately even while the lanes are still
// busy (waiting for the event returns at once, then the lanes are still
// found busy). Bracketed: the call stream waits for every lane (join), so
// the same event cannot complete before the lanes finish. Results are
// identical.
void test_zip_transform_lane_ordered(place_group& group)
{
  const size_t n = size_t{1} << 26; // 512 MiB per array: lanes stay busy for milliseconds
  auto a         = sharded_array<double>::allocate(group, n);
  auto b         = sharded_array<double>::allocate(group, n);
  auto c         = sharded_array<double>::allocate(group, n);
  fill(a, 1.0);
  fill(b, 2.0);
  fill(c, 0.0);
  auto envs = default_envs(c);

  cudaStream_t call_stream;
  cuda_safe_call(cudaStreamCreateWithFlags(&call_stream, cudaStreamNonBlocking));
  cudaEvent_t ev;
  cuda_safe_call(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming));

  const auto stream_prop  = ::cuda::std::execution::prop{::cuda::get_stream, ::cuda::stream_ref{call_stream}};
  const auto bracket_prop = ::cuda::std::execution::prop{get_composition_t{}, composition::bracketed};
  const auto ce_lanes     = ::cuda::std::execution::env{stream_prop};
  const auto ce_bracketed = ::cuda::std::execution::env{stream_prop, bracket_prop};

  constexpr int busy_iters = 32; // ~32 GiB of traffic: several milliseconds of lane work
  auto make_busy           = [&]() {
    for (int i = 0; i < busy_iters; i++)
    {
      transform(a, envs, axpb_op{}, ce_lanes); // lane-ordered; only the lanes work
    }
  };
  // Restore a = 1 after the busy loop so both variants compute the same thing
  auto reset_a = [&]() {
    barrier(envs);
    fill(a, 1.0);
  };

  // --- lane-ordered: two back-to-back calls, no edge to the call stream ---
  make_busy();
  zip_transform(c, envs, add_op{}, ce_lanes, a, b); // c = a + b (a in stream order after the busy loop)
  zip_transform(c, envs, add_op{}, ce_lanes, c, b); // c = a + 2b
  cuda_safe_call(cudaEventRecord(ev, call_stream));
  // The call stream carries nothing from those calls: waiting for the event
  // returns immediately (the record is the only command on that stream)...
  cuda_safe_call(cudaEventSynchronize(ev));
  // ...while the lanes are still busy (the busy loop dwarfs the host latency)
  bool lanes_busy = false;
  for (size_t g = 0; g < envs.size(); g++)
  {
    lanes_busy |= (cudaStreamQuery(::cuda::get_stream(envs[g]).get()) == cudaErrorNotReady);
  }
  EXPECT(lanes_busy);
  barrier(envs);
  cuda_safe_call(cudaGetLastError());
  ::std::vector<double> host(n);
  c.copy_to_host(host.data());
  double a_final = 1.0;
  for (int i = 0; i < busy_iters; i++)
  {
    a_final = 1.0001 * a_final + 1.0;
  }
  for (size_t i = 0; i < n; i += 4093)
  {
    EXPECT(host[i] == a_final + 4.0);
  }

  // --- bracketed: the call stream joins the lanes; the event must wait ---
  reset_a();
  make_busy();
  zip_transform(c, envs, add_op{}, ce_bracketed, a, b);
  zip_transform(c, envs, add_op{}, ce_bracketed, c, b);
  cuda_safe_call(cudaEventRecord(ev, call_stream));
  EXPECT(cudaEventQuery(ev) == cudaErrorNotReady);
  cuda_safe_call(cudaEventSynchronize(ev)); // completing the event completes the lanes' work too
  for (size_t g = 0; g < envs.size(); g++)
  {
    EXPECT(cudaStreamQuery(::cuda::get_stream(envs[g]).get()) == cudaSuccess);
  }
  c.copy_to_host(host.data());
  for (size_t i = 0; i < n; i += 4093)
  {
    EXPECT(host[i] == a_final + 4.0);
  }

  cuda_safe_call(cudaEventDestroy(ev));
  cuda_safe_call(cudaStreamDestroy(call_stream));
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group{make_locality_domain_grid()};

  test_fill_and_sequence(group);
  test_tabulate_generate_for_each(group);
  test_transform(group);
  test_zip_transform_lane_ordered(group);

  return 0;
}
