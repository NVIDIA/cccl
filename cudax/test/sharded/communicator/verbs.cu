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
 * @brief Correctness of the places-communicator verbs over locality domains:
 *        send/recv rendezvous, all_gather, and the fixed-order all_reduce
 *        (including its bit-determinism across repeated runs).
 */

#include <cuda/experimental/sharded.cuh>

#include <cstring>
#include <random>
#include <vector>

using namespace cuda::experimental::sharded;
using cuda::experimental::places::exec_place_scope;
using cuda::experimental::places::place_group;

namespace
{
// Per-place device buffer helpers (allocated from the place's affine data
// place on the group stream, synchronously usable afterwards).
template <typename T>
T* place_alloc(place_group& group, size_t idx, size_t count)
{
  auto dp = group.place(idx).affine_data_place();
  exec_place_scope scope(group.place(idx));
  T* ptr = static_cast<T*>(dp.allocate(static_cast<ptrdiff_t>(count * sizeof(T)), group.get_stream(idx)));
  cuda_safe_call(cudaStreamSynchronize(group.get_stream(idx)));
  return ptr;
}

template <typename T>
void place_free(place_group& group, size_t idx, T* ptr, size_t count)
{
  auto dp = group.place(idx).affine_data_place();
  exec_place_scope scope(group.place(idx));
  dp.deallocate(ptr, count * sizeof(T), group.get_stream(idx));
  cuda_safe_call(cudaStreamSynchronize(group.get_stream(idx)));
}

template <typename T>
void h2d(place_group& group, size_t idx, T* dst, const T* src, size_t count)
{
  exec_place_scope scope(group.place(idx));
  cuda_safe_call(cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyHostToDevice, group.get_stream(idx)));
  cuda_safe_call(cudaStreamSynchronize(group.get_stream(idx)));
}

template <typename T>
void d2h(place_group& group, size_t idx, T* dst, const T* src, size_t count)
{
  exec_place_scope scope(group.place(idx));
  cuda_safe_call(cudaMemcpyAsync(dst, src, count * sizeof(T), cudaMemcpyDeviceToHost, group.get_stream(idx)));
  cuda_safe_call(cudaStreamSynchronize(group.get_stream(idx)));
}

void test_send_recv(place_group& group, ::std::vector<places_communicator>& comms)
{
  const size_t n = 4096;
  ::std::vector<float> host(n);
  for (size_t i = 0; i < n; i++)
  {
    host[i] = static_cast<float>(i) * 0.5f;
  }

  float* src = place_alloc<float>(group, 0, n);
  float* dst = place_alloc<float>(group, 1, n);
  h2d(group, 0, src, host.data(), n);

  {
    auto guard = comms[0].group_guard();
    comms[0].send(guard, src, n * sizeof(float), /*peer=*/1, cuda::stream_ref{group.get_stream(0)});
    comms[1].recv(guard, dst, n * sizeof(float), /*peer=*/0, cuda::stream_ref{group.get_stream(1)});
  }

  ::std::vector<float> got(n);
  d2h(group, 1, got.data(), dst, n);
  EXPECT(::std::memcmp(got.data(), host.data(), n * sizeof(float)) == 0);

  place_free(group, 0, src, n);
  place_free(group, 1, dst, n);
}

void test_all_gather(place_group& group, ::std::vector<places_communicator>& comms)
{
  const int nranks = static_cast<int>(group.size());
  const size_t n   = 1024;

  ::std::vector<float*> send(nranks), recv(nranks);
  ::std::vector<float> host(n * nranks);
  for (int r = 0; r < nranks; r++)
  {
    for (size_t i = 0; i < n; i++)
    {
      host[static_cast<size_t>(r) * n + i] = static_cast<float>(r * 1000) + static_cast<float>(i);
    }
    send[r] = place_alloc<float>(group, r, n);
    recv[r] = place_alloc<float>(group, r, n * nranks);
    h2d(group, r, send[r], host.data() + static_cast<size_t>(r) * n, n);
  }

  {
    auto guard = comms[0].group_guard();
    for (int r = 0; r < nranks; r++)
    {
      comms[r].all_gather(guard, send[r], recv[r], n, cuda::stream_ref{group.get_stream(r)});
    }
  }

  // Every rank sees every rank's slot, in rank order.
  for (int r = 0; r < nranks; r++)
  {
    ::std::vector<float> got(n * nranks);
    d2h(group, r, got.data(), recv[r], n * nranks);
    EXPECT(::std::memcmp(got.data(), host.data(), n * nranks * sizeof(float)) == 0);
  }

  for (int r = 0; r < nranks; r++)
  {
    place_free(group, r, send[r], n);
    place_free(group, r, recv[r], n * nranks);
  }
}

void test_all_reduce(place_group& group, ::std::vector<places_communicator>& comms)
{
  const int nranks = static_cast<int>(group.size());
  const size_t n   = 1 << 16;

  // Random fp32 partials: the interesting case for bit-determinism.
  ::std::mt19937 rng(42);
  ::std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  ::std::vector<::std::vector<float>> host(nranks, ::std::vector<float>(n));
  ::std::vector<double> ref(n, 0.0);
  for (int r = 0; r < nranks; r++)
  {
    for (size_t i = 0; i < n; i++)
    {
      host[r][i] = dist(rng);
      ref[i] += static_cast<double>(host[r][i]);
    }
  }

  ::std::vector<float*> send(nranks), recv(nranks);
  for (int r = 0; r < nranks; r++)
  {
    send[r] = place_alloc<float>(group, r, n);
    recv[r] = place_alloc<float>(group, r, n);
    h2d(group, r, send[r], host[r].data(), n);
  }

  auto run = [&]() {
    auto guard = comms[0].group_guard();
    for (int r = 0; r < nranks; r++)
    {
      comms[r].all_reduce(guard, send[r], recv[r], n, cuda::std::plus<>{}, cuda::stream_ref{group.get_stream(r)});
    }
  };

  // Correctness: every rank receives the same sum, close to the fp64
  // reference; ranks agree bitwise (one fold kernel writes all of them).
  run();
  ::std::vector<::std::vector<float>> got(nranks, ::std::vector<float>(n));
  for (int r = 0; r < nranks; r++)
  {
    d2h(group, r, got[r].data(), recv[r], n);
  }
  for (size_t i = 0; i < n; i++)
  {
    EXPECT(::std::abs(static_cast<double>(got[0][i]) - ref[i]) <= 1e-5 * (ref[i] + 1.0));
  }
  for (int r = 1; r < nranks; r++)
  {
    EXPECT(::std::memcmp(got[r].data(), got[0].data(), n * sizeof(float)) == 0);
  }

  // Bit-determinism: the fold order is fixed (rank 0, 1, ..., n-1), so
  // repeated runs must be bit-identical.
  ::std::vector<float> first = got[0];
  for (int rep = 0; rep < 5; rep++)
  {
    run();
    ::std::vector<float> again(n);
    d2h(group, 0, again.data(), recv[0], n);
    EXPECT(::std::memcmp(again.data(), first.data(), n * sizeof(float)) == 0);
  }

  for (int r = 0; r < nranks; r++)
  {
    place_free(group, r, send[r], n);
    place_free(group, r, recv[r], n);
  }
}
} // namespace

int main()
{
  cuda_safe_call(cudaSetDevice(0));

  auto group = place_group::by_locality_domains();
  if (group.size() < 2)
  {
    // The verbs are cross-place by nature; nothing to exercise on one place.
    return 0;
  }

  auto comms = make_communicators(group);

  test_send_recv(group, comms);
  test_all_gather(group, comms);
  test_all_reduce(group, comms);

  return 0;
}
