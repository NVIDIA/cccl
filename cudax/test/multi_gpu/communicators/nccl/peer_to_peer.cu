//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/devices>
#include <cuda/std/cstdint>

#include <vector>

#include "nccl_test_helpers.cuh"
#include <c2h/vector.h>

namespace
{
struct payload
{
  cuda::std::int32_t from;
  cuda::std::int32_t index;
};
} // namespace

// Ring exchange via send/recv. Rank r contributes {r, r, r}.
NCCL_COMM_TEST("nccl_communicator send/recv ring")
{
  if (cuda::devices.size() == 1)
  {
    // NCCL disallows self send/recv on a 1-rank comm.
    REQUIRE(this->communicators().front().rank() == 0);
    REQUIRE(this->communicators().front().size() == 1);
    return;
  }

  const int n  = static_cast<int>(cuda::devices.size());
  auto streams = nccl_test_util::make_streams();

  std::vector<c2h::device_vector<cuda::std::int32_t>> send;
  std::vector<c2h::device_vector<cuda::std::int32_t>> recv;

  for (int i = 0; i < n; ++i)
  {
    REQUIRE_CUDART(cudaSetDevice(i));
    auto& s = send.emplace_back(c2h::host_vector<cuda::std::int32_t>(/*size*/ 3, i));
    recv.emplace_back(s.size(), -1);
  }

  {
    auto g = this->communicators().front().group_guard();

    for (int i = 0; i < n; ++i)
    {
      const int prev = (i + n - 1) % n;
      const int next = (i + 1) % n;

      this->communicators()[i].recv(g, thrust::raw_pointer_cast(recv[i].data()), recv[i].size(), prev, streams[i]);
      this->communicators()[i].send(g, thrust::raw_pointer_cast(send[i].data()), send[i].size(), next, streams[i]);
    }
  }

  for (auto& stream : streams)
  {
    stream.sync();
  }

  // Rank r received from its predecessor (r-1): {r-1, r-1, r-1}.
  for (int r = 0; r < n; ++r)
  {
    const int prev = (r + n - 1) % n;

    const c2h::host_vector<cuda::std::int32_t> expected(recv[r].size(), prev);
    const c2h::host_vector<cuda::std::int32_t> actual = recv[r];

    REQUIRE_THAT(actual, Equals(expected));
  }
}

NCCL_COMM_TEST("nccl_communicator send/recv transports trivially copyable payload")
{
  if (cuda::devices.size() == 1)
  {
    // NCCL disallows self send/recv on a 1-rank comm.
    REQUIRE(this->communicators().front().rank() == 0);
    REQUIRE(this->communicators().front().size() == 1);
    return;
  }

  const int n  = static_cast<int>(cuda::devices.size());
  auto streams = nccl_test_util::make_streams();

  std::vector<c2h::device_vector<payload>> send;
  std::vector<c2h::device_vector<payload>> recv;

  for (int i = 0; i < n; ++i)
  {
    REQUIRE_CUDART(cudaSetDevice(i));

    c2h::host_vector<payload> h(3);

    for (cuda::std::size_t k = 0; k < h.size(); ++k)
    {
      h[k] = payload{static_cast<cuda::std::int32_t>(i), static_cast<cuda::std::int32_t>(k)};
    }

    auto& s = send.emplace_back(h);
    recv.emplace_back(s.size());
  }

  {
    auto g = this->communicators().front().group_guard();

    for (int i = 0; i < n; ++i)
    {
      const int prev = (i + n - 1) % n;
      const int next = (i + 1) % n;

      this->communicators()[i].recv(g, thrust::raw_pointer_cast(recv[i].data()), recv[i].size(), prev, streams[i]);
      this->communicators()[i].send(g, thrust::raw_pointer_cast(send[i].data()), send[i].size(), next, streams[i]);
    }
  }

  for (auto& stream : streams)
  {
    stream.sync();
  }

  for (int r = 0; r < n; ++r)
  {
    const int prev                         = (r + n - 1) % n;
    const c2h::host_vector<payload> actual = recv[r];

    for (cuda::std::size_t k = 0; k < actual.size(); ++k)
    {
      REQUIRE(actual[k].from == static_cast<cuda::std::int32_t>(prev));
      REQUIRE(actual[k].index == static_cast<cuda::std::int32_t>(k));
    }
  }
}
