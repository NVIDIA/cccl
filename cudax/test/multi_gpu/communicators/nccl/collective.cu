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
#include <cuda/functional>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/functional>

#include <vector>

#include "nccl_test_helpers.cuh"
#include <c2h/vector.h>

namespace
{
constexpr cuda::std::int32_t ROOT_RANK = 0;
} // namespace

NCCL_COMM_TEST("nccl_communicator all_reduce sum")
{
  auto streams = nccl_test_util::make_streams();

  // Rank r contributes {r+1, r+1, r+1, r+1}; the element-wise sum is the n-th triangular number.
  std::vector<c2h::device_vector<cuda::std::int32_t>> send;
  std::vector<c2h::device_vector<cuda::std::int32_t>> recv;

  for (int i = 0; i < static_cast<int>(cuda::devices.size()); ++i)
  {
    REQUIRE_CUDART(cudaSetDevice(i));
    auto& s = send.emplace_back(c2h::host_vector<cuda::std::int32_t>{i + 1, i + 1, i + 1, i + 1});
    recv.emplace_back(s.size(), -1);
  }

  {
    auto g = this->communicators().front().group_guard();

    for (cuda::std::size_t i = 0; i < cuda::devices.size(); ++i)
    {
      this->communicators()[i].all_reduce(
        g,
        thrust::raw_pointer_cast(send[i].data()),
        thrust::raw_pointer_cast(recv[i].data()),
        send[i].size(),
        ::cuda::std::plus<>{},
        streams[i]);
    }
  }

  for (auto& stream : streams)
  {
    stream.sync();
  }

  const auto sum = static_cast<cuda::std::int32_t>(cuda::devices.size() * (cuda::devices.size() + 1) / 2);

  for (auto& buf : recv)
  {
    const c2h::host_vector<cuda::std::int32_t> expected(buf.size(), sum);
    const c2h::host_vector<cuda::std::int32_t> actual = buf;

    REQUIRE_THAT(actual, Equals(expected));
  }
}

NCCL_COMM_TEST("nccl_communicator all_reduce maximum")
{
  auto streams = nccl_test_util::make_streams();

  std::vector<c2h::device_vector<cuda::std::int32_t>> send;
  std::vector<c2h::device_vector<cuda::std::int32_t>> recv;

  for (int i = 0; i < static_cast<int>(cuda::devices.size()); ++i)
  {
    REQUIRE_CUDART(cudaSetDevice(i));
    auto& s = send.emplace_back(c2h::host_vector<cuda::std::int32_t>{i, 100 - i, 2 * i});
    recv.emplace_back(s.size(), -1);
  }

  {
    auto g = this->communicators().front().group_guard();

    for (cuda::std::size_t i = 0; i < cuda::devices.size(); ++i)
    {
      this->communicators()[i].all_reduce(
        g,
        thrust::raw_pointer_cast(send[i].data()),
        thrust::raw_pointer_cast(recv[i].data()),
        send[i].size(),
        ::cuda::maximum<>{},
        streams[i]);
    }
  }

  for (auto& stream : streams)
  {
    stream.sync();
  }

  const auto n                                        = static_cast<cuda::std::int32_t>(cuda::devices.size());
  const c2h::host_vector<cuda::std::int32_t> expected = {n - 1, 100, 2 * (n - 1)};

  for (auto& buf : recv)
  {
    const c2h::host_vector<cuda::std::int32_t> actual = buf;

    REQUIRE_THAT(actual, Equals(expected));
  }
}

NCCL_COMM_TEST("nccl_communicator reduce sum to root 0")
{
  auto streams = nccl_test_util::make_streams();

  std::vector<c2h::device_vector<cuda::std::int32_t>> send;
  std::vector<c2h::device_vector<cuda::std::int32_t>> recv;

  for (int i = 0; i < static_cast<int>(cuda::devices.size()); ++i)
  {
    REQUIRE_CUDART(cudaSetDevice(i));
    auto& s = send.emplace_back(c2h::host_vector<cuda::std::int32_t>{i + 1, i + 1, i + 1, i + 1});
    recv.emplace_back(s.size(), -1);
  }

  {
    auto g = this->communicators().front().group_guard();

    for (cuda::std::size_t i = 0; i < cuda::devices.size(); ++i)
    {
      this->communicators()[i].reduce(
        g,
        thrust::raw_pointer_cast(send[i].data()),
        thrust::raw_pointer_cast(recv[i].data()),
        send[i].size(),
        ::cuda::std::plus<>{},
        ROOT_RANK,
        streams[i]);
    }
  }

  for (auto& stream : streams)
  {
    stream.sync();
  }

  const auto sum = static_cast<cuda::std::int32_t>(cuda::devices.size() * (cuda::devices.size() + 1) / 2);
  const c2h::host_vector<cuda::std::int32_t> actual = recv[ROOT_RANK];
  const c2h::host_vector<cuda::std::int32_t> expected(actual.size(), sum);

  REQUIRE_THAT(actual, Equals(expected));
}

NCCL_COMM_TEST("nccl_communicator broadcast from root 0")
{
  auto streams = nccl_test_util::make_streams();

  // Only root's send buffer is read; give every rank the same literal so the source is obvious.
  std::vector<c2h::device_vector<cuda::std::int32_t>> send;
  std::vector<c2h::device_vector<cuda::std::int32_t>> recv;

  for (int i = 0; i < static_cast<int>(cuda::devices.size()); ++i)
  {
    REQUIRE_CUDART(cudaSetDevice(i));
    auto& s = send.emplace_back(c2h::host_vector<cuda::std::int32_t>{10, 20, 30, 40});
    recv.emplace_back(s.size(), -1);
  }

  {
    auto g = this->communicators().front().group_guard();

    for (cuda::std::size_t i = 0; i < cuda::devices.size(); ++i)
    {
      this->communicators()[i].broadcast(
        g,
        thrust::raw_pointer_cast(send[i].data()),
        thrust::raw_pointer_cast(recv[i].data()),
        send[i].size(),
        ROOT_RANK,
        streams[i]);
    }
  }

  for (auto& stream : streams)
  {
    stream.sync();
  }

  const c2h::host_vector<cuda::std::int32_t> expected = {10, 20, 30, 40};

  for (auto& buf : recv)
  {
    const c2h::host_vector<cuda::std::int32_t> actual = buf;

    REQUIRE_THAT(actual, Equals(expected));
  }
}

NCCL_COMM_TEST("nccl_communicator all_gather")
{
  auto streams = nccl_test_util::make_streams();

  // Rank r contributes {10*r, 10*r+1}; every rank ends up with the concatenation in rank order.
  std::vector<c2h::device_vector<cuda::std::int32_t>> send;
  std::vector<c2h::device_vector<cuda::std::int32_t>> recv;

  for (int i = 0; i < static_cast<int>(cuda::devices.size()); ++i)
  {
    REQUIRE_CUDART(cudaSetDevice(i));
    auto& s = send.emplace_back(c2h::host_vector<cuda::std::int32_t>{10 * i, 10 * i + 1});
    recv.emplace_back(s.size() * cuda::devices.size(), -1);
  }

  {
    auto g = this->communicators().front().group_guard();

    for (cuda::std::size_t i = 0; i < cuda::devices.size(); ++i)
    {
      this->communicators()[i].all_gather(
        g,
        thrust::raw_pointer_cast(send[i].data()),
        thrust::raw_pointer_cast(recv[i].data()),
        send[i].size(),
        streams[i]);
    }
  }

  for (auto& stream : streams)
  {
    stream.sync();
  }

  const cuda::std::size_t per_rank = send.front().size();
  c2h::host_vector<cuda::std::int32_t> expected(per_rank * cuda::devices.size());

  for (cuda::std::size_t r = 0; r < cuda::devices.size(); ++r)
  {
    expected[r * per_rank]     = static_cast<cuda::std::int32_t>(10 * r);
    expected[r * per_rank + 1] = static_cast<cuda::std::int32_t>(10 * r + 1);
  }

  for (auto& buf : recv)
  {
    const c2h::host_vector<cuda::std::int32_t> actual = buf;

    REQUIRE_THAT(actual, Equals(expected));
  }
}

NCCL_COMM_TEST("nccl_communicator gather_v to root 0")
{
  auto streams = nccl_test_util::make_streams();

  // Rank r contributes (2 + r) elements: {10*r, 10*r+1, ...}. Root concatenates them in rank order.
  std::vector<c2h::device_vector<cuda::std::int32_t>> send;

  for (int i = 0; i < static_cast<int>(cuda::devices.size()); ++i)
  {
    REQUIRE_CUDART(cudaSetDevice(i));

    c2h::host_vector<cuda::std::int32_t> h(2 + static_cast<cuda::std::size_t>(i));

    for (cuda::std::size_t k = 0; k < h.size(); ++k)
    {
      h[k] = 10 * i + static_cast<int>(k);
    }

    send.emplace_back(h);
  }

  std::vector<cuda::std::size_t> recv_counts(cuda::devices.size());
  std::vector<cuda::std::size_t> displs(cuda::devices.size());
  cuda::std::size_t total = 0;

  for (cuda::std::size_t r = 0; r < cuda::devices.size(); ++r)
  {
    recv_counts[r] = send[r].size();
    displs[r]      = total;
    total += recv_counts[r];
  }

  REQUIRE_CUDART(cudaSetDevice(ROOT_RANK));

  c2h::device_vector<cuda::std::int32_t> recv(total, -1);

  {
    auto g = this->communicators().front().group_guard();

    for (cuda::std::size_t i = 0; i < cuda::devices.size(); ++i)
    {
      this->communicators()[i].gather_v(
        g,
        thrust::raw_pointer_cast(send[i].data()),
        send[i].size(),
        thrust::raw_pointer_cast(recv.data()),
        recv_counts.data(),
        displs.data(),
        ROOT_RANK,
        streams[i]);
    }
  }

  for (auto& stream : streams)
  {
    stream.sync();
  }

  c2h::host_vector<cuda::std::int32_t> expected(total);

  for (cuda::std::size_t r = 0; r < cuda::devices.size(); ++r)
  {
    for (cuda::std::size_t k = 0; k < recv_counts[r]; ++k)
    {
      expected[displs[r] + k] = static_cast<cuda::std::int32_t>(10 * r + k);
    }
  }

  const c2h::host_vector<cuda::std::int32_t> actual = recv;

  REQUIRE_THAT(actual, Equals(expected));
}

NCCL_COMM_TEST("nccl_communicator all_to_all_v")
{
  auto streams = nccl_test_util::make_streams();

  // Two elements exchanged with each peer; the block for peer j sits at offset block*j.
  constexpr cuda::std::size_t block = 2;

  std::vector<cuda::std::size_t> counts(cuda::devices.size(), block);
  std::vector<cuda::std::size_t> displs(cuda::devices.size());

  for (cuda::std::size_t j = 0; j < cuda::devices.size(); ++j)
  {
    displs[j] = block * j;
  }

  // Rank r block destined for peer j encodes 100*r + 10*j + k.
  std::vector<c2h::device_vector<cuda::std::int32_t>> send;
  std::vector<c2h::device_vector<cuda::std::int32_t>> recv;

  for (int i = 0; i < static_cast<int>(cuda::devices.size()); ++i)
  {
    REQUIRE_CUDART(cudaSetDevice(i));

    c2h::host_vector<cuda::std::int32_t> h(block * cuda::devices.size());

    for (cuda::std::size_t j = 0; j < cuda::devices.size(); ++j)
    {
      h[block * j]     = static_cast<cuda::std::int32_t>(100 * i + 10 * j);
      h[block * j + 1] = static_cast<cuda::std::int32_t>(100 * i + 10 * j + 1);
    }

    auto& s = send.emplace_back(h);
    recv.emplace_back(s.size(), -1);
  }

  {
    auto g = this->communicators().front().group_guard();

    for (cuda::std::size_t i = 0; i < cuda::devices.size(); ++i)
    {
      this->communicators()[i].all_to_all_v(
        g,
        thrust::raw_pointer_cast(send[i].data()),
        counts.data(),
        displs.data(),
        thrust::raw_pointer_cast(recv[i].data()),
        counts.data(),
        displs.data(),
        streams[i]);
    }
  }

  for (auto& stream : streams)
  {
    stream.sync();
  }

  // Rank r receives from peer i the block i sent to r, placed at block*i: 100*i + 10*r + k.
  for (cuda::std::size_t r = 0; r < cuda::devices.size(); ++r)
  {
    c2h::host_vector<cuda::std::int32_t> expected(block * cuda::devices.size());

    for (cuda::std::size_t i = 0; i < cuda::devices.size(); ++i)
    {
      expected[block * i]     = static_cast<cuda::std::int32_t>(100 * i + 10 * r);
      expected[block * i + 1] = static_cast<cuda::std::int32_t>(100 * i + 10 * r + 1);
    }

    const c2h::host_vector<cuda::std::int32_t> actual = recv[r];

    REQUIRE_THAT(actual, Equals(expected));
  }
}

#if NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)

NCCL_COMM_TEST("nccl_communicator gather to root 0")
{
  auto streams = nccl_test_util::make_streams();

  std::vector<c2h::device_vector<cuda::std::int32_t>> send;
  std::vector<c2h::device_vector<cuda::std::int32_t>> recv;

  for (int i = 0; i < static_cast<int>(cuda::devices.size()); ++i)
  {
    REQUIRE_CUDART(cudaSetDevice(i));
    auto& s = send.emplace_back(c2h::host_vector<cuda::std::int32_t>{10 * i, 10 * i + 1});
    recv.emplace_back(s.size() * cuda::devices.size(), -1);
  }

  {
    auto g = this->communicators().front().group_guard();

    for (cuda::std::size_t i = 0; i < cuda::devices.size(); ++i)
    {
      this->communicators()[i].gather(
        g,
        thrust::raw_pointer_cast(send[i].data()),
        thrust::raw_pointer_cast(recv[i].data()),
        send[i].size(),
        ROOT_RANK,
        streams[i]);
    }
  }

  for (auto& stream : streams)
  {
    stream.sync();
  }

  const cuda::std::size_t per_rank = send.front().size();
  c2h::host_vector<cuda::std::int32_t> expected(per_rank * cuda::devices.size());

  for (cuda::std::size_t r = 0; r < cuda::devices.size(); ++r)
  {
    expected[r * per_rank]     = static_cast<cuda::std::int32_t>(10 * r);
    expected[r * per_rank + 1] = static_cast<cuda::std::int32_t>(10 * r + 1);
  }

  const c2h::host_vector<cuda::std::int32_t> actual = recv[ROOT_RANK];

  REQUIRE_THAT(actual, Equals(expected));
}

NCCL_COMM_TEST("nccl_communicator all_to_all")
{
  auto streams = nccl_test_util::make_streams();

  // Two elements exchanged with each peer; the block for peer j sits at offset block*j.
  constexpr cuda::std::size_t block = 2;

  // Rank r block destined for peer j encodes 100*r + 10*j + k.
  std::vector<c2h::device_vector<cuda::std::int32_t>> send;
  std::vector<c2h::device_vector<cuda::std::int32_t>> recv;

  for (int i = 0; i < static_cast<int>(cuda::devices.size()); ++i)
  {
    REQUIRE_CUDART(cudaSetDevice(i));

    c2h::host_vector<cuda::std::int32_t> h(block * cuda::devices.size());

    for (cuda::std::size_t j = 0; j < cuda::devices.size(); ++j)
    {
      h[block * j]     = static_cast<cuda::std::int32_t>(100 * i + 10 * j);
      h[block * j + 1] = static_cast<cuda::std::int32_t>(100 * i + 10 * j + 1);
    }

    auto& s = send.emplace_back(h);
    recv.emplace_back(s.size(), -1);
  }

  {
    auto g = this->communicators().front().group_guard();

    for (cuda::std::size_t i = 0; i < cuda::devices.size(); ++i)
    {
      this->communicators()[i].all_to_all(
        g, thrust::raw_pointer_cast(send[i].data()), thrust::raw_pointer_cast(recv[i].data()), block, streams[i]);
    }
  }

  for (auto& stream : streams)
  {
    stream.sync();
  }

  // Rank r receives from peer i the block i sent to r, placed at block*i: 100*i + 10*r + k.
  for (cuda::std::size_t r = 0; r < cuda::devices.size(); ++r)
  {
    c2h::host_vector<cuda::std::int32_t> expected(block * cuda::devices.size());

    for (cuda::std::size_t i = 0; i < cuda::devices.size(); ++i)
    {
      expected[block * i]     = static_cast<cuda::std::int32_t>(100 * i + 10 * r);
      expected[block * i + 1] = static_cast<cuda::std::int32_t>(100 * i + 10 * r + 1);
    }

    const c2h::host_vector<cuda::std::int32_t> actual = recv[r];

    REQUIRE_THAT(actual, Equals(expected));
  }
}

#endif // NCCL_VERSION_CODE >= NCCL_VERSION(2, 28, 0)
