// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cuda/iterator>
#include <cuda/std/random>
#include <cuda/std/span>

#include <cuda/experimental/mgmn/shuffle>

#include "shuffle_common.cuh"

MULTI_GPU_TEST("HashGraph shuffle range conservation and boundaries", shuffle_test::types)
{
  using T = typename c2h::get<0, TestType>;
  shuffle_test::cases<T, false>(this->communicators());
}

MULTI_GPU_TEST("HashGraph shuffle duplicate values", )
{
  const auto comms    = this->communicators();
  const auto in_place = GENERATE(false, true);
  shuffle_test::check<cuda::std::int64_t, false>(
    comms, std::vector<cuda::std::size_t>(comms.size(), 1031), in_place, 7U, true);
}

MULTI_GPU_TEST("HashGraph shuffle no local communicators preserves generator", )
{
  auto comms = cuda::std::span<cudax::mgmn::nccl_communicator_ref>{};
  auto envs  = cuda::std::span<cuda::stream_ref>{};
  auto iters = cuda::std::span<int*>{};
  auto sizes = cuda::std::span<cuda::std::size_t>{};
  cuda::std::minstd_rand generator{1234};
  const auto original = generator;
  cudax::mgmn::shuffle(cudax::distributed, comms, envs, iters, sizes, generator);
  REQUIRE(generator == original);
}

MULTI_GPU_TEST("HashGraph permutation from counting iterators", )
{
  using index        = cuda::std::uint64_t;
  const auto comms   = this->communicators();
  auto streams_owned = nccl_test_util::make_streams();
  auto streams       = std::vector<cuda::stream_ref>{streams_owned.begin(), streams_owned.end()};
  std::vector<cuda::counting_iterator<index>> inputs;
  std::vector<index*> outputs;
  std::vector<cuda::std::size_t> sizes;
  std::vector<cuda::device_buffer<index>> buffers;
  index first           = 0;
  constexpr index guard = 999999999;
  for (cuda::std::size_t i = 0; i < comms.size(); ++i)
  {
    const auto n = 1031 + i;
    inputs.emplace_back(first);
    sizes.push_back(n);
    first += n;
    buffers.emplace_back(cuda::make_device_buffer<index>(
      streams[i], comms[i].logical_device().underlying_device(), std::vector<index>(n + 2, guard)));
    outputs.push_back(buffers.back().data() + 1);
  }
  cuda::std::minstd_rand generator{7};
  const auto original = generator;
  cudax::mgmn::shuffle_copy(cudax::distributed, comms, streams, inputs, sizes, outputs, generator);
  REQUIRE(generator != original);
  std::vector<index> actual;
  for (const auto& buffer : buffers)
  {
    const auto result = shuffle_test::read(buffer);
    REQUIRE(result.front() == guard);
    REQUIRE(result.back() == guard);
    actual.insert(actual.end(), result.begin() + 1, result.end() - 1);
  }
  std::sort(actual.begin(), actual.end());
  REQUIRE(actual.size() == first);
  for (index i = 0; i < first; ++i)
  {
    REQUIRE(actual[i] == i);
  }
}
