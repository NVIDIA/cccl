// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

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
