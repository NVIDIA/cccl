// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_reduce.cuh>

#include <thrust/device_vector.h>

#include <c2h/catch2_test_helper.h>

template <int BlockThreads>
struct reduce_tuning
{
  _CCCL_API constexpr auto operator()(::cuda::arch_id) const -> cub::detail::reduce::reduce_policy
  {
    auto rp = cub::detail::reduce::agent_reduce_policy{
      BlockThreads, 1, 1, cub::BLOCK_REDUCE_WARP_REDUCTIONS, cub::LOAD_DEFAULT};
    return {rp, rp, rp};
  }
};

struct unrelated_policy
{};

struct unrelated_tuning
{
  // should never be called
  auto operator()(cuda::arch_id /*arch*/) const -> unrelated_policy
  {
    throw 1337;
  }
};

using block_sizes = c2h::type_list<cuda::std::integral_constant<int, 32>, cuda::std::integral_constant<int, 64>>;

C2H_TEST("Device segmented sum can be tuned", "[reduce][device]", block_sizes)
{
  constexpr int target_block_size = c2h::get<0, TestType>::value;

  int num_segments                     = 3;
  thrust::device_vector<int> d_offsets = {0, 3, 3, 7};
  auto d_offsets_it                    = thrust::raw_pointer_cast(d_offsets.data());
  thrust::device_vector<int> d_in{8, 6, 7, 5, 3, 0, 9};
  thrust::device_vector<int> d_out(3);

  // We are expecting that `unrelated_tuning` is ignored
  auto env = cuda::execution::__tune(reduce_tuning<target_block_size>{}, unrelated_tuning{});

  auto error =
    cub::DeviceSegmentedReduce::Sum(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);
  thrust::device_vector<int> expected{21, 0, 17};

  REQUIRE(d_out == expected);
  REQUIRE(error == cudaSuccess);
}
