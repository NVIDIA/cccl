// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_segmented_reduce.cuh>

#include <thrust/device_vector.h>

#include <c2h/catch2_test_helper.h>

template <int BlockThreads>
struct reduce_tuning : cub::detail::reduce::tuning<reduce_tuning<BlockThreads>>
{
  template <class /* AccumT */, class /* Offset */, class /* OpT */>
  struct fn
  {
    struct Policy500 : cub::ChainedPolicy<500, Policy500, Policy500>
    {
      struct ReducePolicy
      {
        static constexpr int VECTOR_LOAD_LENGTH = 1;

        static constexpr cub::BlockReduceAlgorithm BLOCK_ALGORITHM = cub::BLOCK_REDUCE_WARP_REDUCTIONS;

        static constexpr cub::CacheLoadModifier LOAD_MODIFIER = cub::LOAD_DEFAULT;

        static constexpr int ITEMS_PER_THREAD = 1;
        static constexpr int BLOCK_THREADS    = BlockThreads;
      };

      using SingleTilePolicy      = ReducePolicy;
      using SegmentedReducePolicy = ReducePolicy;
    };

    using MaxPolicy = Policy500;
  };
};

struct get_scan_tuning_query_t
{};

struct scan_tuning
{
  [[nodiscard]] _CCCL_NODEBUG_API constexpr auto query(const get_scan_tuning_query_t&) const noexcept
  {
    return *this;
  }

  // Make sure this is not used
  template <class /* AccumT */, class /* Offset */, class /* OpT */>
  struct fn
  {};
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

  // We are expecting that `scan_tuning` is ignored
  auto env = cuda::execution::__tune(reduce_tuning<target_block_size>{}, scan_tuning{});

  auto error =
    cub::DeviceSegmentedReduce::Sum(d_in.begin(), d_out.begin(), num_segments, d_offsets_it, d_offsets_it + 1, env);
  thrust::device_vector<int> expected{21, 0, 17};

  REQUIRE(d_out == expected);
  REQUIRE(error == cudaSuccess);
}
