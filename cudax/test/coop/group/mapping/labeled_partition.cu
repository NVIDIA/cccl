//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__cccl_config>
#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/stream>
#include <cuda/warp>

#include <cuda/experimental/coop/group>

#include <vector_types.h>

#include "group_testing.cuh"

namespace
{
template <class Label>
struct constant_label_fn
{
  Label value;

  __device__ Label label(unsigned) const
  {
    return value;
  }

  template <class MappingResult>
  __device__ Label operator()(const MappingResult&) const
  {
    return value;
  }
};

template <class Label, unsigned Modulus, unsigned Shift = 0>
struct rank_label_fn
{
  __device__ Label label(unsigned rank) const
  {
    // Exercise labels whose values differ from their order of appearance.
    if constexpr (cuda::std::is_enum_v<Label>)
    {
      return static_cast<Label>(rank_label_fn<cuda::std::underlying_type_t<Label>, Modulus, Shift>{}.label(rank));
    }
    else
    {
      return static_cast<Label>(static_cast<Label>(Modulus - 1 - (rank % Modulus)) << Shift);
    }
  }

  template <class MappingResult>
  __device__ Label operator()(const MappingResult& mapping_result) const
  {
    return label(mapping_result.unit_rank());
  }
};

enum class bool_label : bool
{
  false_value = false,
  true_value  = true
};

enum class signed_label : cuda::std::int8_t
{
  negative = -7
};

enum class wide_label : cuda::std::uint64_t
{
};

// All lanes participate, including lanes excluded by a previous mapping.
template <class Result>
__device__ void check_partition(const Result& result, unsigned group_count, unsigned mask)
{
  const auto valid = result.is_valid();
  const auto rank  = valid ? result.group_rank() : 0u;
  REQUIRE(valid == (mask != 0));
  if (valid)
  {
    REQUIRE(result.group_count() == group_count);
    REQUIRE(result.lane_mask() == cuda::device::lane_mask{mask});
    REQUIRE(result.unit_count() == cuda::std::popcount(mask));
    REQUIRE(result.unit_rank() == cuda::std::popcount(mask & cuda::device::lane_mask::all_less().value()));
    REQUIRE(rank < group_count);
  }
  for (int other_lane = 0; other_lane < 32; ++other_lane)
  {
    const auto other_rank = __shfl_sync(0xffffffffu, rank, other_lane);
    const auto other_mask = __shfl_sync(0xffffffffu, mask, other_lane);
    if (valid && other_mask != 0)
    {
      // Every group has one distinct rank; the assignment order is unspecified.
      REQUIRE((rank == other_rank) == (mask == other_mask));
    }
  }
}

template <unsigned Count = 32, class Config, class LabelFn, class... NonExhaustive>
__device__ void test_labeled_partition(Config config, LabelFn label_fn, const NonExhaustive&... non_exhaustive)
{
  using Mapping = cudax::coop::labeled_partition<LabelFn>;

  // Test construction and class template argument deduction.
  static_assert(cuda::std::is_constructible_v<Mapping, LabelFn>);
  const cudax::coop::labeled_partition deduced_mapping{label_fn};
  static_assert(cuda::std::is_same_v<decltype(deduced_mapping), const Mapping>);

  const cudax::coop::this_warp parent_group{config};
  const ThreadsInWarpMappingResult initial_mapping_result;
  const cudax::coop::take prefix{cuda::std::cw<Count>};
  const auto prev_mapping_result =
    prefix.map(cudax::coop::non_exhaustive, cuda::gpu_thread, parent_group, initial_mapping_result);

  Mapping mapping{label_fn};
  const auto result = mapping.map(non_exhaustive..., cuda::gpu_thread, parent_group, prev_mapping_result);
  using Result      = decltype(result);

  static_assert(cudax::coop::__group_mapping_result<Result>);
  static_assert(Result::static_group_count() == cuda::std::dynamic_extent);
  static_assert(Result::static_unit_count() == cuda::std::dynamic_extent);
  static_assert(Result::is_always_exhaustive() == (Count == 32));
  static_assert(!Result::is_always_contiguous());

  const auto lane      = cuda::gpu_thread.rank_as<unsigned>(cuda::warp);
  unsigned mask        = 0;
  unsigned group_count = 0;
  for (unsigned other_lane = 0; other_lane < Count; ++other_lane)
  {
    if (lane < Count && label_fn.label(other_lane) == label_fn.label(lane))
    {
      mask |= 1u << other_lane;
    }
    bool is_first = true;
    for (unsigned preceding_lane = 0; preceding_lane < other_lane; ++preceding_lane)
    {
      if (label_fn.label(preceding_lane) == label_fn.label(other_lane))
      {
        is_first = false;
      }
    }
    group_count += is_first;
  }
  check_partition(result, group_count, mask);

  if constexpr (Count < 32)
  {
    // An entirely excluded input must remain invalid for every lane.
    using PreviousResult = cuda::std::remove_cv_t<decltype(prev_mapping_result)>;
    const auto invalid_result =
      mapping.map(non_exhaustive..., cuda::gpu_thread, parent_group, PreviousResult::__invalid());
    REQUIRE(!invalid_result.is_valid());
  }
}

template <class Label, unsigned Modulus, class Config, class... NonExhaustive>
__device__ void test_previous_partitions(Config config, const NonExhaustive&... non_exhaustive)
{
  const cudax::coop::this_warp parent_group{config};
  const ThreadsInWarpMappingResult initial_mapping_result;
  const auto lane = cuda::gpu_thread.rank_as<unsigned>(cuda::warp);

  // Equal labels in different tiles remain separate, including when half the lanes are excluded.
  for (unsigned count = 8; count >= 4; count /= 2)
  {
    const cudax::coop::group_by tiles{cuda::std::cw<8>};
    const auto tiled_result = tiles.map(cuda::gpu_thread, parent_group, initial_mapping_result);
    const cudax::coop::take prefix{count};
    const auto prev_mapping_result =
      prefix.map(cudax::coop::non_exhaustive, cuda::gpu_thread, parent_group, tiled_result);
    cudax::coop::labeled_partition mapping{rank_label_fn<Label, Modulus>{}};
    const auto result = mapping.map(non_exhaustive..., cuda::gpu_thread, parent_group, prev_mapping_result);

    unsigned mask = 0;
    if (lane % 8 < count)
    {
      for (unsigned other_rank = lane % 8 % Modulus; other_rank < count; other_rank += Modulus)
      {
        mask |= 1u << ((lane / 8 * 8) + other_rank);
      }
    }
    check_partition(result, 4 * Modulus, mask);
  }

  // Ranks and labels are relative to a sparse input group, not physical lanes.
  {
    auto even_rank = [](const auto& previous) {
      return previous.unit_rank() % 2 == 0;
    };
    cudax::coop::binary_partition partition{even_rank};
    const auto prev_mapping_result = partition.map(cuda::gpu_thread, parent_group, initial_mapping_result);
    cudax::coop::labeled_partition mapping{rank_label_fn<Label, Modulus>{}};
    const auto result = mapping.map(non_exhaustive..., cuda::gpu_thread, parent_group, prev_mapping_result);

    unsigned mask = 0;
    for (unsigned other_lane = 0; other_lane < 32; ++other_lane)
    {
      if (other_lane % 2 == lane % 2 && other_lane / 2 % Modulus == lane / 2 % Modulus)
      {
        mask |= 1u << other_lane;
      }
    }

    check_partition(result, 2 * Modulus, mask);
  }
}

template <class Config, class... NonExhaustive>
__device__ void test_labels(Config config, const NonExhaustive&... non_exhaustive)
{
  test_labeled_partition(config, constant_label_fn<bool>{true}, non_exhaustive...);
  test_labeled_partition(config, constant_label_fn<bool>{false}, non_exhaustive...);
  test_labeled_partition(config, rank_label_fn<bool, 2>{}, non_exhaustive...);
  test_labeled_partition(config, constant_label_fn<bool_label>{bool_label::true_value}, non_exhaustive...);
  test_labeled_partition(config, constant_label_fn<bool_label>{bool_label::false_value}, non_exhaustive...);
  test_labeled_partition(config, rank_label_fn<bool_label, 2>{}, non_exhaustive...);

  test_labeled_partition(config, constant_label_fn<int>{-7}, non_exhaustive...);
  test_labeled_partition(config, rank_label_fn<int, 32>{}, non_exhaustive...);
  test_labeled_partition(config, rank_label_fn<int, 3>{}, non_exhaustive...);
  test_labeled_partition(config, rank_label_fn<cuda::std::int8_t, 3>{}, non_exhaustive...);
  test_labeled_partition(config, rank_label_fn<cuda::std::uint8_t, 3>{}, non_exhaustive...);
  test_labeled_partition(config, rank_label_fn<cuda::std::int16_t, 3>{}, non_exhaustive...);
  test_labeled_partition(config, rank_label_fn<cuda::std::uint16_t, 3>{}, non_exhaustive...);
  test_labeled_partition(config, rank_label_fn<cuda::std::uint32_t, 3>{}, non_exhaustive...);
  test_labeled_partition(config, rank_label_fn<cuda::std::uint64_t, 3>{}, non_exhaustive...);
  test_labeled_partition(config, rank_label_fn<cuda::std::int64_t, 3, 40>{}, non_exhaustive...);
  test_labeled_partition(config, rank_label_fn<cuda::std::uint64_t, 3, 40>{}, non_exhaustive...);
  test_labeled_partition(config, constant_label_fn<signed_label>{signed_label::negative}, non_exhaustive...);
  test_labeled_partition(config, rank_label_fn<wide_label, 3, 40>{}, non_exhaustive...);
#if _CCCL_HAS_INT128()
  test_labeled_partition(config, rank_label_fn<__uint128_t, 3>{}, non_exhaustive...);
  test_labeled_partition(config, rank_label_fn<__int128_t, 3, 80>{}, non_exhaustive...);
  test_labeled_partition(config, rank_label_fn<__uint128_t, 3, 80>{}, non_exhaustive...);
#endif // _CCCL_HAS_INT128()

  test_previous_partitions<bool, 2>(config, non_exhaustive...);
  test_previous_partitions<bool_label, 2>(config, non_exhaustive...);
  test_previous_partitions<int, 3>(config, non_exhaustive...);
  test_labeled_partition<17>(config, constant_label_fn<bool>{true}, non_exhaustive...);
  test_labeled_partition<17>(config, constant_label_fn<bool>{false}, non_exhaustive...);
  test_labeled_partition<17>(config, rank_label_fn<bool, 2>{}, non_exhaustive...);
  test_labeled_partition<17>(config, rank_label_fn<bool_label, 2>{}, non_exhaustive...);
  test_labeled_partition<17>(config, rank_label_fn<int, 3>{}, non_exhaustive...);
}

struct test_kernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    test_labels(config);
    test_labels(config, cudax::coop::non_exhaustive);
  }
};
} // namespace

C2H_TEST("Labeled partition mapping", "[group]")
{
  const auto device = cuda::devices[0];
  const cuda::stream stream{device};

  {
    const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<8, 4>());
    cuda::launch(stream, config, test_kernel{});
  }
  {
    const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims(dim3{8, 4}));
    cuda::launch(stream, config, test_kernel{});
  }

  stream.sync();
}
