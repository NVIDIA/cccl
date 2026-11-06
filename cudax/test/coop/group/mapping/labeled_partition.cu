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
#include <cuda/std/cstdint>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
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
    // Reverse label values so group ranks cannot be obtained by sorting labels.
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

template <class Label>
inline constexpr bool is_boolean_label = cuda::std::is_same_v<Label, bool> || cuda::std::is_same_v<Label, bool_label>;

enum class signed_label : cuda::std::int8_t
{
  negative = -7
};

enum class wide_label : cuda::std::uint64_t
{
};

template <class Config, class LabelFn, class... NonExhaustive>
__device__ void test_labeled_partition(Config config, LabelFn label_fn, const NonExhaustive&... non_exhaustive)
{
  using Mapping = cudax::coop::labeled_partition<LabelFn>;
  using Label   = decltype(label_fn.label(0));

  // Test construction and class template argument deduction.
  {
    static_assert(cuda::std::is_constructible_v<Mapping, LabelFn>);
    const cudax::coop::labeled_partition mapping{label_fn};
    static_assert(cuda::std::is_same_v<decltype(mapping), const Mapping>);
  }

  const cudax::coop::this_warp parent_group{config};
  const ThreadsInWarpMappingResult prev_mapping_result;

  static_assert(cudax::coop::__group_mapping_result<decltype(cuda::std::declval<Mapping>().map(
                  non_exhaustive..., cuda::gpu_thread, parent_group, prev_mapping_result))>);

  Mapping mapping{label_fn};
  const auto result = mapping.map(non_exhaustive..., cuda::gpu_thread, parent_group, prev_mapping_result);
  using Result      = decltype(result);

  const auto lane  = cuda::gpu_thread.rank_as<unsigned>(cuda::warp);
  const auto label = label_fn.label(lane);

  unsigned lane_mask_ref = 0;
  unsigned unit_count    = 0;
  unsigned unit_rank     = 0;
  unsigned first_lane    = 32;
  for (unsigned other_lane = 0; other_lane < 32; ++other_lane)
  {
    if (label_fn.label(other_lane) == label)
    {
      lane_mask_ref |= 1u << other_lane;
      ++unit_count;
      unit_rank += other_lane < lane;
      if (first_lane == 32)
      {
        first_lane = other_lane;
      }
    }
  }

  if constexpr (is_boolean_label<Label>)
  {
    // Boolean labels preserve the two logical bins, including an empty bin.
    static_assert(Result::static_group_count() == 2);
    REQUIRE(result.group_count() == 2);
    REQUIRE(result.group_rank() == static_cast<unsigned>(label));
  }
  else
  {
    unsigned group_count = 0;
    unsigned group_rank  = 0;
    for (unsigned other_lane = 0; other_lane < 32; ++other_lane)
    {
      bool is_first = true;
      for (unsigned preceding_lane = 0; preceding_lane < other_lane; ++preceding_lane)
      {
        if (label_fn.label(preceding_lane) == label_fn.label(other_lane))
        {
          is_first = false;
        }
      }
      if (is_first)
      {
        ++group_count;
        group_rank += other_lane < first_lane;
      }
    }

    static_assert(Result::static_group_count() == cuda::std::dynamic_extent);
    REQUIRE(result.group_count() == group_count);
    REQUIRE(result.group_rank() == group_rank);
  }

  static_assert(Result::static_unit_count() == cuda::std::dynamic_extent);
  REQUIRE(result.unit_count() == unit_count);
  REQUIRE(result.unit_rank() == unit_rank);
  REQUIRE(result.lane_mask() == cuda::device::lane_mask{lane_mask_ref});
  REQUIRE(result.is_valid());
  static_assert(Result::is_always_exhaustive());
  static_assert(!Result::is_always_contiguous());
}

template <class Config, class... NonExhaustive>
__device__ void test_previous_partitions(Config config, const NonExhaustive&... non_exhaustive)
{
  const cudax::coop::this_warp parent_group{config};
  const ThreadsInWarpMappingResult initial_mapping_result;
  const auto lane = cuda::gpu_thread.rank_as<unsigned>(cuda::warp);

  // Equal labels in different input tiles must remain in separate groups.
  {
    const cudax::coop::group_by tiles{cuda::std::cw<8>};
    const auto prev_mapping_result = tiles.map(cuda::gpu_thread, parent_group, initial_mapping_result);
    cudax::coop::labeled_partition mapping{rank_label_fn<int, 3>{}};
    const auto result = mapping.map(non_exhaustive..., cuda::gpu_thread, parent_group, prev_mapping_result);

    const auto local_rank = lane % 8;
    const auto label_rank = local_rank % 3;
    const auto mask       = (label_rank == 0 ? 0x49u : (label_rank == 1 ? 0x92u : 0x24u)) << (lane / 8 * 8);

    REQUIRE(result.group_count() == 12);
    REQUIRE(result.group_rank() == (lane / 8 * 3) + label_rank);
    REQUIRE(result.unit_count() == (label_rank == 2 ? 2 : 3));
    REQUIRE(result.unit_rank() == local_rank / 3);
    REQUIRE(result.lane_mask() == cuda::device::lane_mask{mask});
    REQUIRE(result.is_valid());
  }

  // Ranks and labels are relative to a sparse input group, not physical lanes.
  {
    auto even_rank = [](const auto& previous) {
      return previous.unit_rank() % 2 == 0;
    };
    cudax::coop::binary_partition partition{even_rank};
    const auto prev_mapping_result = partition.map(cuda::gpu_thread, parent_group, initial_mapping_result);
    cudax::coop::labeled_partition mapping{rank_label_fn<int, 3>{}};
    const auto result = mapping.map(non_exhaustive..., cuda::gpu_thread, parent_group, prev_mapping_result);

    unsigned mask       = 0;
    unsigned unit_count = 0;
    unsigned unit_rank  = 0;
    for (unsigned other_lane = 0; other_lane < 32; ++other_lane)
    {
      if (other_lane % 2 == lane % 2 && other_lane / 2 % 3 == lane / 2 % 3)
      {
        mask |= 1u << other_lane;
        ++unit_count;
        unit_rank += other_lane < lane;
      }
    }

    REQUIRE(result.group_count() == 6);
    REQUIRE(result.group_rank() == ((lane / 2 % 3) * 2) + (lane % 2));
    REQUIRE(result.unit_count() == unit_count);
    REQUIRE(result.unit_rank() == unit_rank);
    REQUIRE(result.lane_mask() == cuda::device::lane_mask{mask});
    REQUIRE(result.is_valid());
  }
}

template <class Config, class LabelFn, class... NonExhaustive>
__device__ void test_excluded_units(Config config, LabelFn label_fn, const NonExhaustive&... non_exhaustive)
{
  const cudax::coop::this_warp parent_group{config};
  const ThreadsInWarpMappingResult initial_mapping_result;
  const cudax::coop::take prefix{cuda::std::cw<17>};
  const auto prev_mapping_result =
    prefix.map(cudax::coop::non_exhaustive, cuda::gpu_thread, parent_group, initial_mapping_result);

  cudax::coop::labeled_partition mapping{label_fn};
  const auto result = mapping.map(non_exhaustive..., cuda::gpu_thread, parent_group, prev_mapping_result);
  using Result      = decltype(result);

  static_assert(!Result::is_always_exhaustive());
  static_assert(!Result::is_always_contiguous());
  const auto lane = cuda::gpu_thread.rank_as<unsigned>(cuda::warp);
  REQUIRE(result.is_valid() == (lane < 17));
  if (lane < 17)
  {
    const auto label = label_fn.label(lane);
    unsigned mask    = 0;
    unsigned count   = 0;
    unsigned rank    = 0;
    for (unsigned other_lane = 0; other_lane < 17; ++other_lane)
    {
      if (label_fn.label(other_lane) == label)
      {
        mask |= 1u << other_lane;
        ++count;
        rank += other_lane < lane;
      }
    }

    if constexpr (is_boolean_label<decltype(label_fn.label(0))>)
    {
      REQUIRE(result.group_count() == 2);
      REQUIRE(result.group_rank() == static_cast<unsigned>(label));
    }
    else
    {
      REQUIRE(result.group_count() == 3);
      REQUIRE(result.group_rank() == lane % 3);
    }
    REQUIRE(result.unit_count() == count);
    REQUIRE(result.unit_rank() == rank);
    REQUIRE(result.lane_mask() == cuda::device::lane_mask{mask});
  }

  // An entirely excluded input must remain invalid for every lane.
  using PreviousResult = cuda::std::remove_cv_t<decltype(prev_mapping_result)>;
  const auto invalid_result =
    mapping.map(non_exhaustive..., cuda::gpu_thread, parent_group, PreviousResult::__invalid());
  REQUIRE(!invalid_result.is_valid());
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

  test_previous_partitions(config, non_exhaustive...);
  test_excluded_units(config, rank_label_fn<bool, 2>{}, non_exhaustive...);
  test_excluded_units(config, rank_label_fn<bool_label, 2>{}, non_exhaustive...);
  test_excluded_units(config, rank_label_fn<int, 3>{}, non_exhaustive...);
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
