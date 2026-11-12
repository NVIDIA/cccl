//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/stream>
#include <cuda/warp>

#include <cuda/experimental/coop/group>

#include "group_testing.cuh"

template <bool Exhaustive, bool Contiguous>
__device__ void test_result_storage()
{
  // Each group has its own static count; there is no single count shared by all groups.
  using Result = cudax::coop::__mapping_result<cudax::coop::__static_unit_counts<4, 12, 16>, Exhaustive, Contiguous>;

  static_assert(cudax::coop::__group_mapping_result<Result>);
  static_assert(Result::static_group_count() == 3);
  static_assert(!Result::has_uniform_static_unit_count());
  static_assert(Result::static_unit_count() == cuda::std::dynamic_extent);

  static_assert(Result::static_unit_count(0) == 4);
  static_assert(Result::static_unit_count(1) == 12);
  static_assert(Result::static_unit_count(2) == 16);

  static_assert(Result::is_always_exhaustive() == Exhaustive);
  static_assert(Result::is_always_contiguous() == Contiguous);

  // Visit every unit, including ranks larger than the number of groups, to catch
  // accidental indexing of the count pack by unit rank instead of group rank.
  constexpr unsigned counts[]{4, 12, 16};
  for (unsigned group = 0; group < 3; ++group)
  {
    for (unsigned rank = 0; rank < counts[group]; ++rank)
    {
      const auto mask = cuda::device::lane_mask{0x55u};
      const Result result{group, rank, mask};

      CHECK(result.is_valid());
      CHECK(result.group_count() == 3);
      CHECK(result.group_rank() == group);
      CHECK(result.unit_count() == counts[group]);
      CHECK(result.unit_rank() == rank);
      CHECK(result.lane_mask() == mask);
    }
  }

  // Non-exhaustive results must also be able to represent an excluded unit.
  if constexpr (!Exhaustive)
  {
    const auto invalid = Result::__invalid();

    CHECK(!invalid.is_valid());
  }

  // Equal entries in the count pack still expose a uniform static count.
  using Uniform = cudax::coop::__mapping_result<cudax::coop::__static_unit_counts<8, 8, 8, 8>, Exhaustive, Contiguous>;

  static_assert(Uniform::has_uniform_static_unit_count());
  static_assert(Uniform::static_unit_count() == 8);
  static_assert(Uniform::static_unit_count(3) == 8);

  const Uniform uniform{2, 7, cuda::device::lane_mask::all()};

  CHECK(uniform.unit_count() == 8);
}

template <unsigned N, bool Static, bool Exhaustive, bool Contiguous = true, class Config>
__device__ void test_group_by_counts(Config config)
{
  const cudax::coop::this_warp parent{config};

  // Lay out three consecutive groups in one warp: lanes [0, 4), [4, 16), and [16, 32).
  using Prev           = cudax::coop::__mapping_result<cudax::coop::__static_unit_counts<4, 12, 16>, true, Contiguous>;
  const auto lane      = cuda::gpu_thread.rank_as<unsigned>(cuda::warp);
  const unsigned group = lane < 4 ? 0 : (lane < 16 ? 1 : 2);
  constexpr unsigned counts[]{4, 12, 16};
  constexpr unsigned starts[]{0, 4, 16};

  const auto rank = lane - starts[group];
  const auto mask = ((1u << counts[group]) - 1) << starts[group];
  const Prev prev{group, rank, cuda::device::lane_mask{mask}};

  // Split each input group independently, using either a static or a runtime size.
  const cudax::coop::group_by<Static ? N : cuda::std::dynamic_extent> mapping{N};

  const auto result = [&] {
    if constexpr (Exhaustive)
    {
      return mapping.map(cuda::gpu_thread, parent, prev);
    }
    else
    {
      return mapping.map(cudax::coop::non_exhaustive, cuda::gpu_thread, parent, prev);
    }
  }();

  // The total is the sum of the complete subgroups in each input group.
  using Result               = decltype(result);
  constexpr auto group_count = (4 / N) + (12 / N) + (16 / N);

  static_assert(Result::static_group_count() == (Static ? group_count : cuda::std::dynamic_extent));
  static_assert(Result::has_uniform_static_unit_count());
  static_assert(Result::static_unit_count() == (Static ? N : cuda::std::dynamic_extent));
  static_assert(Result::is_always_exhaustive() == (Exhaustive || (Static && N == 4)));

  // Drop only the remainder of each input group. Valid subgroup ranks must include
  // all complete subgroups from earlier input groups.
  CHECK(result.is_valid() == (rank < counts[group] / N * N));
  if (result.is_valid())
  {
    const unsigned offsets[]{0, 4 / N, (4 / N) + (12 / N)};

    CHECK(result.group_count() == group_count);
    CHECK(result.group_rank() == offsets[group] + (rank / N));
    CHECK(result.unit_count() == N);
    CHECK(result.unit_rank() == rank % N);
    CHECK(result.lane_mask() == cuda::device::lane_mask{(~0u >> (32 - N)) << (starts[group] + (rank / N * N))});
  }

  // A unit excluded by an earlier mapping must remain excluded.
  using InvalidPrev  = cudax::coop::__mapping_result<cudax::coop::__static_unit_counts<4, 12, 16>, false, Contiguous>;
  const auto invalid = mapping.map(cudax::coop::non_exhaustive, cuda::gpu_thread, parent, InvalidPrev::__invalid());

  CHECK(!invalid.is_valid());

  // Identity preserves the count pack as well as the current group's count.
  const auto identity = cudax::coop::identity_mapping{}.map(cuda::gpu_thread, parent, prev);

  static_assert(cuda::std::is_same_v<decltype(identity), const Prev>);

  CHECK(identity.unit_count() == counts[group]);

  // Taking three units from every input group produces a uniform static count.
  const auto taken =
    cudax::coop::take{cuda::std::cw<3>}.map(cudax::coop::non_exhaustive, cuda::gpu_thread, parent, prev);

  static_assert(decltype(taken)::static_group_count() == 3);
  static_assert(decltype(taken)::static_unit_count(2) == 3);

  CHECK(taken.is_valid() == (rank < 3));
  if (taken.is_valid())
  {
    CHECK(taken.group_rank() == group);
    CHECK(taken.unit_count() == 3);
    CHECK(taken.unit_rank() == rank);
    CHECK(taken.lane_mask() == cuda::device::lane_mask{7u << starts[group]});
  }
}

struct OddRank
{
  template <class Result>
  __device__ bool operator()(const Result& result) const
  {
    return result.unit_rank() % 2 != 0;
  }
};

struct RankLabel
{
  template <class Result>
  __device__ unsigned operator()(const Result& result) const
  {
    return result.unit_rank() % 2;
  }
};

template <class Config>
__device__ void test_other_mappings(Config config)
{
  const cudax::coop::this_warp parent{config};

  // Lay out three consecutive groups in one warp: lanes [0, 4), [4, 16), and [16, 32).
  using Prev           = cudax::coop::__mapping_result<cudax::coop::__static_unit_counts<4, 12, 16>, true, true>;
  const auto lane      = cuda::gpu_thread.rank_as<unsigned>(cuda::warp);
  const unsigned group = lane < 4 ? 0 : (lane < 16 ? 1 : 2);
  constexpr unsigned counts[]{4, 12, 16};
  constexpr unsigned starts[]{0, 4, 16};

  const auto rank = lane - starts[group];
  const auto mask = ((1u << counts[group]) - 1) << starts[group];
  const Prev prev{group, rank, cuda::device::lane_mask{mask}};

  // Split each group into even and odd ranks. The resulting counts are dynamic,
  // but both partitions must use the correct input group's size.
  cudax::coop::binary_partition binary{OddRank{}};
  cudax::coop::labeled_partition boolean_labels{OddRank{}};
  cudax::coop::labeled_partition integer_labels{RankLabel{}};

  const auto binary_result  = binary.map(cuda::gpu_thread, parent, prev);
  const auto boolean_result = boolean_labels.map(cuda::gpu_thread, parent, prev);
  const auto integer_result = integer_labels.map(cuda::gpu_thread, parent, prev);

  static_assert(decltype(binary_result)::static_group_count() == 6);
  static_assert(decltype(boolean_result)::static_group_count() == cuda::std::dynamic_extent);
  static_assert(decltype(integer_result)::static_group_count() == cuda::std::dynamic_extent);

  // Boolean partitions put all false groups before all true groups. Integer labels
  // number groups by their first physical lane instead.
  CHECK(binary_result.group_count() == 6);
  CHECK(binary_result.group_rank() == group + ((rank % 2) * 3));
  CHECK(boolean_result.group_count() == 6);
  CHECK(boolean_result.group_rank() == integer_result.group_rank());
  CHECK(integer_result.group_count() == 6);
  CHECK(integer_result.group_rank() == (group * 2) + (rank % 2));

  CHECK(binary_result.unit_count() == counts[group] / 2);
  CHECK(boolean_result.unit_count() == counts[group] / 2);
  CHECK(integer_result.unit_count() == counts[group] / 2);

  CHECK(binary_result.unit_rank() == rank / 2);
  CHECK(boolean_result.unit_rank() == rank / 2);
  CHECK(integer_result.unit_rank() == rank / 2);

  const auto partition_mask = mask & ((rank % 2) ? 0xaaaaaaaau : 0x55555555u);

  CHECK(binary_result.lane_mask() == cuda::device::lane_mask{partition_mask});
  CHECK(boolean_result.lane_mask() == binary_result.lane_mask());
  CHECK(integer_result.lane_mask() == binary_result.lane_mask());

  // A second mapping consumes a noncontiguous result with dynamic counts.
  const auto split = cudax::coop::group_by{cuda::std::cw<2>}.map(cuda::gpu_thread, parent, binary_result);

  CHECK(split.group_count() == 16);
  CHECK(split.unit_count() == 2);
  CHECK(split.unit_rank() == rank / 2 % 2);

  // Keep the first four units of each input group, split into groups of one and three.
  constexpr unsigned new_counts[]{1, 3};
  const auto reshaped =
    cudax::coop::group_as{new_counts}.map(cudax::coop::non_exhaustive, cuda::gpu_thread, parent, prev);

  CHECK(reshaped.is_valid() == (rank < 4));
  if (reshaped.is_valid())
  {
    CHECK(reshaped.group_count() == 6);
    CHECK(reshaped.group_rank() == (group * 2) + (rank != 0));
    CHECK(reshaped.unit_count() == (rank == 0 ? 1 : 3));
    CHECK(reshaped.unit_rank() == (rank == 0 ? 0 : rank - 1));
  }

  // Composition must preserve static metadata while excluding the last two units
  // of each four-unit subgroup.
  const auto composed = (cudax::coop::group_by{cuda::std::cw<4>} | cudax::coop::take{cuda::std::cw<2>})
                          .map(cudax::coop::non_exhaustive, cuda::gpu_thread, parent, prev);

  static_assert(decltype(composed)::static_group_count() == 8);
  static_assert(decltype(composed)::static_unit_count(7) == 2);

  CHECK(composed.is_valid() == (rank % 4 < 2));
  if (composed.is_valid())
  {
    CHECK(composed.group_rank() == lane / 4);
    CHECK(composed.unit_rank() == rank % 4);
  }

  // A pack of equal counts must also work through the uniform group_by path.
  using Uniform = cudax::coop::__mapping_result<cudax::coop::__static_unit_counts<8, 8, 8, 8>, true, true>;
  const Uniform uniform{lane / 8, lane % 8, cuda::device::lane_mask{0xffu << (lane / 8 * 8)}};
  const auto uniform_split = cudax::coop::group_by{cuda::std::cw<4>}.map(cuda::gpu_thread, parent, uniform);

  static_assert(decltype(uniform_split)::static_group_count() == 8);
  static_assert(decltype(uniform_split)::is_always_exhaustive());

  CHECK(uniform_split.group_rank() == lane / 4);
  CHECK(uniform_split.unit_rank() == lane % 4);
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    test_other_mappings(config);

    // Storage must preserve both exhaustive and contiguous guarantees independently.
    test_result_storage<true, true>();
    test_result_storage<true, false>();
    test_result_storage<false, true>();
    test_result_storage<false, false>();

    // Four divides every input count, for both exhaustive and non-exhaustive mapping.
    test_group_by_counts<4, true, true>(config);
    test_group_by_counts<4, true, false>(config);
    test_group_by_counts<4, false, true>(config);
    test_group_by_counts<4, false, false>(config);

    // Three leaves remainders in the first and last input groups.
    test_group_by_counts<3, true, false>(config);
    test_group_by_counts<3, false, false>(config);

    // Eight produces no subgroup from the first input group.
    test_group_by_counts<8, true, false>(config);
    test_group_by_counts<8, false, false>(config);

    // Thirty-two excludes every unit, yielding zero output groups.
    test_group_by_counts<32, true, false>(config);
    test_group_by_counts<32, false, false>(config);

    // Exercise lane-mask construction without a static contiguity guarantee.
    test_group_by_counts<4, true, true, false>(config);
    test_group_by_counts<3, true, false, false>(config);
  }
};

C2H_TEST("Mapping result with per-group static unit counts", "[group]")
{
  const cuda::stream stream{cuda::devices[0]};

  // Run with both static and runtime block dimensions.
  cuda::launch(stream, cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<32>()), TestKernel{});
  cuda::launch(stream, cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims(32)), TestKernel{});

  stream.sync();
}
