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
#include <cuda/std/cstddef>
#include <cuda/std/numeric>
#include <cuda/std/span>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>
#include <cuda/warp>

#include <cuda/experimental/coop/group>

#include "group_testing.cuh"

namespace
{
template <class Mapping1, class Mapping2, class Config, class... NonExhaustive>
__device__ void test_composite_mapping(
  const Mapping1& mapping1, const Mapping2& mapping2, Config config, const NonExhaustive&... non_exhaustive)
{
  using Mapping = cudax::coop::composite_mapping<Mapping1, Mapping2>;

  // Test construction from 2 mappings.
  {
    // NOLINTNEXTLINE(misc-const-correctness): decltype must not be const-qualified
    cudax::coop::composite_mapping mapping{mapping1, mapping2};
    static_assert(cuda::std::is_same_v<decltype(mapping), Mapping>);
    static_assert(cuda::std::is_nothrow_constructible_v<Mapping, Mapping1, Mapping2>
                  == (cuda::std::is_nothrow_copy_constructible_v<Mapping1>
                      && cuda::std::is_nothrow_copy_constructible_v<Mapping2>) );
  }

  // Test get().
  {
    const cudax::coop::composite_mapping mapping{mapping1, mapping2};
    static_assert(cuda::std::is_same_v<decltype(mapping.get()), const cuda::std::tuple<Mapping1, Mapping2>&>);
    static_assert(noexcept(mapping.get()));

    decltype(auto) mapping1_ref = cuda::std::get<0>(mapping.get());
    decltype(auto) mapping2_ref = cuda::std::get<1>(mapping.get());

    static_assert(cuda::std::is_same_v<decltype(mapping1_ref), const Mapping1&>);
    static_assert(cuda::std::is_same_v<decltype(mapping2_ref), const Mapping2&>);
  }

  // Test map(...).
  {
    const cudax::coop::this_warp parent_group{config};
    const ThreadsInWarpMappingResult prev_mapping_result;
    cudax::coop::composite_mapping mapping{mapping1, mapping2};

    static_assert(cudax::coop::__group_mapping_result<decltype(mapping.map(
                    non_exhaustive..., cuda::gpu_thread, parent_group, prev_mapping_result))>);

    static_assert(noexcept(mapping.map(non_exhaustive..., cuda::gpu_thread, parent_group, prev_mapping_result)));

    auto result  = mapping.map(non_exhaustive..., cuda::gpu_thread, parent_group, prev_mapping_result);
    using Result = decltype(result);

    const auto rank_in_warp = cuda::gpu_thread.rank_as<unsigned>(parent_group);

    static_assert(Result::static_group_count() == cuda::std::dynamic_extent);
    CHECK(result.group_count() == 16);
    CHECK(result.group_rank() == (rank_in_warp / 4 * 2 + (rank_in_warp % 4 > 0)));

    static_assert(Result::static_unit_count() == cuda::std::dynamic_extent);
    CHECK(result.unit_count() == ((rank_in_warp % 4 > 0) ? 3 : 1));
    CHECK(result.unit_rank() == ((rank_in_warp % 4 > 0) ? (rank_in_warp % 4 - 1) : 0));

    const auto lane_mask_ref = ((rank_in_warp % 4 > 0) ? 0b1110u : 0b0001u) << ((rank_in_warp / 4) * 4);
    CHECK(result.lane_mask() == cuda::device::lane_mask{lane_mask_ref});

    CHECK(result.is_valid());
    static_assert(Result::is_always_exhaustive() == (sizeof...(NonExhaustive) == 0));
    static_assert(Result::is_always_contiguous());
  }

  // Test operator|.
  {
    auto mapping = mapping1 | mapping2;

    static_assert(cuda::std::is_same_v<Mapping, decltype(mapping)>);
    static_assert(noexcept(mapping1 | mapping2));

    decltype(auto) mapping1_ref = cuda::std::get<0>(mapping.get());
    decltype(auto) mapping2_ref = cuda::std::get<1>(mapping.get());

    static_assert(cuda::std::is_same_v<decltype(mapping1_ref), const Mapping1&>);
    static_assert(cuda::std::is_same_v<decltype(mapping2_ref), const Mapping2&>);
  }
  {
    auto mapping = cudax::coop::composite_mapping{mapping1} | mapping2;

    static_assert(cuda::std::is_same_v<Mapping, decltype(mapping)>);
    static_assert(noexcept(cudax::coop::composite_mapping{mapping1} | mapping2));

    decltype(auto) mapping1_ref = cuda::std::get<0>(mapping.get());
    decltype(auto) mapping2_ref = cuda::std::get<1>(mapping.get());

    static_assert(cuda::std::is_same_v<decltype(mapping1_ref), const Mapping1&>);
    static_assert(cuda::std::is_same_v<decltype(mapping2_ref), const Mapping2&>);
  }
  {
    auto mapping = mapping1 | cudax::coop::composite_mapping{mapping2};

    static_assert(cuda::std::is_same_v<Mapping, decltype(mapping)>);
    static_assert(noexcept(mapping1 | cudax::coop::composite_mapping{mapping2}));

    decltype(auto) mapping1_ref = cuda::std::get<0>(mapping.get());
    decltype(auto) mapping2_ref = cuda::std::get<1>(mapping.get());

    static_assert(cuda::std::is_same_v<decltype(mapping1_ref), const Mapping1&>);
    static_assert(cuda::std::is_same_v<decltype(mapping2_ref), const Mapping2&>);
  }
  {
    auto mapping = cudax::coop::composite_mapping{mapping1} | cudax::coop::composite_mapping{mapping2};

    static_assert(cuda::std::is_same_v<Mapping, decltype(mapping)>);
    static_assert(noexcept(cudax::coop::composite_mapping{mapping1} | cudax::coop::composite_mapping{mapping2}));

    decltype(auto) mapping1_ref = cuda::std::get<0>(mapping.get());
    decltype(auto) mapping2_ref = cuda::std::get<1>(mapping.get());

    static_assert(cuda::std::is_same_v<decltype(mapping1_ref), const Mapping1&>);
    static_assert(cuda::std::is_same_v<decltype(mapping2_ref), const Mapping2&>);
  }
}

template <class Mapping, class Config, class... NonExhaustive>
__device__ void test_identity_composite_mapping(Mapping& mapping, Config config, const NonExhaustive&... non_exhaustive)
{
  const cudax::coop::this_warp parent_group{config};
  const ThreadsInWarpMappingResult prev_mapping_result;
  static_assert(noexcept(mapping.map(non_exhaustive..., cuda::gpu_thread, parent_group, prev_mapping_result)));

  const auto result = mapping.map(non_exhaustive..., cuda::gpu_thread, parent_group, prev_mapping_result);
  static_assert(cuda::std::is_same_v<decltype(result), const ThreadsInWarpMappingResult>);
  CHECK(result.group_count() == prev_mapping_result.group_count());
  CHECK(result.group_rank() == prev_mapping_result.group_rank());
  CHECK(result.unit_count() == prev_mapping_result.unit_count());
  CHECK(result.unit_rank() == prev_mapping_result.unit_rank());
  CHECK(result.lane_mask() == prev_mapping_result.lane_mask());
  CHECK(result.is_valid() == prev_mapping_result.is_valid());
}

template <class Config>
__device__ void test_non_exhaustive_composite_mapping(Config config)
{
  const cudax::coop::this_warp parent_group{config};
  const ThreadsInWarpMappingResult prev_mapping_result;
  // The first mapping excludes the incomplete group; the second excludes each group's last unit.
  auto mapping = cudax::coop::group_by{cuda::std::cw<3>} | cudax::coop::take{cuda::std::cw<2>}
               | cudax::coop::identity_mapping{};

  static_assert(
    noexcept(mapping.map(cudax::coop::non_exhaustive, cuda::gpu_thread, parent_group, prev_mapping_result)));
  const auto result = mapping.map(cudax::coop::non_exhaustive, cuda::gpu_thread, parent_group, prev_mapping_result);
  using Result      = decltype(result);

  static_assert(Result::static_group_count() == 10);
  static_assert(Result::static_unit_count() == 2);
  static_assert(!Result::is_always_exhaustive());
  static_assert(Result::is_always_contiguous());

  const auto lane         = cuda::gpu_thread.rank_as<unsigned>(cuda::warp);
  const auto is_valid_ref = lane < 30 && lane % 3 < 2;
  CHECK(result.is_valid() == is_valid_ref);
  if (is_valid_ref)
  {
    CHECK(result.group_count() == 10);
    CHECK(result.group_rank() == lane / 3);
    CHECK(result.unit_count() == 2);
    CHECK(result.unit_rank() == lane % 3);
    CHECK(result.lane_mask() == cuda::device::lane_mask{0b11u << (lane / 3 * 3)});
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    cudax::coop::composite_mapping empty_mapping{};
    static_assert(cuda::std::is_same_v<decltype(empty_mapping.get()), const cuda::std::tuple<>&>);
    test_identity_composite_mapping(empty_mapping, config);
    test_identity_composite_mapping(empty_mapping, config, cudax::coop::non_exhaustive);

    cudax::coop::composite_mapping identity_mapping{cudax::coop::identity_mapping{}};
    test_identity_composite_mapping(identity_mapping, config);
    test_identity_composite_mapping(identity_mapping, config, cudax::coop::non_exhaustive);
    test_non_exhaustive_composite_mapping(config);

    {
      const cudax::coop::group_by mapping1{cuda::std::cw<4>};
      constexpr unsigned counts2[]{1, 3};
      const cudax::coop::group_as mapping2{counts2};
      test_composite_mapping(mapping1, mapping2, config);
      test_composite_mapping(mapping1, mapping2, config, cudax::coop::non_exhaustive);
    }
    {
      const cudax::coop::group_by mapping1{4};
      constexpr unsigned counts2[]{1, 3};
      const cudax::coop::group_as mapping2{counts2};
      test_composite_mapping(mapping1, mapping2, config);
      test_composite_mapping(mapping1, mapping2, config, cudax::coop::non_exhaustive);
    }
    {
      const cudax::coop::group_by mapping1{cuda::std::cw<4>};
      constexpr unsigned counts2[]{1, 3};
      const cudax::coop::group_as mapping2{counts2};
      test_composite_mapping(mapping1, mapping2, config);
      test_composite_mapping(mapping1, mapping2, config, cudax::coop::non_exhaustive);
    }
    {
      const cudax::coop::group_by mapping1{4};
      constexpr unsigned counts2[]{1, 3};
      const cudax::coop::group_as mapping2{counts2};
      test_composite_mapping(mapping1, mapping2, config);
      test_composite_mapping(mapping1, mapping2, config, cudax::coop::non_exhaustive);
    }
  }
};
} // namespace

C2H_TEST("Composite mapping", "[group]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  {
    const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<8, 4>());
    cuda::launch(stream, config, TestKernel{});
  }
  {
    const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims(dim3{8, 4}));
    cuda::launch(stream, config, TestKernel{});
  }

  stream.sync();
}
