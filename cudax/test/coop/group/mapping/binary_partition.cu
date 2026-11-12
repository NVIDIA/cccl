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
#include <cuda/std/bit>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>
#include <cuda/warp>

#include <cuda/experimental/coop/group>

#include "group_testing.cuh"

namespace
{
struct AlwaysTruePredFn
{
  template <class MappingResult>
  __device__ bool operator()(const MappingResult&) const
  {
    return true;
  }
};

struct AlwaysFalsePredFn
{
  template <class MappingResult>
  __device__ bool operator()(const MappingResult&) const
  {
    return false;
  }
};

struct IsEvenPredFn
{
  template <class MappingResult>
  __device__ bool operator()(const MappingResult& mapping_result) const
  {
    return mapping_result.unit_rank() % 2 == 0;
  }
};

struct MutablePredFn
{
  unsigned* calls;

  template <class MappingResult>
  __device__ bool operator()(const MappingResult& mapping_result) &
  {
    ++*calls;
    return mapping_result.unit_rank() % 2 == 0;
  }
};

struct OverloadedPredFn : MutablePredFn
{
  using MutablePredFn::operator();

  template <class MappingResult>
  __device__ bool operator()(const MappingResult& mapping_result) const&
  {
    ++*calls;
    return mapping_result.unit_rank() % 2 != 0;
  }
};

struct ConstOnlyPredFn
{
  template <class MappingResult>
  __device__ bool operator()(const MappingResult&) & = delete;

  template <class MappingResult>
  __device__ bool operator()(const MappingResult& mapping_result) const&
  {
    return mapping_result.unit_rank() % 2 == 0;
  }
};

struct ConvertibleBool
{
  bool value;

  // NOLINTNEXTLINE(cppcoreguidelines-explicit-constructor, misc-explicit-constructor): test implicit conversion.
  __device__ operator bool() const
  {
    return value;
  }
};

struct ConvertiblePredFn
{
  template <class MappingResult>
  __device__ ConvertibleBool operator()(const MappingResult& mapping_result) const
  {
    return {mapping_result.unit_rank() % 2 == 0};
  }
};

template <class Mapping, class ParentGroup, class... NonExhaustive>
__device__ void check_binary_partition(
  Mapping& mapping, const ParentGroup& parent_group, unsigned true_mask, const NonExhaustive&... non_exhaustive)
{
  const ThreadsInWarpMappingResult prev_mapping_result;
  const auto result = mapping.map(non_exhaustive..., cuda::gpu_thread, parent_group, prev_mapping_result);
  using Result      = decltype(result);

  static_assert(cudax::coop::__group_mapping_result<Result>);
  static_assert(Result::static_group_count() == 2);
  static_assert(Result::static_unit_count() == cuda::std::dynamic_extent);
  static_assert(Result::is_always_exhaustive());
  static_assert(!Result::is_always_contiguous());

  const auto lane = cuda::gpu_thread.rank_as<unsigned>(cuda::warp);
  const bool pred = ((true_mask >> lane) & 1u) != 0;
  const auto mask = pred ? true_mask : ~true_mask;
  REQUIRE(result.is_valid());
  REQUIRE(result.group_count() == 2);
  REQUIRE(result.group_rank() == static_cast<unsigned>(pred));
  REQUIRE(result.unit_count() == cuda::std::popcount(mask));
  REQUIRE(result.unit_rank() == cuda::std::popcount(mask & cuda::device::lane_mask::all_less().value()));
  REQUIRE(result.lane_mask() == cuda::device::lane_mask{mask});
}

template <bool ConstCallable, class Config, class Pred, class... NonExhaustive>
__device__ void test_predicate(
  Config config, Pred pred, unsigned mutable_mask, unsigned const_mask, const NonExhaustive&... non_exhaustive)
{
  using Mapping = cudax::coop::binary_partition<Pred>;
  cudax::coop::binary_partition mapping{pred};
  static_assert(cuda::std::is_same_v<decltype(mapping), Mapping>);
  static_assert(cuda::std::is_constructible_v<Mapping, Pred>);

  const cudax::coop::this_warp parent_group{config};
  using Parent = decltype(parent_group);
  static_assert(can_map_group<Mapping, NonExhaustive..., cuda::thread_level, Parent, ThreadsInWarpMappingResult>);
  static_assert(can_map_group<const Mapping, NonExhaustive..., cuda::thread_level, Parent, ThreadsInWarpMappingResult>
                == ConstCallable);
  using UncallableMapping = cudax::coop::binary_partition<int>;
  static_assert(
    !can_map_group<UncallableMapping, NonExhaustive..., cuda::thread_level, Parent, ThreadsInWarpMappingResult>);
  static_assert(
    !can_map_group<const UncallableMapping, NonExhaustive..., cuda::thread_level, Parent, ThreadsInWarpMappingResult>);

  check_binary_partition(mapping, parent_group, mutable_mask, non_exhaustive...);
  if constexpr (ConstCallable)
  {
    check_binary_partition(cuda::std::as_const(mapping), parent_group, const_mask, non_exhaustive...);
  }
}

template <class Config, class... NonExhaustive>
__device__ void test_binary_partition(Config config, const NonExhaustive&... non_exhaustive)
{
  constexpr unsigned even_mask = 0x5555'5555u;
  constexpr unsigned odd_mask  = 0xaaaa'aaaau;
  test_predicate<true>(config, AlwaysTruePredFn{}, ~0u, ~0u, non_exhaustive...);
  test_predicate<true>(config, AlwaysFalsePredFn{}, 0u, 0u, non_exhaustive...);
  test_predicate<true>(config, IsEvenPredFn{}, even_mask, even_mask, non_exhaustive...);
  test_predicate<true>(config, ConvertiblePredFn{}, even_mask, even_mask, non_exhaustive...);
  test_predicate<true>(config, ConstOnlyPredFn{}, even_mask, even_mask, non_exhaustive...);

  unsigned calls = 0;
  test_predicate<false>(config, MutablePredFn{&calls}, even_mask, 0u, non_exhaustive...);
  REQUIRE(calls == 1);

  calls = 0;
  test_predicate<true>(config, OverloadedPredFn{{&calls}}, even_mask, odd_mask, non_exhaustive...);
  REQUIRE(calls == 2);
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    test_binary_partition(config);
    test_binary_partition(config, cudax::coop::non_exhaustive);
  }
};
} // namespace

C2H_TEST("Binary partition mapping", "[group]")
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
