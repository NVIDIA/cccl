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
#include <cuda/std/ranges>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>
#include <cuda/type_traits>

#include <cuda/experimental/coop/group>

#include "group_testing.cuh"

// Mutable iteration yields Barrier&, while const iteration yields const Barrier&.
template <class Barrier>
struct ConstPropagatingBarrierView : NonConstView<Barrier>
{
  using NonConstView<Barrier>::NonConstView;
  using NonConstView<Barrier>::begin;
  using NonConstView<Barrier>::end;

  __host__ __device__ const Barrier* begin() const
  {
    return this->values;
  }

  __host__ __device__ const Barrier* end() const
  {
    return this->values + this->count;
  }
};

_CCCL_BEGIN_NAMESPACE_CUDA_STD_RANGES

template <class Barrier>
inline constexpr bool enable_borrowed_range<ConstPropagatingBarrierView<Barrier>> = true;

_CCCL_END_NAMESPACE_CUDA_STD_RANGES

namespace
{
template <class Synchronizer, class ParentGroup, class MappingResult>
__device__ void test_synchronizer_instance(
  Synchronizer& synchronizer, const ParentGroup& parent_group, const MappingResult& mapping_result)
{
  const auto& hierarchy = parent_group.hierarchy();

  auto synchronizer_instance = synchronizer.make_instance(cuda::gpu_thread, parent_group, mapping_result);

  using SynchronizerInstance = decltype(synchronizer_instance);

  static_assert(cuda::is_trivially_copyable_v<SynchronizerInstance>);
  static_assert(cuda::std::is_trivially_destructible_v<SynchronizerInstance>);

  using ArrivalToken = typename SynchronizerInstance::arrival_token;

  // Test arrival_token type.
  static_assert(cuda::std::is_move_constructible_v<ArrivalToken>);
  static_assert(cuda::std::is_move_assignable_v<ArrivalToken>);
  static_assert(cuda::std::is_destructible_v<ArrivalToken>);

  // Test do_sync(...).
  static_assert(cuda::std::is_same_v<void, decltype(synchronizer_instance.do_sync(hierarchy))>);
  synchronizer_instance.do_sync(hierarchy);

  // Test do_arrive(...).
  static_assert(cuda::std::is_same_v<ArrivalToken, decltype(synchronizer_instance.do_arrive(hierarchy))>);
  auto arrival_token = synchronizer_instance.do_arrive(hierarchy);

  // Test do_wait(...).
  static_assert(
    cuda::std::is_same_v<void, decltype(synchronizer_instance.do_wait(hierarchy, cuda::std::move(arrival_token)))>);
  synchronizer_instance.do_wait(hierarchy, cuda::std::move(arrival_token));

  // Test do_sync_aligned(...).
  static_assert(cuda::std::is_same_v<void, decltype(synchronizer_instance.do_sync_aligned(hierarchy))>);
  synchronizer_instance.do_sync_aligned(hierarchy);

  // Test do_arrive_aligned(...).
  static_assert(cuda::std::is_same_v<ArrivalToken, decltype(synchronizer_instance.do_arrive_aligned(hierarchy))>);
  auto arrival_token_a = synchronizer_instance.do_arrive_aligned(hierarchy);

  // Test do_wait_aligned(...).
  static_assert(
    cuda::std::is_same_v<void,
                         decltype(synchronizer_instance.do_wait_aligned(hierarchy, cuda::std::move(arrival_token_a)))>);
  synchronizer_instance.do_wait_aligned(hierarchy, cuda::std::move(arrival_token_a));

  // Test deinit(...);
  static_assert(cuda::std::is_same_v<void, decltype(synchronizer_instance.deinit(mapping_result, hierarchy))>);
  synchronizer_instance.deinit(mapping_result, hierarchy);
  parent_group.sync();
}

template <bool ConstCallable, class Range, class ParentGroup, class MappingResult>
__device__ void
test_barrier_range(Range&& barriers, const ParentGroup& parent_group, const MappingResult& mapping_result)
{
  using View         = cuda::std::views::all_t<Range>;
  using Synchronizer = cudax::coop::barrier_synchronizer<View>;

  cudax::coop::barrier_synchronizer synchronizer{cuda::std::forward<Range>(barriers)};
  static_assert(cuda::std::is_same_v<decltype(synchronizer), Synchronizer>);
  static_assert(cuda::std::is_constructible_v<Synchronizer, Range>);
  static_assert(can_make_synchronizer_instance<Synchronizer, cuda::thread_level, ParentGroup, MappingResult>);
  static_assert(can_make_synchronizer_instance<const Synchronizer, cuda::thread_level, ParentGroup, MappingResult>
                == ConstCallable);

  test_synchronizer_instance(synchronizer, parent_group, mapping_result);
  if constexpr (ConstCallable)
  {
    test_synchronizer_instance(cuda::std::as_const(synchronizer), parent_group, mapping_result);
  }
}

template <class Level, class Config>
__device__ void test_barrier_synchronizer(const Level& level, Config config)
{
  constexpr cuda::std::size_t nbarriers = 8;
  auto& barriers                        = get_barriers<nbarriers>(level);
  using Barrier = cuda::std::remove_all_extents_t<cuda::std::remove_reference_t<decltype(barriers)>>;

  const auto parent_group = cudax::coop::make_this_group(level, config);
  const ThreadsInWarpMappingResult prev_mapping_result;
  const cudax::coop::group_by mapping{4};
  const auto mapping_result = mapping.map(cuda::gpu_thread, parent_group, prev_mapping_result);

  // Array and span views support both mutable and const synchronizers.
  test_barrier_range<true>(barriers, parent_group, mapping_result);
  test_barrier_range<true>(cuda::std::span{barriers}, parent_group, mapping_result);

  // A view without const iteration requires a mutable synchronizer.
  test_barrier_range<false>(NonConstView<Barrier>{barriers, nbarriers}, parent_group, mapping_result);

  // Const iteration exists, but it exposes const barriers and cannot initialize them.
  using ConstView = ConstPropagatingBarrierView<Barrier>;
  static_assert(cuda::std::ranges::borrowed_range<ConstView>);
  static_assert(cuda::std::ranges::sized_range<const ConstView>);
  static_assert(cuda::std::ranges::forward_range<const ConstView>);
  static_assert(cuda::std::is_same_v<cuda::std::ranges::range_reference_t<ConstView>, Barrier&>);
  static_assert(cuda::std::is_same_v<cuda::std::ranges::range_reference_t<const ConstView>, const Barrier&>);
  test_barrier_range<false>(ConstView{barriers, nbarriers}, parent_group, mapping_result);
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    test_barrier_synchronizer(cuda::warp, config);
    test_barrier_synchronizer(cuda::block, config);
    test_barrier_synchronizer(cuda::cluster, config);
    test_barrier_synchronizer(cuda::grid, config);
  }
};
} // namespace

C2H_TEST("Barrier synchronizer", "[group]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  {
    const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<8, 4>(), cuda::cooperative_launch{});
    cuda::launch(stream, config, TestKernel{});
  }
  {
    const auto config =
      cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims(dim3{8, 4}), cuda::cooperative_launch{});
    cuda::launch(stream, config, TestKernel{});
  }

  stream.sync();
}
