//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/barrier>
#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cuda/experimental/group.cuh>

#include "group_testing.cuh"

struct WarpsInBlockMappingResult
{
  __device__ static constexpr ::cuda::std::size_t static_group_count()
  {
    return 1;
  }

  __device__ unsigned group_count() const
  {
    return 1;
  }

  __device__ unsigned group_rank() const
  {
    return 0;
  }

  __device__ static constexpr ::cuda::std::size_t static_unit_count()
  {
    return cuda::std::dynamic_extent;
  }

  __device__ unsigned unit_count() const
  {
    return cuda::warp.count_as<unsigned>(cuda::block);
  }

  __device__ unsigned unit_rank() const
  {
    return cuda::warp.rank_as<unsigned>(cuda::block);
  }

  __device__ cuda::device::lane_mask lane_mask() const noexcept
  {
    return cuda::device::lane_mask::all();
  }

  __device__ bool is_valid() const
  {
    return true;
  }

  __device__ static constexpr bool is_always_exhaustive() noexcept
  {
    return true;
  }

  __device__ static constexpr bool is_always_contiguous() noexcept
  {
    return true;
  }
};

template <class Config>
__device__ void test_interwarp_synchronizer(Config config)
{
  const auto& hierarchy = config.hierarchy();

  const int barrier_ids[]{5, 3, 15, 1};

  using Synchronizer = decltype(cudax::interwarp_synchronizer{barrier_ids});

  // Test default constructor.
  static_assert(!cuda::std::is_default_constructible_v<Synchronizer>);

  // Test constructor from range of barrier ids.
  {
    static_assert(cuda::std::is_nothrow_constructible_v<Synchronizer, decltype(barrier_ids)>);
    [[maybe_unused]] const cudax::interwarp_synchronizer synchronizer{barrier_ids};
  }

  // Test make_instance(...).
  {
    const cudax::this_block parent_group{config};
    const WarpsInBlockMappingResult prev_mapping_result;

    const cudax::group_by mapping{2};
    const cudax::interwarp_synchronizer synchronizer{barrier_ids};

    const auto mapping_result  = mapping.map(cuda::warp, parent_group, prev_mapping_result);
    auto synchronizer_instance = synchronizer.make_instance(cuda::warp, parent_group, mapping_result);

    // Test do_sync(...).
    static_assert(cuda::std::is_same_v<void, decltype(synchronizer_instance.do_sync(mapping_result, hierarchy))>);
    static_assert(noexcept(synchronizer_instance.do_sync(mapping_result, hierarchy)));
    synchronizer_instance.do_sync(mapping_result, hierarchy);

    // Test do_sync_aligned(...).
    static_assert(
      cuda::std::is_same_v<void, decltype(synchronizer_instance.do_sync_aligned(mapping_result, hierarchy))>);
    static_assert(noexcept(synchronizer_instance.do_sync_aligned(mapping_result, hierarchy)));
    synchronizer_instance.do_sync_aligned(mapping_result, hierarchy);

    // Test view().
    static_assert(cuda::std::is_same_v<decltype(synchronizer_instance), decltype(synchronizer_instance.view())>);
    static_assert(noexcept(synchronizer_instance.view()));
    auto synchronizer_instance_view = synchronizer_instance.view();
    synchronizer_instance_view.do_sync(mapping_result, hierarchy);
    synchronizer_instance_view.do_sync_aligned(mapping_result, hierarchy);
    (void) synchronizer_instance_view.view();
    synchronizer_instance_view.deinit(mapping_result, hierarchy); // should be noop

    // Test deinit(...);
    static_assert(cuda::std::is_same_v<void, decltype(synchronizer_instance.deinit(mapping_result, hierarchy))>);
    static_assert(noexcept(synchronizer_instance.deinit(mapping_result, hierarchy)));
    synchronizer_instance.deinit(mapping_result, hierarchy);
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    test_interwarp_synchronizer(config);
  }
};

C2H_TEST("Barrier synchronizer", "[group]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  {
    const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<256>());
    cuda::launch(stream, config, TestKernel{});
  }
  {
    const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims(dim3{256}));
    cuda::launch(stream, config, TestKernel{});
  }

  stream.sync();
}
