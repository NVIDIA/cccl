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

#include "testing.cuh"

template <class Level, class Config>
__device__ void test_barrier_synchronizer(const Level& level, Config config)
{
  constexpr cuda::std::size_t nbarriers = 8;

  using Barrier         = cuda::barrier<cuda::thread_scope_block>;
  using BarriersStorage = cuda::std::aligned_storage_t<8 * sizeof(Barrier), alignof(Barrier)>;
  __shared__ BarriersStorage barriers_storage;
  auto& barriers = reinterpret_cast<Barrier(&)[nbarriers]>(barriers_storage);

  // Test constructor from static span of barriers.
  {
    cuda::std::span<Barrier, nbarriers> barriers_span{barriers, nbarriers};
    cudax::barrier_synchronizer synchronizer{barriers_span};

    static_assert(cuda::std::is_same_v<cudax::barrier_synchronizer<Barrier, nbarriers>, decltype(synchronizer)>);
    static_assert(cuda::std::is_nothrow_constructible_v<decltype(synchronizer), decltype(barriers_span)>);

    CUDAX_CHECK(synchronizer.barriers().data() == barriers);
    CUDAX_CHECK(synchronizer.barriers().size() == nbarriers);
  }

  // Test constructor from dynamic span of barriers.
  {
    cuda::std::span<Barrier> barriers_span{barriers, nbarriers};
    cudax::barrier_synchronizer synchronizer{barriers_span};

    static_assert(
      cuda::std::is_same_v<cudax::barrier_synchronizer<Barrier, cuda::std::dynamic_extent>, decltype(synchronizer)>);
    static_assert(cuda::std::is_nothrow_constructible_v<decltype(synchronizer), decltype(barriers_span)>);

    CUDAX_CHECK(synchronizer.barriers().data() == barriers);
    CUDAX_CHECK(synchronizer.barriers().size() == nbarriers);
  }

  // Test constructor from array of barriers.
  {
    cudax::barrier_synchronizer synchronizer{barriers};

    static_assert(cuda::std::is_same_v<cudax::barrier_synchronizer<Barrier, nbarriers>, decltype(synchronizer)>);
    static_assert(cuda::std::is_nothrow_constructible_v<decltype(synchronizer), decltype(barriers)>);

    CUDAX_CHECK(synchronizer.barriers().data() == barriers);
    CUDAX_CHECK(synchronizer.barriers().size() == nbarriers);
  }

  // Test barriers().
  {
    const cudax::barrier_synchronizer synchronizer{barriers};

    static_assert(cuda::std::is_same_v<cuda::std::span<Barrier, nbarriers>, decltype(synchronizer.barriers())>);
    static_assert(noexcept(synchronizer.barriers()));

    CUDAX_CHECK(synchronizer.barriers().data() == barriers);
    CUDAX_CHECK(synchronizer.barriers().size() == nbarriers);
  }

  // Test make_instance(...).
  {
    const cudax::group_by mapping{4};
    const cudax::barrier_synchronizer synchronizer{barriers};

    const auto mapping_result        = mapping.map(cuda::gpu_thread, level, config.hierarchy());
    const auto synchronizer_instance = synchronizer.make_instance(cuda::gpu_thread, level, mapping, mapping_result);

    // Test do_sync(...).
    static_assert(cuda::std::is_same_v<void, decltype(synchronizer_instance.do_sync(mapping_result, synchronizer))>);
    static_assert(noexcept(synchronizer_instance.do_sync(mapping_result, synchronizer)));
    synchronizer_instance.do_sync(mapping_result, synchronizer);

    // Test do_sync_aligned(...).
    static_assert(
      cuda::std::is_same_v<void, decltype(synchronizer_instance.do_sync_aligned(mapping_result, synchronizer))>);
    static_assert(noexcept(synchronizer_instance.do_sync_aligned(mapping_result, synchronizer)));
    synchronizer_instance.do_sync_aligned(mapping_result, synchronizer);
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    // todo(dabayer): Test other levels once supported.
    test_barrier_synchronizer(cuda::block, config);
  }
};

C2H_TEST("Barrier synchronizer", "[group]")
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
