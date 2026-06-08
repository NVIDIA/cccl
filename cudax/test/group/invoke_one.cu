//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/atomic>
#include <cuda/devices>
#include <cuda/hierarchy>
#include <cuda/launch>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <cuda/experimental/group.cuh>

#include "group_testing.cuh"

namespace
{
__device__ cuda::std::size_t invoke_count;
__device__ int global_value = 1;

__device__ void update_invoke_count() noexcept
{
  cuda::atomic_ref<cuda::std::size_t, cuda::thread_scope_device>
  {
    invoke_count
  }
  ++;
}

template <class Group>
__device__ void check_and_reset_invoke_count(const Group& group)
{
  group.sync_aligned();
  __threadfence();

  REQUIRE(cuda::atomic_ref<cuda::std::size_t, cuda::thread_scope_device>{invoke_count} == 1);
  __threadfence();

  if (cuda::gpu_thread.is_root_rank(group))
  {
    cuda::atomic_ref<cuda::std::size_t, cuda::thread_scope_device>{invoke_count} = 0;
  }
  __threadfence();
  group.sync_aligned();
}

template <class Group>
__device__ void test_invoke_one(const Group& group)
{
  // We need only 1 group for these tests.
  if (group.rank(cuda::grid) > 0)
  {
    return;
  }

  // invoke_one callable with void return type
  {
    auto callable = []() {
      update_invoke_count();
    };

    static_assert(cuda::std::is_same_v<void, decltype(cudax::invoke_one(group, callable))>);
    static_assert(!noexcept(cudax::invoke_one(group, callable)));

    cudax::invoke_one(group, callable);

    check_and_reset_invoke_count(group);
  }

  // invoke_one nothrow callable with void return type
  {
    auto callable = []() noexcept {
      update_invoke_count();
    };

    static_assert(cuda::std::is_same_v<void, decltype(cudax::invoke_one(group, callable))>);
    static_assert(noexcept(cudax::invoke_one(group, callable)));

    cudax::invoke_one(group, callable);

    check_and_reset_invoke_count(group);
  }

  // invoke_one callable with value return type
  {
    auto callable = []() -> int {
      update_invoke_count();
      return 1;
    };

    static_assert(cuda::std::is_same_v<cuda::std::optional<int>, decltype(cudax::invoke_one(group, callable))>);
    static_assert(!noexcept(cudax::invoke_one(group, callable)));

    const auto ret = cudax::invoke_one(group, callable);
    REQUIRE(ret.has_value() == cudax::__elect_one(group));
    if (ret.has_value())
    {
      REQUIRE(ret == 1);
    }

    check_and_reset_invoke_count(group);
  }

  // invoke_one nothrow callable with value return type
  {
    auto callable = []() noexcept -> int {
      update_invoke_count();
      return 1;
    };

    static_assert(cuda::std::is_same_v<cuda::std::optional<int>, decltype(cudax::invoke_one(group, callable))>);
    static_assert(noexcept(cudax::invoke_one(group, callable)));

    const auto ret = cudax::invoke_one(group, callable);
    REQUIRE(ret.has_value() == cudax::__elect_one(group));
    if (ret.has_value())
    {
      REQUIRE(ret == 1);
    }

    check_and_reset_invoke_count(group);
  }

  // invoke_one callable with l-value reference return type
  {
    auto callable = []() -> int& {
      update_invoke_count();
      return global_value;
    };

    static_assert(cuda::std::is_same_v<cuda::std::optional<int&>, decltype(cudax::invoke_one(group, callable))>);
    static_assert(!noexcept(cudax::invoke_one(group, callable)));

    const auto ret = cudax::invoke_one(group, callable);
    REQUIRE(ret.has_value() == cudax::__elect_one(group));
    if (ret.has_value())
    {
      REQUIRE(ret == 1);
      REQUIRE(&ret.value() == &global_value);
    }

    check_and_reset_invoke_count(group);
  }

  // invoke_one callable with l-value reference return type
  {
    auto callable = []() noexcept -> int& {
      update_invoke_count();
      return global_value;
    };

    static_assert(cuda::std::is_same_v<cuda::std::optional<int&>, decltype(cudax::invoke_one(group, callable))>);
    static_assert(noexcept(cudax::invoke_one(group, callable)));

    const auto ret = cudax::invoke_one(group, callable);
    REQUIRE(ret.has_value() == cudax::__elect_one(group));
    if (ret.has_value())
    {
      REQUIRE(ret == 1);
      REQUIRE(&ret.value() == &global_value);
    }

    check_and_reset_invoke_count(group);
  }

  // invoke_one callable with r-value reference return type
  {
    auto callable = []() -> int&& {
      update_invoke_count();
      return cuda::std::move(global_value);
    };

    static_assert(cuda::std::is_same_v<cuda::std::optional<int>, decltype(cudax::invoke_one(group, callable))>);
    static_assert(!noexcept(cudax::invoke_one(group, callable)));

    const auto ret = cudax::invoke_one(group, callable);
    REQUIRE(ret.has_value() == cudax::__elect_one(group));
    if (ret.has_value())
    {
      REQUIRE(ret == 1);
    }

    check_and_reset_invoke_count(group);
  }

  // invoke_one nothrow callable with r-value reference return type
  {
    auto callable = []() noexcept -> int&& {
      update_invoke_count();
      return cuda::std::move(global_value);
    };

    static_assert(cuda::std::is_same_v<cuda::std::optional<int>, decltype(cudax::invoke_one(group, callable))>);
    static_assert(noexcept(cudax::invoke_one(group, callable)));

    const auto ret = cudax::invoke_one(group, callable);
    REQUIRE(ret.has_value() == cudax::__elect_one(group));
    if (ret.has_value())
    {
      REQUIRE(ret == 1);
    }

    check_and_reset_invoke_count(group);
  }

  // Check that invoke_one correctly forwards the arguments for invocable with void return type.
  {
    auto callable = [](auto&& arg1, auto&& arg2) -> void {
      static_assert(cuda::std::is_same_v<int&, decltype(arg1)>);
      static_assert(cuda::std::is_same_v<unsigned&&, decltype(arg2)>);

      REQUIRE(arg1 == 2);
      REQUIRE(arg2 == 20u);
    };

    int arg1{2};
    unsigned arg2{20};
    cudax::invoke_one(group, callable, arg1, cuda::std::move(arg2));
  }

  // Check that invoke_one correctly forwards the arguments for invocable with non-void return type.
  {
    auto callable = [](auto&& arg1, auto&& arg2) -> int {
      static_assert(cuda::std::is_same_v<int&, decltype(arg1)>);
      static_assert(cuda::std::is_same_v<unsigned&&, decltype(arg2)>);

      REQUIRE(arg1 == 2);
      REQUIRE(arg2 == 20u);

      return 1;
    };

    int arg1{2};
    unsigned arg2{20};
    const auto ret = cudax::invoke_one(group, callable, arg1, cuda::std::move(arg2));

    REQUIRE(ret.has_value() == cudax::__elect_one(group));
    if (ret.has_value())
    {
      REQUIRE(ret == 1);
    }
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    // Test this groups.
    test_invoke_one(cudax::this_thread{config});
    test_invoke_one(cudax::this_warp{config});
    test_invoke_one(cudax::this_block{config});
    test_invoke_one(cudax::this_cluster{config});

    // Test custom groups.
    {
      cudax::group group{cuda::gpu_thread, cudax::this_warp{config}, cudax::group_by<4>{}, cudax::lane_synchronizer{}};
      test_invoke_one(group);
    }
  }
};
} // namespace

C2H_TEST("Invoke one", "[group]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  const auto config = cuda::make_config(cuda::grid_dims<2>(), cuda::block_dims<128>(), cuda::cooperative_launch{});
  cuda::launch(stream, config, TestKernel{});

  if (cuda::device_attributes::compute_capability(device) >= cuda::compute_capability{90})
  {
    const auto config_cluster = cuda::make_config(
      cuda::grid_dims<2>(), cuda::cluster_dims<3>(), cuda::block_dims<128>(), cuda::cooperative_launch{});
    cuda::launch(stream, config_cluster, TestKernel{});
  }

  stream.sync();
}
