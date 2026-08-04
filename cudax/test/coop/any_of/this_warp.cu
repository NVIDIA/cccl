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
#include <cuda/stream>

#include <cuda/experimental/coop.cuh>
#include <cuda/experimental/group.cuh>

#include "testing.cuh"

template <class Group>
__device__ void test_group(const Group& group)
{
  const auto my_rank = cuda::gpu_thread.rank_as<unsigned>(group);

  // Single value tests.
  {
    // Test all threads with false.
    {
      const auto result = cudax::coop::any_of(group, false);
      REQUIRE(result.has_value() == (my_rank == 0));
      if (result.has_value())
      {
        REQUIRE(!result.value());
      }
    }

    // Test all threads with true.
    {
      const auto result = cudax::coop::any_of(group, true);
      REQUIRE(result.has_value() == (my_rank == 0));
      if (result.has_value())
      {
        REQUIRE(result.value());
      }
    }

    // Test all threads except 1 with false.
    {
      const auto result = cudax::coop::any_of(group, (my_rank == cuda::gpu_thread.count_as<unsigned>(group) - 1));
      REQUIRE(result.has_value() == (my_rank == 0));
      if (result.has_value())
      {
        REQUIRE(result.value());
      }
    }

    // Test all threads with false (broadcasted).
    {
      const bool result = cudax::coop::any_of(cudax::broadcasted, group, false);
      REQUIRE(!result);
    }

    // Test all threads with true (broadcasted).
    {
      const bool result = cudax::coop::any_of(cudax::broadcasted, group, true);
      REQUIRE(result);
    }

    // Test all threads except 1 with false (broadcasted).
    {
      const bool result =
        cudax::coop::any_of(cudax::broadcasted, group, (my_rank == cuda::gpu_thread.count_as<unsigned>(group) - 1));
      REQUIRE(result);
    }
  }

  // Array tests.
  {
    // Test all threads with false.
    {
      bool in[]{false, false, false};
      const auto result = cudax::coop::any_of(group, in);
      REQUIRE(result.has_value() == (my_rank == 0));
      if (result.has_value())
      {
        REQUIRE(!result.value());
      }
    }

    // Test all threads with true.
    {
      bool in[]{true, true, true, true};
      const auto result = cudax::coop::any_of(group, in);
      REQUIRE(result.has_value() == (my_rank == 0));
      if (result.has_value())
      {
        REQUIRE(result.value());
      }
    }

    // Test all threads except 1 with false.
    {
      bool in[]{false, false, my_rank == cuda::gpu_thread.count_as<unsigned>(group) - 1};
      const auto result = cudax::coop::any_of(group, in);
      REQUIRE(result.has_value() == (my_rank == 0));
      if (result.has_value())
      {
        REQUIRE(result.value());
      }
    }

    // Test all threads with false (broadcasted).
    {
      bool in[]{false, false, false};
      const bool result = cudax::coop::any_of(cudax::broadcasted, group, in);
      REQUIRE(!result);
    }

    // Test all threads with true (broadcasted).
    {
      bool in[]{true, true, true, true};
      const bool result = cudax::coop::any_of(cudax::broadcasted, group, in);
      REQUIRE(result);
    }

    // Test all threads except 1 with false (broadcasted).
    {
      bool in[]{false, false, my_rank == cuda::gpu_thread.count_as<unsigned>(group) - 1};
      const bool result = cudax::coop::any_of(cudax::broadcasted, group, in);
      REQUIRE(result);
    }
  }
}

struct TestKernel
{
  template <class Config>
  __device__ void operator()(const Config& config)
  {
    test_group(cudax::this_warp{});
    test_group(cudax::this_warp{config});
  }
};

C2H_TEST("any_of/this_warp", "[any_of][this_warp]")
{
  const auto device = cuda::devices[0];

  const cuda::stream stream{device};

  const auto config = cuda::make_config(cuda::grid_dims<1>(), cuda::block_dims<32>());
  cuda::launch(stream, config, TestKernel{});

  stream.sync();
}
