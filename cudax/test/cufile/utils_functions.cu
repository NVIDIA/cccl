//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory>

#include <cuda/experimental/__cufile/utils.hpp>

#include <stdexcept>

#include "test_utils.h"
#include <c2h/catch2_test_helper.h>

namespace cuda::experimental::cufile
{

using namespace cuda::experimental::cufile::utils;

TEST_CASE("CUDA memory functions", "[utils][cuda_memory]")
{
  if (!test_utils::is_cuda_available())
  {
    SKIP("CUDA not available");
  }

  SECTION("is_gpu_memory function")
  {
    // Test with GPU memory
    test_utils::GPUMemoryRAII gpu_mem(1024);
    REQUIRE(is_gpu_memory(gpu_mem.get()));

    // Test with host memory
    test_utils::HostMemoryRAII host_mem(1024);
    REQUIRE_FALSE(is_gpu_memory(host_mem.get()));

    // Test with regular memory
    test_utils::RegularMemoryRAII regular_mem(1024);
    REQUIRE_FALSE(is_gpu_memory(regular_mem.get()));

    // Test with null pointer
    REQUIRE_FALSE(is_gpu_memory(nullptr));
  }

  SECTION("get_device_id function")
  {
    // Test with GPU memory
    test_utils::GPUMemoryRAII gpu_mem(1024);
    int device_id = get_device_id(gpu_mem.get());
    REQUIRE(device_id >= 0);

    // Test with host memory - should not throw but return a device ID
    test_utils::HostMemoryRAII host_mem(1024);
    int host_device_id = get_device_id(host_mem.get());
    REQUIRE(host_device_id >= 0);

    // Test with regular memory - may return -1 for non-CUDA memory
    test_utils::RegularMemoryRAII regular_mem(1024);
    int regular_device_id = get_device_id(regular_mem.get());
    // Regular memory may return -1, which is valid for non-CUDA memory
    REQUIRE(regular_device_id >= -1);
  }

  SECTION("is_cufile_compatible function")
  {
    // Test with GPU memory (should be compatible)
    test_utils::GPUMemoryRAII gpu_mem(1024);
    REQUIRE(is_cufile_compatible(gpu_mem.get()));

    // Test with pinned host memory (should be compatible)
    test_utils::HostMemoryRAII host_mem(1024);
    REQUIRE(is_cufile_compatible(host_mem.get()));

    // Test with regular memory (should not be compatible)
    test_utils::RegularMemoryRAII regular_mem(1024);
    REQUIRE_FALSE(is_cufile_compatible(regular_mem.get()));

    // Test with null pointer (should not be compatible)
    REQUIRE_FALSE(is_cufile_compatible(nullptr));
  }
}

TEST_CASE("Corner cases and error conditions", "[utils][edge_cases]")
{
  SECTION("Null pointer handling")
  {
    // Test functions with null pointer
    REQUIRE_FALSE(is_gpu_memory(nullptr));
    REQUIRE_FALSE(is_cufile_compatible(nullptr));

    // Note: get_device_id(nullptr) behavior depends on CUDA implementation
    // Some CUDA versions may not throw for null pointers
    try
    {
      get_device_id(nullptr);
      // If it doesn't throw, that's also acceptable behavior
    }
    catch (const ::std::runtime_error&)
    {
      // Expected behavior - null pointer caused an error
    }
  }
}

TEST_CASE("Integration tests", "[utils][integration]")
{
  if (!test_utils::is_cuda_available())
  {
    SKIP("CUDA not available");
  }

  SECTION("Memory type integration")
  {
    // Test that GPU memory is identified correctly and is cuFile compatible
    test_utils::GPUMemoryRAII gpu_mem(4096); // Use aligned size
    void* gpu_ptr = gpu_mem.get();

    REQUIRE(is_gpu_memory(gpu_ptr));
    REQUIRE(is_cufile_compatible(gpu_ptr));

    int device_id = get_device_id(gpu_ptr);
    REQUIRE(device_id >= 0);

    // Test that host memory is not GPU memory but is cuFile compatible
    test_utils::HostMemoryRAII host_mem(4096);
    void* host_ptr = host_mem.get();

    REQUIRE_FALSE(is_gpu_memory(host_ptr));
    REQUIRE(is_cufile_compatible(host_ptr));

    // Test that regular memory is neither GPU memory nor cuFile compatible
    test_utils::RegularMemoryRAII regular_mem(4096);
    void* regular_ptr = regular_mem.get();

    REQUIRE_FALSE(is_gpu_memory(regular_ptr));
    REQUIRE_FALSE(is_cufile_compatible(regular_ptr));
  }
}

} // namespace cuda::experimental::cufile
