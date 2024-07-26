//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#define LIBCUDACXX_ENABLE_EXCEPTIONS
#include <cuda/experimental/device.cuh>

#include "../hierarchy/testing_common.cuh"
#include "cuda/std/__type_traits/is_same.h"

void device_smoke_test()
{
  using cudax::device;

  SECTION("Device count")
  {
    int count = device::count();
    CUDAX_REQUIRE(count > 0);
  }

  SECTION("Attributes")
  {
    device dev0(0);

    SECTION("max_threads_per_block")
    {
      STATIC_REQUIRE(::cudaDevAttrMaxThreadsPerBlock == device::attrs::max_threads_per_block);
      STATIC_REQUIRE(::cuda::std::is_same_v<int, device::attrs::max_threads_per_block_t::type>);
      STATIC_REQUIRE(::cuda::std::is_same_v<int, device::attr_result_t<device::attrs::max_threads_per_block>>);

      auto max = dev0.attr(device::attrs::max_threads_per_block);
      STATIC_REQUIRE(::cuda::std::is_same_v<decltype(max), int>);
      CUDAX_REQUIRE(max > 0);
      CUDAX_REQUIRE(max == dev0.attr<cudaDevAttrMaxThreadsPerBlock>());
    }

    SECTION("compute_mode")
    {
      STATIC_REQUIRE(::cudaDevAttrComputeMode == device::attrs::compute_mode);
      STATIC_REQUIRE(::cuda::std::is_same_v<::cudaComputeMode, device::attrs::compute_mode_t::type>);
      STATIC_REQUIRE(::cuda::std::is_same_v<::cudaComputeMode, device::attr_result_t<device::attrs::compute_mode>>);
      STATIC_REQUIRE(::cudaComputeModeDefault == device::attrs::compute_mode._default);
      STATIC_REQUIRE(::cudaComputeModeProhibited == device::attrs::compute_mode.prohibited);
      STATIC_REQUIRE(::cudaComputeModeExclusiveProcess == device::attrs::compute_mode.exclusive_process);

      auto mode = dev0.attr(device::attrs::compute_mode);
      STATIC_REQUIRE(::cuda::std::is_same_v<decltype(mode), ::cudaComputeMode>);
      CUDAX_REQUIRE((mode == device::attrs::compute_mode._default || //
                     mode == device::attrs::compute_mode.prohibited || //
                     mode == device::attrs::compute_mode.exclusive_process));
      CUDAX_REQUIRE(mode == dev0.attr<::cudaDevAttrComputeMode>());
    }
  }
}

TEST_CASE("Smoke", "[device]")
{
  device_smoke_test();
}
