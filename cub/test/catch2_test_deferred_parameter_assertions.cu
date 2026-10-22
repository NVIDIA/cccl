// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Each variant runs in its own process: an expected device assertion invalidates the CUDA context.
// %PARAM% ASSERTION_CASE case 0:1:2:3:4:5:6:7

#include <cub/detail/deferred_parameter.cuh>

#include <cuda/argument>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <cuda_runtime_api.h>

#include "cub_test_macros.h"
#include <c2h/catch2_test_helper.h>

using count_t = cuda::std::conditional_t<
  ASSERTION_CASE == 1 || ASSERTION_CASE == 2 || ASSERTION_CASE == 7,
  cuda::std::int64_t,
  cuda::std::conditional_t<ASSERTION_CASE == 3,
                           cuda::std::uint32_t,
                           cuda::std::conditional_t<ASSERTION_CASE == 4, cuda::std::uint64_t, cuda::std::int32_t>>>;

template <typename KernelArgT>
_CCCL_KERNEL_ATTRIBUTES void read_invalid_num_items(KernelArgT num_items, cuda::std::uint64_t* result)
{
  *result = cub::detail::parameter_from_device<cuda::std::uint64_t>(num_items);
}

CUB_TEST("Invalid deferred problem sizes trigger a device assertion",
         "[deferred_parameter][skip-cs-memcheck][skip-cs-racecheck][skip-cs-initcheck][skip-cs-synccheck]",
         CUB_SMALL)
{
  constexpr count_t count = static_cast<count_t>(
    ASSERTION_CASE < 2                           ? -1
    : ASSERTION_CASE == 2 || ASSERTION_CASE == 4 ? 9
    : ASSERTION_CASE == 3 || ASSERTION_CASE == 5 ? 21
    : ASSERTION_CASE == 6                        ? 5
                                                 : 25);
  count_t* device_count              = nullptr;
  cuda::std::uint64_t* device_result = nullptr;
  REQUIRE(cudaSuccess == cudaMalloc(&device_count, sizeof(count_t)));
  REQUIRE(cudaSuccess == cudaMalloc(&device_result, sizeof(cuda::std::uint64_t)));
  REQUIRE(cudaSuccess == cudaMemcpy(device_count, &count, sizeof(count), cudaMemcpyHostToDevice));

  const auto arg = [=] {
    if constexpr (ASSERTION_CASE < 2)
    {
      return cuda::args::deferred{device_count};
    }
    else if constexpr (ASSERTION_CASE < 4)
    {
      return cuda::args::deferred{device_count, cuda::args::bounds<count_t{10}, count_t{20}>()};
    }
    else if constexpr (ASSERTION_CASE < 6)
    {
      return cuda::args::deferred{device_count, cuda::args::bounds(count_t{10}, count_t{20})};
    }
    else
    {
      return cuda::args::deferred{
        device_count, cuda::args::bounds<count_t{0}, count_t{20}>(), cuda::args::bounds(count_t{10}, count_t{30})};
    }
  }();

  read_invalid_num_items<<<1, 1>>>(cub::detail::make_num_items_kernel_arg(arg), device_result);
  REQUIRE(cudaSuccess == cudaGetLastError());
  REQUIRE(cudaErrorAssert == cudaDeviceSynchronize());
  // The invalid CUDA context cannot free these allocations; process teardown reclaims them.
}
