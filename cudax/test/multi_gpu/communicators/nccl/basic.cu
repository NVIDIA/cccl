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
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/__multi_gpu/nccl_communicator_ref.h>

#include <nccl.h>

#include "nccl_test_helpers.cuh"

C2H_TEST("nccl_communicator_ref typedefs", "[multi_gpu]")
{
  STATIC_REQUIRE(::cuda::std::is_same_v<cudax::nccl_communicator_ref::native_handle_type, ncclComm_t>);
  STATIC_REQUIRE(
    ::cuda::std::is_same_v<cudax::nccl_communicator_ref::group_guard_type,
                           decltype(::cuda::std::declval<const cudax::nccl_communicator_ref&>().group_guard())>);
}

C2H_TEST("nccl_communicator_ref not constructible from NCCL_COMM_NULL", "[multi_gpu]")
{
  STATIC_REQUIRE(!::cuda::std::is_constructible_v<cudax::nccl_communicator_ref, decltype(NCCL_COMM_NULL)>);
  STATIC_REQUIRE(!::cuda::std::is_constructible_v<cudax::nccl_communicator_ref, cuda::std::nullptr_t>);
}

NCCL_COMM_TEST("nccl_communicator_ref basic")
{
  SECTION("rank and size")
  {
    int i = 0;

    for (auto& comm : this->communicators())
    {
      REQUIRE(comm.rank() == i);
      REQUIRE(comm.size() == static_cast<int>(cuda::devices.size()));
      ++i;
    }
  }

  SECTION("native handle")
  {
    int i = 0;

    for (auto& comm : this->communicators())
    {
      REQUIRE(comm.native_handle() == this->handles()[i]);
      ++i;
    }
  }

  SECTION("logical device")
  {
    int i = 0;

    for (auto& comm : this->communicators())
    {
      REQUIRE(comm.logical_device().underlying_device() == cuda::devices[i]);
      ++i;
    }
  }

  SECTION("group_guard round trip")
  {
    // Opening and closing a guard with no enqueued ops must not throw.
    [[maybe_unused]] auto g = this->communicators().front().group_guard();
  }

  SECTION("device mismatch throws")
  {
    if (cuda::devices.size() > 1)
    {
      REQUIRE_THROWS_WITH(
        cudax::nccl_communicator_ref(this->handles().front(), cudax::logical_device{cuda::devices[1]}),
        "Inconsistent devices, NCCL communicator device and provided logical device do not match");
    }
  }
}
