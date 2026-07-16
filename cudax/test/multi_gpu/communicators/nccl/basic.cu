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

#include <cuda/experimental/__multi_gpu/nccl_communicator.h>
#include <cuda/experimental/__multi_gpu/nccl_communicator_ref.h>

#include <nccl.h>
#include <nccl_test_common.h>

namespace
{
[[nodiscard]] ncclComm_t make_nccl_communicator_handle()
{
  if (cuda::devices.size() == 0)
  {
    SKIP("No CUDA devices visible");
  }

  const int device = cuda::devices[0].get();
  ncclComm_t handle{};

  const ncclResult_t result = ncclCommInitAll(&handle, 1, &device);

  INFO("NCCL: " << ncclGetErrorString(result));
  REQUIRE(result == ncclSuccess);

  return handle;
}
} // namespace

C2H_TEST("nccl_communicator_ref typedefs", "[multi_gpu]")
{
  STATIC_REQUIRE(::cuda::std::is_same_v<cudax::nccl_communicator_ref::native_handle_type, ncclComm_t>);
  STATIC_REQUIRE(
    ::cuda::std::is_same_v<cudax::nccl_communicator_ref::group_guard_type,
                           decltype(::cuda::std::declval<const cudax::nccl_communicator_ref&>().group_guard())>);
}

C2H_TEST("nccl_communicator(s) not constructible from NCCL_COMM_NULL", "[multi_gpu]")
{
  SECTION("ref")
  {
    STATIC_REQUIRE(!::cuda::std::is_constructible_v<cudax::nccl_communicator_ref, decltype(NCCL_COMM_NULL)>);
    STATIC_REQUIRE(!::cuda::std::is_constructible_v<cudax::nccl_communicator_ref, cuda::std::nullptr_t>);
  }

  SECTION("owning")
  {
    STATIC_REQUIRE(!::cuda::std::is_constructible_v<cudax::nccl_communicator, decltype(NCCL_COMM_NULL)>);
    STATIC_REQUIRE(!::cuda::std::is_constructible_v<cudax::nccl_communicator, cuda::std::nullptr_t>);
  }
}

C2H_TEST("nccl_communicator", "[multi_gpu][nccl]")
{
  SECTION("is move-only")
  {
    STATIC_REQUIRE(!cuda::std::is_copy_constructible_v<cudax::nccl_communicator>);
    STATIC_REQUIRE(!cuda::std::is_copy_assignable_v<cudax::nccl_communicator>);
    STATIC_REQUIRE(cuda::std::is_nothrow_move_constructible_v<cudax::nccl_communicator>);
    STATIC_REQUIRE(cuda::std::is_nothrow_move_assignable_v<cudax::nccl_communicator>);
  }

  SECTION("ownership")
  {
    //! [nccl_communicator_construction]
    const ncclComm_t handle = make_nccl_communicator_handle();

    auto comm = cuda::experimental::nccl_communicator{handle};

    // comm owns the handle now
    REQUIRE(comm.native_handle() == handle);
    //! [nccl_communicator_construction]
  }

  SECTION("release")
  {
    //! [nccl_communicator_release]
    const ncclComm_t handle = make_nccl_communicator_handle();

    auto comm = cuda::experimental::nccl_communicator{handle};

    const auto released_handle = comm.release();

    // comm contains the null handle after release
    REQUIRE(comm.native_handle() == NCCL_COMM_NULL);
    REQUIRE(released_handle == handle);
    //! [nccl_communicator_release]
  }

  SECTION("move construction")
  {
    //! [nccl_communicator_move_construction]
    const ncclComm_t handle = make_nccl_communicator_handle();

    auto source      = cudax::nccl_communicator{handle};
    auto destination = cudax::nccl_communicator{cuda::std::move(source)};

    // moved-from communicator is now invalid
    REQUIRE(source.native_handle() == NCCL_COMM_NULL);
    REQUIRE(destination.native_handle() == handle);
    //! [nccl_communicator_move_construction]
  }

  SECTION("move assignment")
  {
    //! [nccl_communicator_move_assignment]
    auto source      = cuda::experimental::nccl_communicator{make_nccl_communicator_handle()};
    auto destination = cuda::experimental::nccl_communicator{make_nccl_communicator_handle()};

    // Save the native handle to verify that ownership is transferred.
    const auto handle = source.native_handle();

    destination = cuda::std::move(source);

    REQUIRE(source.native_handle() == NCCL_COMM_NULL);
    REQUIRE(destination.native_handle() == handle);
    //! [nccl_communicator_move_assignment]
  }
}

MULTI_GPU_TEST("nccl_communicator_ref basic", )
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
    for (auto& comm : this->communicators())
    {
      REQUIRE(comm.native_handle() != NCCL_COMM_NULL);
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
        cudax::nccl_communicator_ref(this->communicators()[0].native_handle(), cudax::logical_device{cuda::devices[1]}),
        "Inconsistent devices, NCCL communicator device and provided logical device do not match");
    }
  }
}
