//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// `__logical_device` is not reachable from `<cuda/devices>`, so the internal headers are included
// directly here.
#include <cuda/__device/logical_device.h>
#include <cuda/__device/logical_device_ref.h>
#include <cuda/__driver/driver_api.h>
#include <cuda/devices>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/stream>

#include <testing.cuh>

// Green contexts require CTK 12.5, so the `__logical_device_ref` stream constructor is only declared
// from that version on.
#if _CCCL_CTK_AT_LEAST(12, 5)

namespace
{
//! Create a green context that spans the whole of `device`. Returns an owning `__logical_device`.
cuda::__logical_device make_logical_device(cuda::device_ref device)
{
  const auto gctx = cuda::__driver::__greenCtxCreate(cuda::__driver::__deviceGet(device.get()));
  return cuda::__logical_device::from_native_handle(device, gctx);
}
} // namespace

C2H_CCCLRT_TEST("Stream construction from a logical device is explicit", "[stream][logical_device]")
{
  SECTION("The overload exists and takes an optional priority")
  {
    constexpr bool from_ref              = cuda::std::is_constructible_v<cuda::stream, cuda::__logical_device_ref>;
    constexpr bool from_ref_and_priority = cuda::std::is_constructible_v<cuda::stream, cuda::__logical_device_ref, int>;
    constexpr bool from_owner            = cuda::std::is_constructible_v<cuda::stream, const cuda::__logical_device&>;

    STATIC_REQUIRE(from_ref);
    STATIC_REQUIRE(from_ref_and_priority);
    // An owner slices into a ref, so it selects the same overload.
    STATIC_REQUIRE(from_owner);
  }

  SECTION("The overload never converts implicitly")
  {
    // An implicit conversion would let a logical device turn into a stream in an argument list,
    // which hides the stream creation from the caller.
    STATIC_REQUIRE(!cuda::std::is_convertible_v<cuda::__logical_device_ref, cuda::stream>);
  }
}

C2H_CCCLRT_TEST("Stream from a logical device", "[stream][logical_device]")
{
  if (test::cuda_driver_version() < 12050)
  {
    SUCCEED("Driver is too old for green context tests");
    return;
  }

  const auto device = cuda::devices[0];

  SECTION("The stream is valid and runs work")
  {
    auto ldev = ::make_logical_device(device);
    cuda::stream str{static_cast<const cuda::__logical_device_ref&>(ldev)};

    REQUIRE(str.get() != nullptr);

    ::test::pinned<int> value(0);
    ::test::launch_kernel_single_thread(str, ::test::assign_42{}, value.get());
    str.sync();
    REQUIRE(*value == 42);
  }

  SECTION("An owning logical_device selects the same overload")
  {
    auto ldev = ::make_logical_device(device);
    cuda::stream str{ldev};

    ::test::pinned<int> value(0);
    ::test::launch_kernel_single_thread(str, ::test::assign_42{}, value.get());
    str.sync();
    REQUIRE(*value == 42);
  }

  SECTION("The stream reports the device that owns the green context")
  {
    auto ldev = ::make_logical_device(device);
    cuda::stream str{ldev};
    REQUIRE(str.device() == device);
    REQUIRE(str.device() == ldev.underlying_device());
  }

  SECTION("The stream belongs to the context of the green context")
  {
    auto ldev = ::make_logical_device(device);
    cuda::stream str{ldev};

    const auto green_ctx  = cuda::__driver::__ctxFromGreenCtx(ldev.green_context());
    const auto stream_ctx = cuda::__driver::__streamGetCtx(str.get());
    REQUIRE(stream_ctx == green_ctx);
  }

  SECTION("The default priority is used when none is given")
  {
    auto ldev = ::make_logical_device(device);
    cuda::stream str{ldev};
    REQUIRE(str.priority() == cuda::stream::default_priority);
  }

  SECTION("An explicit priority reaches the stream")
  {
    auto ldev = ::make_logical_device(device);

    // The driver clamps a priority to the supported range, so the requested value is only checked
    // on a device that supports more than one priority.
    int least_priority{};
    int greatest_priority{};
    {
      cuda::__ensure_current_context guard(device);
      CUDART(cudaDeviceGetStreamPriorityRange(&least_priority, &greatest_priority));
    }

    if (least_priority != greatest_priority)
    {
      const auto priority = cuda::stream::default_priority - 1;
      cuda::stream str{ldev, priority};
      REQUIRE(str.priority() == priority);
    }
    else
    {
      SUCCEED("The device supports a single stream priority");
    }
  }

  SECTION("Streams from different green contexts are distinct")
  {
    auto ldev0 = ::make_logical_device(device);
    auto ldev1 = ::make_logical_device(device);

    cuda::stream str0{ldev0};
    cuda::stream str1{ldev1};

    REQUIRE(str0 != str1);
    REQUIRE(str0.id() != str1.id());
  }

  SECTION("Move construction keeps the handle usable")
  {
    // The stream keeps working after the handle moves, which proves the constructor produced an
    // owned handle and not a borrowed one.
    auto ldev = ::make_logical_device(device);
    cuda::stream source{ldev};
    const auto handle = source.get();

    cuda::stream destination{cuda::std::move(source)};
    REQUIRE(destination.get() == handle);

    ::test::pinned<int> value(0);
    ::test::launch_kernel_single_thread(destination, ::test::assign_42{}, value.get());
    destination.sync();
    REQUIRE(*value == 42);
  }

  SECTION("A stream on a second device reports that device")
  {
    if (cuda::devices.size() > 1)
    {
      const auto second = cuda::devices[1];
      auto ldev         = ::make_logical_device(second);
      cuda::stream str{ldev};

      REQUIRE(str.device() == second);

      ::test::pinned<int> value(0);
      ::test::launch_kernel_single_thread(str, ::test::assign_42{}, value.get());
      str.sync();
      REQUIRE(*value == 42);
    }
  }
}

C2H_CCCLRT_TEST("Stream from a logical device supports dependencies", "[stream][logical_device]")
{
  if (test::cuda_driver_version() < 12050)
  {
    SUCCEED("Driver is too old for green context tests");
    return;
  }

  auto ldev = ::make_logical_device(cuda::devices[0]);
  cuda::stream waiter{ldev};
  cuda::stream waitee{ldev};

  ::test::pinned<int> value(0);
  ::cuda::atomic_ref atomic_value(*value);

  ::test::launch_kernel_single_thread(waitee, ::test::spin_until_80{}, value.get());
  ::test::launch_kernel_single_thread(waitee, ::test::assign_42{}, value.get());

  waiter.wait(waitee);

  ::test::launch_kernel_single_thread(waiter, ::test::verify_42{}, value.get());
  REQUIRE(atomic_value.load() != 42);
  REQUIRE(!waiter.is_done());

  atomic_value.store(80);
  waiter.sync();
  waitee.sync();
}

#else // ^^^ _CCCL_CTK_AT_LEAST(12, 5) ^^^ / vvv _CCCL_CTK_BELOW(12, 5) vvv

// The test binary must contain at least one test case, otherwise Catch2 reports a failure.
C2H_CCCLRT_TEST("Stream from a logical device requires CTK 12.5", "[stream][logical_device]")
{
  SUCCEED("Green contexts are unavailable before CTK 12.5");
}

#endif // ^^^ _CCCL_CTK_BELOW(12, 5) ^^^
