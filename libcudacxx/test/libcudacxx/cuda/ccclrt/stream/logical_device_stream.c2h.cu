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

C2H_CCCLRT_TEST("Stream logical device without a green context", "[stream][logical_device]")
{
  const auto device = cuda::devices[0];

  SECTION("A stream on a device reports a device-backed logical device")
  {
    cuda::stream str{device};
    const auto ldev = str.__logical_device();

    REQUIRE(ldev.kind() == cuda::__logical_device_ref::kinds::device);
    REQUIRE(ldev.green_context() == nullptr);
    REQUIRE(ldev.underlying_device() == device);
    REQUIRE(ldev.context() == device.__primary_context());
  }

  SECTION("The reported logical device equals one built from the device")
  {
    cuda::stream str{device};

    REQUIRE(str.__logical_device() == cuda::__logical_device_ref{device});
  }

  SECTION("Two streams on the same device report the same logical device")
  {
    cuda::stream str0{device};
    cuda::stream str1{device};

    REQUIRE(str0.__logical_device() == str1.__logical_device());
  }

  SECTION("A stream created through the runtime reports its device")
  {
    cuda::__ensure_current_context guard(device);
    ::cudaStream_t handle{};
    CUDART(cudaStreamCreate(&handle));

    const cuda::stream_ref str{handle};
    REQUIRE(str.__logical_device() == cuda::__logical_device_ref{device});

    CUDART(cudaStreamDestroy(handle));
  }

  SECTION("The query leaves the driver context stack unchanged")
  {
    // The fixture only checks that the stack is empty at the end of the test, so it does not catch
    // a query that pushes a context onto a non-empty stack.
    cuda::stream str{device};
    cuda::__ensure_current_context guard(device);

    const auto before = ::test::count_driver_stack();
    (void) str.__logical_device();
    REQUIRE(::test::count_driver_stack() == before);
  }

  SECTION("A stream on a second device reports that device")
  {
    if (cuda::devices.size() > 1)
    {
      const auto second = cuda::devices[1];
      cuda::stream str{second};

      REQUIRE(str.__logical_device() == cuda::__logical_device_ref{second});
      REQUIRE(str.__logical_device() != cuda::__logical_device_ref{device});
    }
  }
}

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

  SECTION("The stream reports the green context it was created on")
  {
    // A green context stream rejects cuStreamGetCtx(), so a query that reaches for the legacy
    // context of the stream throws instead of reporting the green context.
    auto ldev = ::make_logical_device(device);
    cuda::stream str{ldev};

    REQUIRE(str.__logical_device().kind() == cuda::__logical_device_ref::kinds::green_context);
    REQUIRE(str.__logical_device().green_context() == ldev.green_context());
    REQUIRE(str.__logical_device().underlying_device() == device);
  }

  SECTION("The reported logical device carries the context of the green context")
  {
    auto ldev = ::make_logical_device(device);
    cuda::stream str{ldev};

    REQUIRE(str.__logical_device().context() == cuda::__driver::__ctxFromGreenCtx(ldev.green_context()));
    REQUIRE(str.__logical_device().context() != device.__primary_context());
  }

  SECTION("The reported logical device compares equal to the one the stream was built from")
  {
    auto ldev = ::make_logical_device(device);
    cuda::stream str{ldev};

    REQUIRE(str.__logical_device() == static_cast<const cuda::__logical_device_ref&>(ldev));
  }

  SECTION("Streams from different green contexts report different logical devices")
  {
    auto ldev0 = ::make_logical_device(device);
    auto ldev1 = ::make_logical_device(device);

    cuda::stream str0{ldev0};
    cuda::stream str1{ldev1};

    REQUIRE(str0.__logical_device() != str1.__logical_device());
  }

  SECTION("A stream_ref reports the same logical device as the owning stream")
  {
    auto ldev = ::make_logical_device(device);
    cuda::stream str{ldev};
    const cuda::stream_ref ref{str.get()};

    REQUIRE(ref.__logical_device() == str.__logical_device());
  }

  SECTION("The stream belongs to the context of the green context")
  {
    auto ldev = ::make_logical_device(device);
    cuda::stream str{ldev};

    const auto green_ctx  = cuda::__driver::__ctxFromGreenCtx(ldev.green_context());
    const auto stream_ctx = cuda::__driver::__streamGetCtx(str.get());
    REQUIRE(stream_ctx == green_ctx);
  }

  SECTION("A device-backed logical device gives a stream on the primary context")
  {
    const cuda::__logical_device_ref ldev{device};
    cuda::stream str{ldev};

    REQUIRE(str.get() != nullptr);
    REQUIRE(str.device() == device);
    REQUIRE(cuda::__driver::__streamGetCtx(str.get()) == device.__primary_context());

    ::test::pinned<int> value(0);
    ::test::launch_kernel_single_thread(str, ::test::assign_42{}, value.get());
    str.sync();
    REQUIRE(*value == 42);
  }

  SECTION("A device-backed stream matches one built from the device_ref directly")
  {
    const cuda::__logical_device_ref ldev{device};
    cuda::stream from_logical{ldev};
    cuda::stream from_device{device};

    REQUIRE(cuda::__driver::__streamGetCtx(from_logical.get()) == cuda::__driver::__streamGetCtx(from_device.get()));
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
