//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

#include <testing.cuh>

// Green contexts require CTK 12.5, so `cuda::__logical_device` is only declared from that version on.
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

C2H_CCCLRT_TEST("logical_device_ref traits", "[device][logical_device]")
{
  using cuda::__logical_device;
  using cuda::__logical_device_ref;

  SECTION("A ref is a copyable value type")
  {
    STATIC_REQUIRE(cuda::std::is_copy_constructible_v<__logical_device_ref>);
    STATIC_REQUIRE(cuda::std::is_copy_assignable_v<__logical_device_ref>);
    STATIC_REQUIRE(cuda::std::is_trivially_destructible_v<__logical_device_ref>);
  }

  SECTION("An owner is move-only and destroys its green context")
  {
    STATIC_REQUIRE(!cuda::std::is_copy_constructible_v<__logical_device>);
    STATIC_REQUIRE(!cuda::std::is_copy_assignable_v<__logical_device>);
    STATIC_REQUIRE(cuda::std::is_move_constructible_v<__logical_device>);
    STATIC_REQUIRE(cuda::std::is_move_assignable_v<__logical_device>);
    STATIC_REQUIRE(!cuda::std::is_trivially_destructible_v<__logical_device>);
  }

  SECTION("An owner converts to a ref, but a ref does not slice into an owner")
  {
    // The commas of the trait arguments would split the macro argument list, so each trait is
    // evaluated into a named constant first.
    constexpr bool owner_derives_from_ref = cuda::std::is_base_of_v<__logical_device_ref, __logical_device>;
    constexpr bool owner_converts_to_ref =
      cuda::std::is_convertible_v<const __logical_device&, const __logical_device_ref&>;
    constexpr bool ref_builds_owner = cuda::std::is_constructible_v<__logical_device, __logical_device_ref>;

    STATIC_REQUIRE(owner_derives_from_ref);
    STATIC_REQUIRE(owner_converts_to_ref);
    STATIC_REQUIRE(!ref_builds_owner);
  }

  SECTION("Accessors return the expected types")
  {
    using device_result_type = decltype(cuda::std::declval<__logical_device_ref>().underlying_device());
    using gctx_result_type   = decltype(cuda::std::declval<__logical_device_ref>().green_context());

    constexpr bool device_matches = cuda::std::is_same_v<cuda::device_ref, device_result_type>;
    constexpr bool gctx_matches   = cuda::std::is_same_v<::CUgreenCtx, gctx_result_type>;

    STATIC_REQUIRE(device_matches);
    STATIC_REQUIRE(gctx_matches);
  }

  SECTION("A ref is usable in a constant expression")
  {
    constexpr __logical_device_ref ref{cuda::device_ref{0}, nullptr};
    constexpr __logical_device_ref same{cuda::device_ref{0}, nullptr};

    STATIC_REQUIRE(ref.underlying_device() == 0);
    STATIC_REQUIRE(ref.green_context() == nullptr);
    STATIC_REQUIRE(ref == same);
  }
}

C2H_CCCLRT_TEST("logical_device_ref comparison", "[device][logical_device]")
{
  using cuda::__logical_device_ref;

  // Two distinct non-null handles. They are never dereferenced, so fabricating them is safe and
  // keeps this test free of any driver call.
  auto* const gctx0 = reinterpret_cast<::CUgreenCtx>(0x10);
  auto* const gctx1 = reinterpret_cast<::CUgreenCtx>(0x20);

  const __logical_device_ref dev0_null{cuda::device_ref{0}, nullptr};
  const __logical_device_ref dev0_gctx0{cuda::device_ref{0}, gctx0};
  const __logical_device_ref dev0_gctx1{cuda::device_ref{0}, gctx1};

  SECTION("Equal when both the device and the green context match")
  {
    const __logical_device_ref same_null{cuda::device_ref{0}, nullptr};
    const __logical_device_ref same_gctx0{cuda::device_ref{0}, gctx0};
    CCCLRT_REQUIRE(dev0_null == same_null);
    CCCLRT_REQUIRE(dev0_gctx0 == same_gctx0);
  }

  SECTION("Different green contexts on the same device are not equal")
  {
    CCCLRT_REQUIRE(dev0_gctx0 != dev0_gctx1);
    CCCLRT_REQUIRE(dev0_gctx0 != dev0_null);
  }

  SECTION("The same green context on different devices is not equal")
  {
    if (cuda::devices.size() > 1)
    {
      const __logical_device_ref dev1_gctx0{cuda::device_ref{1}, gctx0};
      const __logical_device_ref dev1_null{cuda::device_ref{1}, nullptr};
      CCCLRT_REQUIRE(dev0_gctx0 != dev1_gctx0);
      CCCLRT_REQUIRE(dev0_null != dev1_null);
    }
  }
}

C2H_CCCLRT_TEST("logical_device owns a green context", "[device][logical_device]")
{
  if (test::cuda_driver_version() < 12050)
  {
    SUCCEED("Driver is too old for green context tests");
    return;
  }

  using cuda::__logical_device;
  using cuda::__logical_device_ref;

  SECTION("from_native_handle stores the device and the handle")
  {
    const auto device = cuda::devices[0];
    auto ldev         = ::make_logical_device(device);
    CCCLRT_REQUIRE(ldev.underlying_device() == device);
    CCCLRT_REQUIRE(ldev.green_context() != nullptr);
  }

  SECTION("Two green contexts on the same device compare unequal")
  {
    auto ldev0 = ::make_logical_device(cuda::devices[0]);
    auto ldev1 = ::make_logical_device(cuda::devices[0]);
    CCCLRT_REQUIRE(ldev0.green_context() != ldev1.green_context());
    CCCLRT_REQUIRE(static_cast<const __logical_device_ref&>(ldev0) != static_cast<const __logical_device_ref&>(ldev1));
  }

  SECTION("Move construction transfers the handle and nulls the source")
  {
    auto source       = ::make_logical_device(cuda::devices[0]);
    const auto gctx   = source.green_context();
    const auto device = source.underlying_device();
    auto destination  = cuda::std::move(source);

    CCCLRT_REQUIRE(destination.green_context() == gctx);
    CCCLRT_REQUIRE(destination.underlying_device() == device);
    CCCLRT_REQUIRE(source.green_context() == nullptr);
  }

  SECTION("Move assignment transfers the handle and nulls the source")
  {
    auto source      = ::make_logical_device(cuda::devices[0]);
    auto destination = ::make_logical_device(cuda::devices[0]);
    const auto gctx  = source.green_context();

    // `destination` owns a different green context, which this assignment must destroy.
    destination = cuda::std::move(source);

    CCCLRT_REQUIRE(destination.green_context() == gctx);
    CCCLRT_REQUIRE(source.green_context() == nullptr);
  }

  SECTION("Self move assignment leaves the green context intact")
  {
    auto ldev       = ::make_logical_device(cuda::devices[0]);
    const auto gctx = ldev.green_context();

    auto& alias = ldev;
    ldev        = cuda::std::move(alias);

    CCCLRT_REQUIRE(ldev.green_context() == gctx);
  }

  SECTION("A moved-from logical_device destroys nothing")
  {
    // The moved-from object runs `__reset()` on a null handle at scope exit. The fixture's
    // driver-stack check catches a stray context push, and compute-sanitizer catches a double free.
    auto source = ::make_logical_device(cuda::devices[0]);
    {
      [[maybe_unused]] auto destination = cuda::std::move(source);
    }
    CCCLRT_REQUIRE(source.green_context() == nullptr);
  }

  SECTION("The green context converts to a usable CUcontext")
  {
    auto ldev      = ::make_logical_device(cuda::devices[0]);
    const auto ctx = cuda::__driver::__ctxFromGreenCtx(ldev.green_context());
    CCCLRT_REQUIRE(ctx != nullptr);
  }

  SECTION("A logical_device on a second device reports that device")
  {
    if (cuda::devices.size() > 1)
    {
      auto ldev = ::make_logical_device(cuda::devices[1]);
      CCCLRT_REQUIRE(ldev.underlying_device() == cuda::devices[1]);
      CCCLRT_REQUIRE(ldev.green_context() != nullptr);
    }
  }
}

#else // ^^^ _CCCL_CTK_AT_LEAST(12, 5) ^^^ / vvv _CCCL_CTK_BELOW(12, 5) vvv

// The test binary must contain at least one test case, otherwise Catch2 reports a failure.
C2H_CCCLRT_TEST("logical_device requires CTK 12.5", "[device][logical_device]")
{
  SUCCEED("Green contexts are unavailable before CTK 12.5");
}

#endif // ^^^ _CCCL_CTK_BELOW(12, 5) ^^^
