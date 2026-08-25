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
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <testing.cuh>

// Green contexts require CTK 12.5, so `cuda::__logical_device` is only declared from that version on.
#if _CCCL_CTK_AT_LEAST(12, 5)

namespace
{
//! True when `__logical_device::from_native_handle(device_ref, Handle)` names a viable overload.
//!
//! `from_native_handle` is an overload set with deleted members, so `decltype` cannot name it and
//! `is_invocable_v` is not usable. This detects the call expression instead.
template <class Handle, class = void>
inline constexpr bool can_make_from_native_handle_v = false;

template <class Handle>
inline constexpr bool
  can_make_from_native_handle_v<Handle,
                                cuda::std::void_t<decltype(cuda::__logical_device::from_native_handle(
                                  cuda::std::declval<cuda::device_ref>(), cuda::std::declval<Handle>()))>> = true;

//! Create a green context that spans the whole of `device`. Returns an owning `__logical_device`.
cuda::__logical_device make_logical_device(cuda::device_ref device)
{
  const auto gctx = cuda::__driver::__greenCtxCreate(cuda::__driver::__deviceGet(device.get()));
  return cuda::__logical_device::from_native_handle(device, gctx);
}
} // namespace

C2H_CCCLRT_TEST("logical_device_ref traits", "[device][logical_device]", )
{
  SECTION("A ref is a copyable value type")
  {
    STATIC_REQUIRE(cuda::std::is_copy_constructible_v<cuda::__logical_device_ref>);
    STATIC_REQUIRE(cuda::std::is_copy_assignable_v<cuda::__logical_device_ref>);
    STATIC_REQUIRE(cuda::std::is_move_constructible_v<cuda::__logical_device_ref>);
    STATIC_REQUIRE(cuda::std::is_move_assignable_v<cuda::__logical_device_ref>);
    STATIC_REQUIRE(cuda::std::is_trivially_destructible_v<cuda::__logical_device_ref>);
  }

  SECTION("A ref copies without any user-provided work")
  {
    // The ref only holds an ordinal and a handle. A non-trivial copy would mean it took ownership
    // of something.
    STATIC_REQUIRE(cuda::std::is_trivially_copy_constructible_v<cuda::__logical_device_ref>);
    STATIC_REQUIRE(cuda::std::is_trivially_copy_assignable_v<cuda::__logical_device_ref>);
    STATIC_REQUIRE(cuda::std::is_trivially_copyable_v<cuda::__logical_device_ref>);
  }

  SECTION("A ref has no default constructor")
  {
    // A default-constructed ref would name device 0 with a null handle, which is not a logical
    // device. Both members must always come from the caller.
    STATIC_REQUIRE(!cuda::std::is_default_constructible_v<cuda::__logical_device_ref>);
  }

  SECTION("A ref is built from exactly a device and a handle")
  {
    constexpr bool from_device_and_gctx =
      cuda::std::is_constructible_v<cuda::__logical_device_ref, cuda::device_ref, ::CUgreenCtx>;
    constexpr bool from_ordinal_and_gctx = cuda::std::is_constructible_v<cuda::__logical_device_ref, int, ::CUgreenCtx>;
    constexpr bool from_device_alone     = cuda::std::is_constructible_v<cuda::__logical_device_ref, cuda::device_ref>;
    constexpr bool from_gctx_alone       = cuda::std::is_constructible_v<cuda::__logical_device_ref, ::CUgreenCtx>;

    STATIC_REQUIRE(from_device_and_gctx);
    // `device_ref` converts implicitly from `int`, so an ordinal is accepted too.
    STATIC_REQUIRE(from_ordinal_and_gctx);
    STATIC_REQUIRE(!from_device_alone);
    STATIC_REQUIRE(!from_gctx_alone);
  }

  SECTION("An owner is move-only and destroys its green context")
  {
    STATIC_REQUIRE(!cuda::std::is_copy_constructible_v<cuda::__logical_device>);
    STATIC_REQUIRE(!cuda::std::is_copy_assignable_v<cuda::__logical_device>);
    STATIC_REQUIRE(cuda::std::is_move_constructible_v<cuda::__logical_device>);
    STATIC_REQUIRE(cuda::std::is_move_assignable_v<cuda::__logical_device>);
    STATIC_REQUIRE(!cuda::std::is_trivially_destructible_v<cuda::__logical_device>);
  }

  SECTION("An owner moves with user-provided work")
  {
    // A trivial move would copy the handle and leave the source owning it too, so the green context
    // would be destroyed twice.
    STATIC_REQUIRE(!cuda::std::is_trivially_move_constructible_v<cuda::__logical_device>);
    STATIC_REQUIRE(!cuda::std::is_trivially_move_assignable_v<cuda::__logical_device>);
    STATIC_REQUIRE(!cuda::std::is_trivially_copyable_v<cuda::__logical_device>);
  }

  SECTION("An owner has no default constructor and no public constructor")
  {
    // Every owner must come from `from_native_handle`, which is the only place that pairs a handle
    // with the device it was created on.
    constexpr bool from_device_and_gctx =
      cuda::std::is_constructible_v<cuda::__logical_device, cuda::device_ref, ::CUgreenCtx>;

    STATIC_REQUIRE(!cuda::std::is_default_constructible_v<cuda::__logical_device>);
    STATIC_REQUIRE(!from_device_and_gctx);
  }

  SECTION("from_native_handle takes a device and a real handle only")
  {
    // The `int` and `nullptr_t` overloads are deleted, so a caller cannot pass an ordinal where a
    // handle belongs, nor build an owner of nothing.
    STATIC_REQUIRE(::can_make_from_native_handle_v<::CUgreenCtx>);
    STATIC_REQUIRE(!::can_make_from_native_handle_v<int>);
    STATIC_REQUIRE(!::can_make_from_native_handle_v<cuda::std::nullptr_t>);
  }

  SECTION("from_native_handle returns an owner by value")
  {
    using result_type = decltype(cuda::__logical_device::from_native_handle(
      cuda::std::declval<cuda::device_ref>(), cuda::std::declval<::CUgreenCtx>()));

    constexpr bool returns_owner_by_value = cuda::std::is_same_v<result_type, cuda::__logical_device>;
    STATIC_REQUIRE(returns_owner_by_value);
  }

  SECTION("An owner converts to a ref, but a ref does not slice into an owner")
  {
    // The commas of the trait arguments would split the macro argument list, so each trait is
    // evaluated into a named constant first.
    constexpr bool owner_derives_from_ref = cuda::std::is_base_of_v<cuda::__logical_device_ref, cuda::__logical_device>;
    constexpr bool owner_converts_to_ref =
      cuda::std::is_convertible_v<const cuda::__logical_device&, const cuda::__logical_device_ref&>;
    constexpr bool ref_builds_owner = cuda::std::is_constructible_v<cuda::__logical_device, cuda::__logical_device_ref>;

    STATIC_REQUIRE(owner_derives_from_ref);
    STATIC_REQUIRE(owner_converts_to_ref);
    STATIC_REQUIRE(!ref_builds_owner);
  }

  SECTION("An owner never takes a ref by assignment")
  {
    // Assigning a ref into an owner would leak the owned green context and adopt a handle the owner
    // does not own.
    constexpr bool owner_takes_ref = cuda::std::is_assignable_v<cuda::__logical_device&, cuda::__logical_device_ref>;
    constexpr bool owner_takes_ref_rvalue =
      cuda::std::is_assignable_v<cuda::__logical_device&, cuda::__logical_device_ref&&>;

    STATIC_REQUIRE(!owner_takes_ref);
    STATIC_REQUIRE(!owner_takes_ref_rvalue);
  }

  SECTION("A ref does take an owner by copy, which does not transfer ownership")
  {
    // Slicing an owner into a ref is the intended way to observe it, so it must compile. The
    // resulting ref is non-owning, which the base class already guarantees.
    constexpr bool ref_builds_from_owner =
      cuda::std::is_constructible_v<cuda::__logical_device_ref, const cuda::__logical_device&>;
    constexpr bool ref_takes_owner =
      cuda::std::is_assignable_v<cuda::__logical_device_ref&, const cuda::__logical_device&>;

    STATIC_REQUIRE(ref_builds_from_owner);
    STATIC_REQUIRE(ref_takes_owner);
  }

  SECTION("Accessors return the expected types")
  {
    using device_result_type = decltype(cuda::std::declval<cuda::__logical_device_ref>().underlying_device());
    using gctx_result_type   = decltype(cuda::std::declval<cuda::__logical_device_ref>().green_context());

    constexpr bool device_matches = cuda::std::is_same_v<cuda::device_ref, device_result_type>;
    constexpr bool gctx_matches   = cuda::std::is_same_v<::CUgreenCtx, gctx_result_type>;

    STATIC_REQUIRE(device_matches);
    STATIC_REQUIRE(gctx_matches);
  }

  SECTION("A ref is usable in a constant expression")
  {
    constexpr cuda::__logical_device_ref ref{cuda::device_ref{0}, nullptr};
    constexpr cuda::__logical_device_ref same{cuda::device_ref{0}, nullptr};

    STATIC_REQUIRE(ref.underlying_device() == 0);
    STATIC_REQUIRE(ref.green_context() == nullptr);
    STATIC_REQUIRE(ref == same);
  }
}

C2H_CCCLRT_TEST("logical_device_ref locality domain", "[device][logical_device]", )
{
  SECTION("A ref with no green context is not localized")
  {
    // A null handle must not reach the driver, so this holds even on a machine with no GPU.
    const cuda::__logical_device_ref ref{cuda::device_ref{0}, nullptr};
    const auto result = ref.locality_domain();

    REQUIRE_FALSE(result.localized);
    // A non-localized ref covers the whole device, which is reported as domain 0.
    REQUIRE(result.domain_id == 0);
  }

  if (test::cuda_driver_version() < 12050)
  {
    SUCCEED("Driver is too old for green context tests");
    return;
  }

  SECTION("A whole-device green context is not localized")
  {
    // `__greenCtxCreate` with a null descriptor covers the full device, so the driver reports no
    // locality domain id for it.
    auto ldev         = ::make_logical_device(cuda::devices[0]);
    const auto result = ldev.locality_domain();

    REQUIRE_FALSE(result.localized);
    REQUIRE(result.domain_id == 0);
  }

#  if _CCCL_CTK_AT_LEAST(13, 4)
  SECTION("Every cached domain reports its own index")
  {
    // `__locality_domains()` splits the device by ascending locality domain id, so the position in
    // the span is the id the driver must report back.
    for (auto dev : cuda::devices)
    {
      auto domains = dev.__locality_domains();

      for (cuda::std::size_t i = 0; i < domains.size(); ++i)
      {
        const auto result = domains[i].locality_domain();

        // Every cached domain comes from a split by locality domain id, so it is always localized.
        REQUIRE(result.localized);
        REQUIRE(result.domain_id == i);
      }
    }
  }

  SECTION("Repeated queries return the same id")
  {
    // The id is re-read from the driver on every call, so the accessor must hold no state.
    for (auto dev : cuda::devices)
    {
      for (auto& domain : dev.__locality_domains())
      {
        const auto first  = domain.locality_domain();
        const auto second = domain.locality_domain();

        REQUIRE(first.domain_id == second.domain_id);
        REQUIRE(first.localized == second.localized);
      }
    }
  }

  SECTION("Reading the id leaves the driver stack empty")
  {
    test::empty_driver_stack();
    for (auto dev : cuda::devices)
    {
      for (auto& domain : dev.__locality_domains())
      {
        static_cast<void>(domain.locality_domain());
      }
    }
    REQUIRE(test::count_driver_stack() == 0);
  }
#  endif // _CCCL_CTK_AT_LEAST(13, 4)
}

C2H_CCCLRT_TEST("logical_device_ref comparison", "[device][logical_device]")
{
  // Two distinct non-null handles. They are never dereferenced, so fabricating them is safe and
  // keeps this test free of any driver call.
  auto* const gctx0 = reinterpret_cast<::CUgreenCtx>(0x10);
  auto* const gctx1 = reinterpret_cast<::CUgreenCtx>(0x20);

  const cuda::__logical_device_ref dev0_null{cuda::device_ref{0}, nullptr};
  const cuda::__logical_device_ref dev0_gctx0{cuda::device_ref{0}, gctx0};
  const cuda::__logical_device_ref dev0_gctx1{cuda::device_ref{0}, gctx1};

  SECTION("Equal when both the device and the green context match")
  {
    const cuda::__logical_device_ref same_null{cuda::device_ref{0}, nullptr};
    const cuda::__logical_device_ref same_gctx0{cuda::device_ref{0}, gctx0};
    REQUIRE(dev0_null == same_null);
    REQUIRE(dev0_gctx0 == same_gctx0);
  }

  SECTION("Different green contexts on the same device are not equal")
  {
    REQUIRE(dev0_gctx0 != dev0_gctx1);
    REQUIRE(dev0_gctx0 != dev0_null);
  }

  SECTION("The same green context on different devices is not equal")
  {
    if (cuda::devices.size() > 1)
    {
      const cuda::__logical_device_ref dev1_gctx0{cuda::device_ref{1}, gctx0};
      const cuda::__logical_device_ref dev1_null{cuda::device_ref{1}, nullptr};
      REQUIRE(dev0_gctx0 != dev1_gctx0);
      REQUIRE(dev0_null != dev1_null);
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

  SECTION("from_native_handle stores the device and the handle")
  {
    const auto device = cuda::devices[0];
    auto ldev         = ::make_logical_device(device);
    REQUIRE(ldev.underlying_device() == device);
    REQUIRE(ldev.green_context() != nullptr);
  }

  SECTION("Two green contexts on the same device compare unequal")
  {
    auto ldev0 = ::make_logical_device(cuda::devices[0]);
    auto ldev1 = ::make_logical_device(cuda::devices[0]);
    REQUIRE(ldev0.green_context() != ldev1.green_context());
    REQUIRE(
      static_cast<const cuda::__logical_device_ref&>(ldev0) != static_cast<const cuda::__logical_device_ref&>(ldev1));
  }

  SECTION("Move construction transfers the handle and nulls the source")
  {
    auto source       = ::make_logical_device(cuda::devices[0]);
    const auto gctx   = source.green_context();
    const auto device = source.underlying_device();
    auto destination  = cuda::std::move(source);

    REQUIRE(destination.green_context() == gctx);
    REQUIRE(destination.underlying_device() == device);
    REQUIRE(source.green_context() == nullptr);
  }

  SECTION("Move assignment transfers the handle and nulls the source")
  {
    auto source      = ::make_logical_device(cuda::devices[0]);
    auto destination = ::make_logical_device(cuda::devices[0]);
    const auto gctx  = source.green_context();

    // `destination` owns a different green context, which this assignment must destroy.
    destination = cuda::std::move(source);

    REQUIRE(destination.green_context() == gctx);
    REQUIRE(source.green_context() == nullptr);
  }

  SECTION("Self move assignment leaves the green context intact")
  {
    auto ldev       = ::make_logical_device(cuda::devices[0]);
    const auto gctx = ldev.green_context();

    auto& alias = ldev;
    ldev        = cuda::std::move(alias);

    REQUIRE(ldev.green_context() == gctx);
  }

  SECTION("A moved-from logical_device destroys nothing")
  {
    // The moved-from object runs `__reset()` on a null handle at scope exit. The fixture's
    // driver-stack check catches a stray context push, and compute-sanitizer catches a double free.
    auto source = ::make_logical_device(cuda::devices[0]);
    {
      [[maybe_unused]] auto destination = cuda::std::move(source);
    }
    REQUIRE(source.green_context() == nullptr);
  }

  SECTION("The green context converts to a usable CUcontext")
  {
    auto ldev      = ::make_logical_device(cuda::devices[0]);
    const auto ctx = cuda::__driver::__ctxFromGreenCtx(ldev.green_context());
    REQUIRE(ctx != nullptr);
  }

  SECTION("A logical_device on a second device reports that device")
  {
    if (cuda::devices.size() > 1)
    {
      auto ldev = ::make_logical_device(cuda::devices[1]);
      REQUIRE(ldev.underlying_device() == cuda::devices[1]);
      REQUIRE(ldev.green_context() != nullptr);
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
