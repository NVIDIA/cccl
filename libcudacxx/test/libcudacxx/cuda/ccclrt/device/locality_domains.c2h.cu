//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// `__locality_domains()` lives in `<cuda/__device/physical_device.h>` and is reachable via
// `<cuda/devices>`. `__logical_device_ref` is not, so its header is included directly.
#include <cuda/__device/logical_device_ref.h>
#include <cuda/__driver/driver_api.h>
#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/devices>
#include <cuda/std/memory>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <testing.cuh>

// Green contexts require CTK 12.5, so locality domains are only available from that version on.
#if _CCCL_CTK_AT_LEAST(12, 5)

namespace
{
//! The type that `device_ref::__locality_domains()` returns.
using domains_span = decltype(cuda::std::declval<const cuda::device_ref&>().__locality_domains());

//! Number of SMs the driver reports for `device`.
int sm_count(cuda::device_ref device)
{
  return cuda::__driver::__deviceGetAttribute(
    ::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, cuda::__driver::__deviceGet(device.get()));
}
} // namespace

C2H_CCCLRT_TEST("locality domains static interface", "[device][locality_domain]")
{
  // These hold no matter which driver is loaded, so they run before any version check.
  SECTION("The accessor returns a span of immutable refs")
  {
    constexpr bool returns_span_of_const_refs =
      cuda::std::is_same_v<domains_span, cuda::std::span<const cuda::__logical_device_ref>>;
    STATIC_REQUIRE(returns_span_of_const_refs);
  }

  SECTION("The accessor is callable on a const device_ref")
  {
    // `devices[i]` yields a `const device_ref&`, so a non-const overload would break every use below.
    constexpr bool callable_on_const =
      cuda::std::is_invocable_v<decltype(&cuda::device_ref::__locality_domains), const cuda::device_ref&>;
    STATIC_REQUIRE(callable_on_const);
  }

  SECTION("The domains are never copies the caller can mutate")
  {
    // Handing out `__logical_device` by value would let a caller destroy a cached green context
    // through an owning type. The element type must stay the non-owning ref, spelled const.
    constexpr bool element_is_const_ref =
      cuda::std::is_same_v<domains_span::element_type, const cuda::__logical_device_ref>;
    constexpr bool value_is_plain_ref = cuda::std::is_same_v<domains_span::value_type, cuda::__logical_device_ref>;
    STATIC_REQUIRE(element_is_const_ref);
    STATIC_REQUIRE(value_is_plain_ref);
  }
}

C2H_CCCLRT_TEST("locality domains", "[device][locality_domain]")
{
  if (test::cuda_driver_version() < 12050)
  {
    SUCCEED("Driver is too old for green context tests");
    return;
  }

  SECTION("Returns at least one domain per device")
  {
    for (auto dev : cuda::devices)
    {
      auto domains = dev.__locality_domains();
      REQUIRE(domains.size() >= 1);
      REQUIRE(!domains.empty());
      REQUIRE(domains.data() != nullptr);
    }
  }

  SECTION("Every domain refers back to the queried device")
  {
    for (auto dev : cuda::devices)
    {
      for (auto& domain : dev.__locality_domains())
      {
        REQUIRE(domain.underlying_device() == dev);
        REQUIRE(domain.underlying_device().get() == dev.get());
      }
    }
  }

  SECTION("Every domain has a non-null green context")
  {
    for (auto dev : cuda::devices)
    {
      for (auto& domain : dev.__locality_domains())
      {
        REQUIRE(domain.green_context() != nullptr);
      }
    }
  }

  SECTION("Domains on the same device have distinct green contexts")
  {
    for (auto dev : cuda::devices)
    {
      auto domains = dev.__locality_domains();
      for (::cuda::std::size_t i = 0; i < domains.size(); ++i)
      {
        for (::cuda::std::size_t j = i + 1; j < domains.size(); ++j)
        {
          REQUIRE(domains[i].green_context() != domains[j].green_context());
          REQUIRE(domains[i] != domains[j]);
        }
      }
    }
  }

  SECTION("Domains of different devices are all distinct")
  {
    // Green context handles come from a process-wide driver allocator, so no handle may repeat
    // across devices either.
    for (auto lhs_dev : cuda::devices)
    {
      for (auto rhs_dev : cuda::devices)
      {
        if (lhs_dev == rhs_dev)
        {
          continue;
        }

        for (auto& lhs : lhs_dev.__locality_domains())
        {
          for (auto& rhs : rhs_dev.__locality_domains())
          {
            REQUIRE(lhs.green_context() != rhs.green_context());
            REQUIRE(lhs != rhs);
          }
        }
      }
    }
  }

  SECTION("Every device reports its own span")
  {
    // A shared cache slot would make two devices hand out the same storage.
    for (::cuda::std::size_t i = 0; i < cuda::devices.size(); ++i)
    {
      for (::cuda::std::size_t j = i + 1; j < cuda::devices.size(); ++j)
      {
        REQUIRE(cuda::devices[i].__locality_domains().data() != cuda::devices[j].__locality_domains().data());
      }
    }
  }

  SECTION("Second call returns the same span contents (cached)")
  {
    auto dev    = cuda::devices[0];
    auto first  = dev.__locality_domains();
    auto second = dev.__locality_domains();
    REQUIRE(first.size() == second.size());
    for (::cuda::std::size_t i = 0; i < first.size(); ++i)
    {
      REQUIRE(first[i] == second[i]);
    }
  }

  SECTION("The cache hands out the same storage, not a rebuilt copy")
  {
    // Comparing values alone would also pass if the domains were recreated, because the driver may
    // reuse a freed handle. Comparing the pointer proves that no second set of green contexts was
    // built.
    for (auto dev : cuda::devices)
    {
      auto first  = dev.__locality_domains();
      auto second = dev.__locality_domains();
      REQUIRE(first.data() == second.data());
      REQUIRE(first.size() == second.size());
    }
  }

  SECTION("The cache is reachable through any device_ref naming the same device")
  {
    // The cache lives in the physical device, not in the `device_ref`, so a freshly built ref must
    // observe it.
    for (auto dev : cuda::devices)
    {
      cuda::device_ref other{dev.get()};
      REQUIRE(other.__locality_domains().data() == dev.__locality_domains().data());
    }
  }

  SECTION("Querying other cached device properties does not disturb the domains")
  {
    // `peers()`, `name()` and the domains share one physical device object and each has its own
    // `once_flag`. Filling one must not reset another.
    auto dev    = cuda::devices[0];
    auto before = dev.__locality_domains();

    static_cast<void>(dev.name());
    static_cast<void>(dev.peers());
    dev.init();

    auto after = dev.__locality_domains();
    REQUIRE(before.data() == after.data());
    REQUIRE(before.size() == after.size());
  }

  SECTION("Each green context converts to a usable driver context on the right device")
  {
    for (auto dev : cuda::devices)
    {
      auto expected_device = cuda::__driver::__deviceGet(dev.get());
      for (auto& domain : dev.__locality_domains())
      {
        auto ctx = cuda::__driver::__ctxFromGreenCtx(domain.green_context());
        REQUIRE(ctx != nullptr);

        // The fixture checks that the driver stack is empty at test exit, so the push must be undone
        // even if a check throws.
        cuda::__ensure_current_context guard{domain};
        REQUIRE(cuda::__driver::__ctxGetCurrent() == ctx);
        REQUIRE(cuda::__driver::__ctxGetDevice() == expected_device);
      }
    }
  }

  SECTION("A green context is not the primary context")
  {
    // A domain must be a real partition, not a rebranded primary context.
    for (auto dev : cuda::devices)
    {
      auto primary = dev.__primary_context();
      for (auto& domain : dev.__locality_domains())
      {
        REQUIRE(cuda::__driver::__ctxFromGreenCtx(domain.green_context()) != primary);
      }
    }
  }

  SECTION("Reading the domains leaves the driver stack empty")
  {
    // The fixture checks this at test exit as well, but this pins the failure on this API.
    test::empty_driver_stack();
    for (auto dev : cuda::devices)
    {
      static_cast<void>(dev.__locality_domains());
    }
    REQUIRE(test::count_driver_stack() == 0);
  }

  SECTION("A domain never claims more SMs than the device has")
  {
    for (auto dev : cuda::devices)
    {
      auto device_sms = ::sm_count(dev);
      REQUIRE(device_sms > 0);
      // A partition cannot be larger than the whole, so the domain count is bounded by the SM count.
      REQUIRE(dev.__locality_domains().size() <= static_cast<::cuda::std::size_t>(device_sms));
    }
  }

  SECTION("Domain count matches expected count on supported devices")
  {
    for (auto dev : cuda::devices)
    {
#  if _CCCL_CTK_AT_LEAST(13, 4)
      const auto expected = static_cast<::cuda::std::size_t>(::cuda::__driver::__deviceGetAttribute(
        ::CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, ::cuda::__driver::__deviceGet(dev.get())));
#  else // ^^^ 13.4+ ^^^ / vvv 13.3- vvv
      const auto expected = 1;
#  endif // ^^^ 13.3- ^^^
      auto domains = dev.__locality_domains();

      REQUIRE(domains.size() == expected);
    }
  }

#  if _CCCL_CTK_AT_LEAST(13, 4)
  SECTION("The domains partition the device SMs on 13.4+")
  {
    for (auto dev : cuda::devices)
    {
      auto cu_device = ::cuda::__driver::__deviceGet(dev.get());
      auto expected  = static_cast<::cuda::std::size_t>(
        ::cuda::__driver::__deviceGetAttribute(::CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, cu_device));

      // Re-split the device the same way the implementation does and compare SM counts. This checks
      // that each cached domain really carries the SMs of its own locality domain. A device that
      // reports zero locality domains has no split to reproduce, so only the SM total is checked.
      auto full = ::cuda::__driver::__deviceGetDevResource(cu_device, ::CU_DEV_RESOURCE_TYPE_SM);
      REQUIRE(full.sm.smCount == static_cast<unsigned int>(::sm_count(dev)));

      auto params = ::cuda::std::make_unique<::CU_DEV_SM_RESOURCE_GROUP_PARAMS[]>(expected);
      for (::cuda::std::size_t i = 0; i < expected; ++i)
      {
        params[i].flags            = ::CU_DEV_SM_RESOURCE_GROUP_LOCALITY_DOMAIN_ID;
        params[i].localityDomainId = static_cast<unsigned int>(i);
      }

      if (expected != 0)
      {
        auto groups = ::cuda::std::make_unique<::CUdevResource[]>(expected);
        static_cast<void>(::cuda::__driver::__devSmResourceSplit(
          groups.get(), static_cast<unsigned int>(expected), full, params.get()));

        unsigned int total = 0;
        for (::cuda::std::size_t i = 0; i < expected; ++i)
        {
          // A locality domain that owns no SM would make the corresponding green context useless.
          REQUIRE(groups[i].sm.smCount > 0);
          total += groups[i].sm.smCount;
        }
        REQUIRE(total == full.sm.smCount);
      }
    }
  }
#  endif // _CCCL_CTK_AT_LEAST(13, 4)
}

#else // ^^^ _CCCL_CTK_AT_LEAST(12, 5) ^^^ / vvv _CCCL_CTK_BELOW(12, 5) vvv

C2H_CCCLRT_TEST("locality domains require CTK 12.5", "[device][locality_domain]")
{
  SUCCEED("Green contexts are unavailable before CTK 12.5");
}

#endif // ^^^ _CCCL_CTK_BELOW(12, 5) ^^^
