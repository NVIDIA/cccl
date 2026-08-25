//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__device/logical_device_ref.h>
#include <cuda/__memory_pool/locality_domain_memory_pool.h>
#include <cuda/__runtime/ensure_current_context.h>
#include <cuda/devices>
#include <cuda/memory_pool>
#include <cuda/std/cstddef>

#include <testing.cuh>

#include "pool_availability.cuh"

// Green contexts require CTK 12.5, so a locality domain cannot exist before that version.
#if _CCCL_CTK_AT_LEAST(12, 5)

C2H_CCCLRT_TEST("locality domain memory pool of a non-localized device", "[memory_resource][locality_domain]")
{
  test::skip_if_unsupported_memory_pool<cuda::device_memory_pool_ref>();

  SECTION("A ref with no green context yields the plain device default pool")
  {
    for (auto dev : cuda::devices)
    {
      const cuda::__logical_device_ref ref{dev, nullptr};

      auto& pool = cuda::__device_default_memory_pool(ref);

      // Not merely equal: it must be the very same cached object, so that a caller who sets an
      // attribute through one accessor observes it through the other.
      REQUIRE(&pool == &cuda::device_default_memory_pool(dev));
      REQUIRE(pool == cuda::device_default_memory_pool(dev));
    }
  }

  SECTION("Different non-localized devices yield different pools")
  {
    if (cuda::devices.size() > 1)
    {
      const cuda::__logical_device_ref first{cuda::devices[0], nullptr};
      const cuda::__logical_device_ref second{cuda::devices[1], nullptr};

      REQUIRE(cuda::__device_default_memory_pool(first) != cuda::__device_default_memory_pool(second));
    }
  }
}

C2H_CCCLRT_TEST("locality domain memory pool", "[memory_resource][locality_domain]")
{
  test::skip_if_unsupported_memory_pool<cuda::device_memory_pool_ref>();

  SECTION("Every domain has a pool with a real handle")
  {
    for (auto dev : cuda::devices)
    {
      for (auto& domain : dev.__locality_domains())
      {
        REQUIRE(cuda::__device_default_memory_pool(domain).get() != nullptr);
      }
    }
  }

  SECTION("The pool of a domain is not the device default pool")
  {
    // A localized pool that fell back to the device default would allocate from the wrong place and
    // silently lose the locality the caller asked for. A device that reports no locality domains
    // yields one whole-device ref, which correctly maps to the device default pool.
    for (auto dev : cuda::devices)
    {
      const auto device_pool = cuda::device_default_memory_pool(dev);

      for (auto& domain : dev.__locality_domains())
      {
        if (domain.locality_domain().localized)
        {
          REQUIRE(cuda::__device_default_memory_pool(domain) != device_pool);
        }
        else
        {
          REQUIRE(cuda::__device_default_memory_pool(domain) == device_pool);
        }
      }
    }
  }

  SECTION("Domains of one device have distinct pools")
  {
    for (auto dev : cuda::devices)
    {
      auto domains = dev.__locality_domains();

      for (cuda::std::size_t i = 0; i < domains.size(); ++i)
      {
        for (cuda::std::size_t j = i + 1; j < domains.size(); ++j)
        {
          auto& lhs = cuda::__device_default_memory_pool(domains[i]);
          auto& rhs = cuda::__device_default_memory_pool(domains[j]);

          REQUIRE(&lhs != &rhs);
          REQUIRE(lhs.get() != rhs.get());
          REQUIRE(lhs != rhs);
        }
      }
    }
  }

  SECTION("Domains of different devices have distinct pools")
  {
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
            REQUIRE(cuda::__device_default_memory_pool(lhs) != cuda::__device_default_memory_pool(rhs));
          }
        }
      }
    }
  }

  SECTION("The pools are cached, not rebuilt")
  {
    // Comparing handles alone would also pass if the pools were queried again, because the driver
    // returns the same default handle every time. Comparing the address proves that the `once_flag`
    // held and that the array was built once.
    for (auto dev : cuda::devices)
    {
      for (auto& domain : dev.__locality_domains())
      {
        auto& first  = cuda::__device_default_memory_pool(domain);
        auto& second = cuda::__device_default_memory_pool(domain);

        REQUIRE(&first == &second);
        REQUIRE(first == second);
      }
    }
  }

  SECTION("A freshly built ref reaches the same cached pool")
  {
    // The cache is keyed by device ordinal and domain id, not by the identity of the ref, so an
    // equal ref built by hand must land on the same pool.
    for (auto dev : cuda::devices)
    {
      for (auto& domain : dev.__locality_domains())
      {
        const cuda::__logical_device_ref rebuilt{domain.underlying_device(), domain.green_context()};

        REQUIRE(&cuda::__device_default_memory_pool(rebuilt) == &cuda::__device_default_memory_pool(domain));
      }
    }
  }

  SECTION("Reading a pool leaves the driver stack empty")
  {
    test::empty_driver_stack();

    for (auto dev : cuda::devices)
    {
      for (auto& domain : dev.__locality_domains())
      {
        static_cast<void>(cuda::__device_default_memory_pool(domain));
      }
    }
    REQUIRE(test::count_driver_stack() == 0);
  }

#  if _CCCL_CTK_AT_LEAST(13, 3)
  SECTION("Each pool reports its own device and locality domain")
  {
    for (auto dev : cuda::devices)
    {
      for (auto& domain : dev.__locality_domains())
      {
        auto& pool = cuda::__device_default_memory_pool(domain);

        REQUIRE(pool.attribute(cuda::memory_pool_attributes::allocation_type) == ::cudaMemAllocationTypePinned);

        const auto location = pool.attribute(cuda::memory_pool_attributes::location);

        REQUIRE(location.id == dev.get());
        // A whole-device ref maps to the device default pool, whose location type is `Device`.
        if (domain.locality_domain().localized)
        {
          // A pool whose location type stayed `Device` was not localized at all.
          REQUIRE(location.type != ::cudaMemLocationTypeDevice);
        }
      }
    }
  }
#  endif // _CCCL_CTK_AT_LEAST(13, 3)

  SECTION("Every domain pool allocates usable device memory")
  {
    for (auto dev : cuda::devices)
    {
      cuda::__ensure_current_context guard{dev};

      for (auto& domain : dev.__locality_domains())
      {
        auto& pool = cuda::__device_default_memory_pool(domain);

        auto* ptr = pool.allocate_sync(42);

        REQUIRE(ptr != nullptr);

        ::cudaPointerAttributes attributes{};
        REQUIRE(::cudaPointerGetAttributes(&attributes, ptr) == ::cudaSuccess);
        REQUIRE(attributes.type == ::cudaMemoryTypeDevice);
        REQUIRE(attributes.device == dev.get());

        pool.deallocate_sync(ptr, 42);
      }
    }
  }
}

#else // ^^^ _CCCL_CTK_AT_LEAST(12, 5) ^^^ / vvv _CCCL_CTK_BELOW(12, 5) vvv

// The test binary must contain at least one test case, otherwise Catch2 reports a failure.
C2H_CCCLRT_TEST("locality domain memory pools require CTK 12.5", "[memory_resource][locality_domain]")
{
  SUCCEED("Green contexts are unavailable before CTK 12.5");
}

#endif // ^^^ _CCCL_CTK_BELOW(12, 5) ^^^
