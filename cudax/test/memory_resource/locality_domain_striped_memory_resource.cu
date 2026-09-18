//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__cccl_config>

#include <cuda/experimental/memory_resource.cuh>

#include "testing.cuh"

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC) && _CCCL_CTK_AT_LEAST(13, 4)

#  include <cuda/__driver/driver_api.h>
#  include <cuda/std/type_traits>

#  include <stdexcept>
#  include <string>

#  include <cuda.h>

namespace
{
struct test_env
{
  ::cuda::device_ref device;
  ::cuda::std::size_t stride;
  ::cuda::std::size_t locality_domains;
};

[[nodiscard]] ::CUmemAllocationProp make_prop(::cuda::device_ref device, ::cuda::std::size_t domain)
{
  ::CUmemAllocationProp prop{};
  prop.type                                = ::CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type                       = ::CU_MEM_LOCATION_TYPE_DEVICE_LOCALITY_DOMAIN;
  prop.location.localized.deviceId         = static_cast<unsigned char>(device.get());
  prop.location.localized.localityDomainId = static_cast<unsigned char>(domain);
  return prop;
}

[[nodiscard]] test_env make_test_env()
{
  try
  {
    const auto device_count = ::cuda::__driver::__deviceGetCount();
    if (device_count == 0)
    {
      SKIP("No CUDA devices visible");
    }

    ::cuda::device_ref device{0};
    const auto cu_device = ::cuda::__driver::__deviceGet(device.get());
    const auto vmm_supported =
      ::cuda::__driver::__deviceGetAttribute(::CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, cu_device);
    if (vmm_supported == 0)
    {
      SKIP("Virtual memory management is not supported");
    }

    const auto domain_count =
      ::cuda::__driver::__deviceGetAttribute(::CU_DEVICE_ATTRIBUTE_LOCALITY_DOMAIN_COUNT, cu_device);
    if (domain_count <= 0)
    {
      SKIP("Locality domains are not supported");
    }

    const auto prop = make_prop(device, 0);
    const auto stride =
      ::cuda::experimental::__driver::__memGetAllocationGranularity(&prop, ::CU_MEM_ALLOC_GRANULARITY_MINIMUM);

    return {device, stride, static_cast<::cuda::std::size_t>(domain_count)};
  }
  catch (const ::cuda::cuda_error& error)
  {
    SKIP("CUDA driver API prerequisite query failed: " << error.what());
  }

  return {::cuda::device_ref{0}, 1, 1};
}
} // namespace

TEST_CASE("locality_domain_striped_memory_resource traits", "[memory_resource]")
{
  using resource = cudax::locality_domain_striped_memory_resource;

  STATIC_REQUIRE(::cuda::std::is_copy_constructible_v<resource>);
  STATIC_REQUIRE(::cuda::std::is_copy_assignable_v<resource>);
  STATIC_REQUIRE(::cuda::std::is_move_constructible_v<resource>);
  STATIC_REQUIRE(::cuda::std::is_move_assignable_v<resource>);
  STATIC_REQUIRE(::cuda::mr::synchronous_resource_with<resource, ::cuda::mr::device_accessible>);
  STATIC_REQUIRE(::cuda::mr::synchronous_resource_with<resource, cudax::locality_domain_stride_t>);
  STATIC_REQUIRE(::cuda::has_property_with<resource, cudax::locality_domain_stride_t, ::cuda::std::size_t>);
}

C2H_CCCLRT_TEST("locality_domain_striped_memory_resource rejects zero stride", "[memory_resource]")
{
  const auto env = make_test_env();

  bool caught = false;
  try
  {
    (void) cudax::locality_domain_striped_memory_resource{env.device, 0};
  }
  catch (const ::std::invalid_argument& error)
  {
    caught = true;
    REQUIRE(::std::string{error.what()} == "locality_domain_striped_memory_resource requires a non-zero stride");
  }
  REQUIRE(caught);

  if (env.stride > 1)
  {
    caught = false;
    try
    {
      (void) cudax::locality_domain_striped_memory_resource{env.device, env.stride - 1};
    }
    catch (const ::std::invalid_argument& error)
    {
      caught = true;
      REQUIRE(::std::string{error.what()}
              == "locality_domain_striped_memory_resource stride must be a multiple of the minimum VMM allocation "
                 "granularity");
    }
    REQUIRE(caught);
  }
}

C2H_CCCLRT_TEST("locality_domain_striped_memory_resource basic allocation", "[memory_resource]")
{
  const auto env = make_test_env();

  cudax::locality_domain_striped_memory_resource resource{env.device, env.stride};
  REQUIRE(resource.device() == env.device);
  REQUIRE(resource.stride() == env.stride);
  REQUIRE(resource.locality_domain_count() == env.locality_domains);
  REQUIRE(get_property(resource, cudax::locality_domain_stride) == env.stride);

  cudax::locality_domain_striped_memory_resource copy = resource;
  REQUIRE(copy == resource);
  REQUIRE(get_property(copy, cudax::locality_domain_stride) == env.stride);

  REQUIRE(resource.allocate_sync(0) == nullptr);
  copy.deallocate_sync(nullptr, 0);

  void* ptr = resource.allocate_sync(1);
  REQUIRE(ptr != nullptr);
  copy.deallocate_sync(ptr, 1);
}

C2H_CCCLRT_TEST("locality_domain_striped_memory_resource rejects invalid alignment", "[memory_resource]")
{
  const auto env = make_test_env();
  cudax::locality_domain_striped_memory_resource resource{env.device, env.stride};

  bool caught = false;
  try
  {
    (void) resource.allocate_sync(1, 3);
  }
  catch (const ::std::invalid_argument& error)
  {
    caught = true;
    REQUIRE(::std::string{error.what()}
            == "Invalid alignment passed to locality_domain_striped_memory_resource::allocate_sync.");
  }
  REQUIRE(caught);
}

#else // ^^^ CTK 13.4+ ^^^ / vvv older toolkits vvv

TEST_CASE("locality_domain_striped_memory_resource unavailable before CUDA 13.4", "[memory_resource]")
{
  SUCCEED();
}

#endif // older toolkits
