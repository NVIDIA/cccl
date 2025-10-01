//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc

// cuda::mr::synchronous_resource_ref properties

#include <cuda/memory_resource>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "types.h"

template <class... Properties>
struct resource_base
{
  virtual ~resource_base() = default;

  virtual void* allocate_sync(std::size_t, std::size_t) = 0;

  virtual void deallocate_sync(void* ptr, std::size_t, std::size_t) noexcept = 0;

  bool operator==(const resource_base& other) const
  {
    return true;
  }
  bool operator!=(const resource_base& other) const
  {
    return false;
  }

  _CCCL_TEMPLATE(class Property)
  _CCCL_REQUIRES((!cuda::property_with_value<Property>) && ::cuda::std::__is_included_in_v<Property, Properties...>) //
  friend void get_property(const resource_base&, Property) noexcept {}

  _CCCL_TEMPLATE(class Property)
  _CCCL_REQUIRES(cuda::property_with_value<Property>&& ::cuda::std::__is_included_in_v<Property, Properties...>) //
  friend typename Property::value_type get_property(const resource_base& res, Property) noexcept
  {
    return 42;
  }
};

template <class... Properties>
struct resource_derived_first : public resource_base<Properties...>
{
  using super_t = resource_base<Properties...>;

  resource_derived_first(const int val)
      : _val(val)
  {}

  void* allocate_sync(std::size_t, std::size_t) override
  {
    return &_val;
  }

  void deallocate_sync(void* ptr, std::size_t, std::size_t) noexcept override {}

  bool operator==(const resource_derived_first& other) const
  {
    return true;
  }
  bool operator!=(const resource_derived_first& other) const
  {
    return false;
  }

  int _val = 0;
};
static_assert(cuda::mr::synchronous_resource<resource_derived_first<>>, "");

struct some_data
{
  int _val;
};

template <class... Properties>
struct resource_derived_second : public resource_base<Properties...>
{
  using super_t = resource_base<Properties...>;

  resource_derived_second(some_data* val)
      : _val(val)
  {}

  void* allocate_sync(std::size_t, std::size_t) override
  {
    return &_val->_val;
  }

  void deallocate_sync(void* ptr, std::size_t, std::size_t) noexcept override {}

  bool operator==(const resource_derived_second& other) const
  {
    return true;
  }
  bool operator!=(const resource_derived_second& other) const
  {
    return false;
  }

  some_data* _val = 0;
};

template <class... Properties>
void test_resource_ref()
{
  some_data input{1337};
  resource_derived_first<cuda::mr::host_accessible, Properties...> first{42};
  resource_derived_second<cuda::mr::host_accessible, Properties...> second{&input};

  cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible, Properties...> ref_first{first};
  cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible, Properties...> ref_second{second};

  // Ensure that we properly pass on the allocate function
  assert(ref_first.allocate_sync(0, 0) == first.allocate_sync(0, 0));
  assert(ref_second.allocate_sync(0, 0) == second.allocate_sync(0, 0));

  // Ensure that assignment still works
  ref_second = ref_first;
  assert(ref_second.allocate_sync(0, 0) == first.allocate_sync(0, 0));
}

template <class... Properties>
cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible, Properties...>
indirection(resource_base<cuda::mr::host_accessible, Properties...>* res)
{
  return {res};
}

template <class... Properties>
void test_resource_ref_from_pointer()
{
  some_data input{1337};
  resource_derived_first<cuda::mr::host_accessible, Properties...> first{42};
  resource_derived_second<cuda::mr::host_accessible, Properties...> second{&input};

  cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible, Properties...> ref_first  = indirection(&first);
  cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible, Properties...> ref_second = indirection(&second);

  // Ensure that we properly pass on the allocate function
  assert(ref_first.allocate_sync(0, 0) == first.allocate_sync(0, 0));
  assert(ref_second.allocate_sync(0, 0) == second.allocate_sync(0, 0));

  // Ensure that assignment still works
  ref_second = ref_first;
  assert(ref_second.allocate_sync(0, 0) == first.allocate_sync(0, 0));
}

// clang complains about pure virtual functions being called, so ensure that we properly crash if so
extern "C" void __cxa_pure_virtual()
{
  while (1)
    ;
}

int main(int, char**)
{
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      // Test some basic combinations of properties w/o state
      test_resource_ref<property_with_value<short>, property_with_value<int>>();
      test_resource_ref<property_with_value<short>, property_without_value<int>>();
      test_resource_ref<property_without_value<short>, property_without_value<int>>();

      test_resource_ref_from_pointer<property_with_value<short>, property_with_value<int>>();
      test_resource_ref_from_pointer<property_with_value<short>, property_without_value<int>>();
      test_resource_ref_from_pointer<property_without_value<short>, property_without_value<int>>();))

  return 0;
}
