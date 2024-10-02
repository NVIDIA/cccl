//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc

// cuda::mr::resource_ref properties

#include <cuda/memory_resource>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "types.h"

template <class... Properties>
struct async_resource_base
{
  virtual ~async_resource_base() = default;

  virtual void* allocate(std::size_t, std::size_t) = 0;

  virtual void deallocate(void* ptr, std::size_t, std::size_t) noexcept = 0;

  virtual void* allocate_async(std::size_t, std::size_t, cuda::stream_ref) = 0;

  virtual void deallocate_async(void* ptr, std::size_t, std::size_t, cuda::stream_ref) = 0;

  bool operator==(const async_resource_base& other) const
  {
    return true;
  }
  bool operator!=(const async_resource_base& other) const
  {
    return false;
  }

  _LIBCUDACXX_TEMPLATE(class Property)
  _LIBCUDACXX_REQUIRES((!cuda::property_with_value<Property>)
                       && _CUDA_VSTD::__is_included_in<Property, Properties...>) //
  friend void get_property(const async_resource_base&, Property) noexcept {}

  _LIBCUDACXX_TEMPLATE(class Property)
  _LIBCUDACXX_REQUIRES(cuda::property_with_value<Property>&& _CUDA_VSTD::__is_included_in<Property, Properties...>) //
  friend typename Property::value_type get_property(const async_resource_base& res, Property) noexcept
  {
    return 42;
  }
};

template <class... Properties>
struct async_resource_derived_first : public async_resource_base<Properties...>
{
  using super_t = async_resource_base<Properties...>;

  async_resource_derived_first(const int val)
      : _val(val)
  {}

  void* allocate(std::size_t, std::size_t) override
  {
    return &_val;
  }

  void deallocate(void* ptr, std::size_t, std::size_t) noexcept override {}

  void* allocate_async(std::size_t, std::size_t, cuda::stream_ref) override
  {
    return &_val;
  }

  void deallocate_async(void* ptr, std::size_t, std::size_t, cuda::stream_ref) override {}

  bool operator==(const async_resource_derived_first& other) const
  {
    return true;
  }
  bool operator!=(const async_resource_derived_first& other) const
  {
    return false;
  }

  int _val = 0;
};
static_assert(cuda::mr::async_resource<async_resource_derived_first<cuda::mr::host_accessible>>, "");

struct some_data
{
  int _val;
};

template <class... Properties>
struct async_resource_derived_second : public async_resource_base<Properties...>
{
  using super_t = async_resource_base<Properties...>;

  async_resource_derived_second(some_data* val)
      : _val(val)
  {}

  void* allocate(std::size_t, std::size_t) override
  {
    return &_val->_val;
  }

  void deallocate(void* ptr, std::size_t, std::size_t) noexcept override {}

  void* allocate_async(std::size_t, std::size_t, cuda::stream_ref) override
  {
    return &_val->_val;
  }

  void deallocate_async(void* ptr, std::size_t, std::size_t, cuda::stream_ref) override {}

  bool operator==(const async_resource_derived_second& other) const
  {
    return true;
  }
  bool operator!=(const async_resource_derived_second& other) const
  {
    return false;
  }

  some_data* _val = 0;
};

template <class... Properties>
void test_async_resource_ref()
{
  some_data input{1337};
  async_resource_derived_first<cuda::mr::host_accessible, Properties...> first{42};
  async_resource_derived_second<cuda::mr::host_accessible, Properties...> second{&input};

  cuda::mr::async_resource_ref<cuda::mr::host_accessible, Properties...> ref_first{first};
  cuda::mr::async_resource_ref<cuda::mr::host_accessible, Properties...> ref_second{second};

  // Ensure that we properly pass on the allocate function
  assert(ref_first.allocate_async(0, 0, {}) == first.allocate_async(0, 0, {}));
  assert(ref_second.allocate_async(0, 0, {}) == second.allocate_async(0, 0, {}));

  // Ensure that assignment still works
  ref_second = ref_first;
  assert(ref_second.allocate_async(0, 0, {}) == first.allocate_async(0, 0, {}));
}

template <class... Properties>
cuda::mr::async_resource_ref<cuda::mr::host_accessible, Properties...>
indirection(async_resource_base<cuda::mr::host_accessible, Properties...>* res)
{
  return {res};
}

template <class... Properties>
void test_async_resource_ref_from_pointer()
{
  some_data input{1337};
  async_resource_derived_first<cuda::mr::host_accessible, Properties...> first{42};
  async_resource_derived_second<cuda::mr::host_accessible, Properties...> second{&input};

  cuda::mr::async_resource_ref<cuda::mr::host_accessible, Properties...> ref_first  = indirection(&first);
  cuda::mr::async_resource_ref<cuda::mr::host_accessible, Properties...> ref_second = indirection(&second);

  // Ensure that we properly pass on the allocate function
  assert(ref_first.allocate_async(0, 0, {}) == first.allocate_async(0, 0, {}));
  assert(ref_second.allocate_async(0, 0, {}) == second.allocate_async(0, 0, {}));

  // Ensure that assignment still works
  ref_second = ref_first;
  assert(ref_second.allocate_async(0, 0, {}) == first.allocate_async(0, 0, {}));
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
      test_async_resource_ref<property_with_value<short>, property_with_value<int>>();
      test_async_resource_ref<property_with_value<short>, property_without_value<int>>();
      test_async_resource_ref<property_without_value<short>, property_without_value<int>>();

      test_async_resource_ref_from_pointer<property_with_value<short>, property_with_value<int>>();
      test_async_resource_ref_from_pointer<property_with_value<short>, property_without_value<int>>();
      test_async_resource_ref_from_pointer<property_without_value<short>, property_without_value<int>>();))

  return 0;
}
