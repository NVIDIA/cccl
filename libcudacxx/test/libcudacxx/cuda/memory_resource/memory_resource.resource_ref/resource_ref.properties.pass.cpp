//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: msvc-19.16
// UNSUPPORTED: nvrtc

// cuda::mr::resource_ref properties

#include <cuda/memory_resource>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

template <class T>
struct property
{
  using value_type = T;
};

template <class T>
struct property_without_value
{};

namespace properties_test
{
static_assert(cuda::property_with_value<property<int>>, "");
static_assert(cuda::property_with_value<property<struct someStruct>>, "");

static_assert(!cuda::property_with_value<property_without_value<int>>, "");
static_assert(!cuda::property_with_value<property_without_value<struct otherStruct>>, "");
} // namespace properties_test

namespace resource_test
{
template <class... Properties>
struct resource
{
  void* allocate(std::size_t, std::size_t)
  {
    return nullptr;
  }

  void deallocate(void* ptr, std::size_t, std::size_t) noexcept {}

  bool operator==(const resource& other) const
  {
    return true;
  }
  bool operator!=(const resource& other) const
  {
    return false;
  }

  int _val = 0;

  _LIBCUDACXX_TEMPLATE(class Property)
  _LIBCUDACXX_REQUIRES((!cuda::property_with_value<Property>) && _CUDA_VSTD::_One_of<Property, Properties...>)
  friend void get_property(const resource&, Property) noexcept {}

  _LIBCUDACXX_TEMPLATE(class Property)
  _LIBCUDACXX_REQUIRES(cuda::property_with_value<Property>&& _CUDA_VSTD::_One_of<Property, Properties...>)
  friend typename Property::value_type get_property(const resource& res, Property) noexcept
  {
    return static_cast<typename Property::value_type>(res._val);
  }
};

// Ensure we have the right size
static_assert(sizeof(cuda::mr::resource_ref<property<short>, property<int>>) == (4 * sizeof(void*)), "");
static_assert(sizeof(cuda::mr::resource_ref<property<short>, property_without_value<int>>) == (3 * sizeof(void*)), "");
static_assert(sizeof(cuda::mr::resource_ref<property_without_value<short>, property<int>>) == (3 * sizeof(void*)), "");
static_assert(sizeof(cuda::mr::resource_ref<property_without_value<short>, property_without_value<int>>)
                == (2 * sizeof(void*)),
              "");

_LIBCUDACXX_TEMPLATE(class Property, class Ref)
_LIBCUDACXX_REQUIRES((!cuda::property_with_value<Property>) ) //
int InvokeIfWithValue(const Ref& ref)
{
  return -1;
}

_LIBCUDACXX_TEMPLATE(class Property, class Ref)
_LIBCUDACXX_REQUIRES(cuda::property_with_value<Property>) //
typename Property::value_type InvokeIfWithValue(const Ref& ref)
{
  return get_property(ref, Property{});
}

_LIBCUDACXX_TEMPLATE(class Property, class Ref)
_LIBCUDACXX_REQUIRES(cuda::property_with_value<Property>) //
int InvokeIfWithoutValue(const Ref& ref)
{
  return -1;
}

_LIBCUDACXX_TEMPLATE(class Property, class Ref)
_LIBCUDACXX_REQUIRES((!cuda::property_with_value<Property>) ) //
int InvokeIfWithoutValue(const Ref& ref)
{
  get_property(ref, Property{});
  return 1;
}

template <class... Properties>
void test_resource_ref()
{
  constexpr int expected_initially = 42;
  resource<Properties...> input{expected_initially};
  cuda::mr::resource_ref<Properties...> ref{input};

  // Check all the potentially stateful properties
  const int properties_with_value[] = {InvokeIfWithValue<Properties>(ref)...};
  const int expected_with_value[]   = {((cuda::property_with_value<Properties>) ? expected_initially : -1)...};
  for (std::size_t i = 0; i < sizeof...(Properties); ++i)
  {
    assert(properties_with_value[i] == expected_with_value[i]);
  }

  const int properties_without_value[] = {InvokeIfWithoutValue<Properties>(ref)...};
  const int expected_without_value[]   = {((cuda::property_with_value<Properties>) ? -1 : 1)...};
  for (std::size_t i = 0; i < sizeof...(Properties); ++i)
  {
    assert(properties_without_value[i] == expected_without_value[i]);
  }

  constexpr int expected_after_change = 1337;
  input._val                          = expected_after_change;

  // Check whether we truly get the right value
  const int properties_with_value2[] = {InvokeIfWithValue<Properties>(ref)...};
  const int expected_with_value2[]   = {((cuda::property_with_value<Properties>) ? expected_after_change : -1)...};
  for (std::size_t i = 0; i < sizeof...(Properties); ++i)
  {
    assert(properties_with_value2[i] == expected_with_value2[i]);
  }
}

void test_property_forwarding()
{
  using res = resource<property<short>, property<int>>;
  using ref = cuda::mr::resource_ref<property<short>>;

  static_assert(cuda::mr::resource_with<res, property<short>, property<int>>, "");
  static_assert(!cuda::mr::resource_with<ref, property<short>, property<int>>, "");

  static_assert(cuda::mr::resource_with<res, property<short>>, "");
}

void test_resource_ref()
{
  // Test some basic combinations of properties w/o state
  test_resource_ref<property<short>, property<int>>();
  test_resource_ref<property<short>, property_without_value<int>>();
  test_resource_ref<property_without_value<short>, property_without_value<int>>();

  // Test duplicated properties
  test_resource_ref<property<short>, property<int>, property<short>>();

  test_resource_ref<property_without_value<short>, property_without_value<int>, property_without_value<short>>();

  // Ensure we only forward requested properties
  test_property_forwarding();
}

} // namespace resource_test

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (resource_test::test_resource_ref();))

  return 0;
}
