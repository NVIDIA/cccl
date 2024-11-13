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

// cuda::mr::async_resource_ref properties

#include <cuda/memory_resource>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/stream_ref>

#include "types.h"

namespace properties_test
{
static_assert(cuda::property_with_value<property_with_value<int>>, "");
static_assert(cuda::property_with_value<property_with_value<struct someStruct>>, "");

static_assert(!cuda::property_with_value<property_without_value<int>>, "");
static_assert(!cuda::property_with_value<property_without_value<struct otherStruct>>, "");
} // namespace properties_test

namespace resource_test
{

// Ensure we have the right size
static_assert(
  sizeof(cuda::mr::async_resource_ref<cuda::mr::host_accessible, property_with_value<short>, property_with_value<int>>)
    == (4 * sizeof(void*)),
  "");
static_assert(
  sizeof(
    cuda::mr::async_resource_ref<cuda::mr::host_accessible, property_with_value<short>, property_without_value<int>>)
    == (3 * sizeof(void*)),
  "");
static_assert(
  sizeof(
    cuda::mr::async_resource_ref<cuda::mr::host_accessible, property_without_value<short>, property_with_value<int>>)
    == (3 * sizeof(void*)),
  "");
static_assert(
  sizeof(
    cuda::mr::async_resource_ref<cuda::mr::host_accessible, property_without_value<short>, property_without_value<int>>)
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
void test_async_resource_ref()
{
  constexpr int expected_initially = 42;
  async_resource<cuda::mr::host_accessible, Properties...> input{expected_initially};
  cuda::mr::async_resource_ref<cuda::mr::host_accessible, Properties...> ref{input};

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
  using res = async_resource<cuda::mr::host_accessible, property_with_value<short>, property_with_value<int>>;
  using ref = cuda::mr::async_resource_ref<cuda::mr::host_accessible, property_with_value<short>>;

  static_assert(cuda::mr::async_resource_with<res, property_with_value<short>, property_with_value<int>>, "");
  static_assert(!cuda::mr::async_resource_with<ref, property_with_value<short>, property_with_value<int>>, "");

  static_assert(cuda::mr::async_resource_with<res, property_with_value<short>>, "");
}

void test_async_resource_ref()
{
  // Test some basic combinations of properties w/o state
  test_async_resource_ref<property_with_value<short>, property_with_value<int>>();
  test_async_resource_ref<property_with_value<short>, property_without_value<int>>();
  test_async_resource_ref<property_without_value<short>, property_without_value<int>>();

  // Test duplicated properties
  test_async_resource_ref<property_with_value<short>, property_with_value<int>, property_with_value<short>>();

  test_async_resource_ref<property_without_value<short>, property_without_value<int>, property_without_value<short>>();

  // Ensure we only forward requested properties
  test_property_forwarding();
}
} // namespace resource_test

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (resource_test::test_async_resource_ref();))

  return 0;
}
