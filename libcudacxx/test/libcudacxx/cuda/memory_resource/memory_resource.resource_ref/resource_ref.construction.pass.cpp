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

// cuda::mr::resource_ref construction

#include <cuda/memory_resource>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

template <class T>
struct property_with_value
{
  using value_type = T;
};

template <class T>
struct property_without_value
{};

template <class... Properties>
struct resource
{
  void* allocate(std::size_t, std::size_t)
  {
    return &_val;
  }

  void deallocate(void* ptr, std::size_t, std::size_t) noexcept
  {
    // ensure that we did get the right inputs forwarded
    _val = *static_cast<int*>(ptr);
  }

  bool operator==(const resource& other) const
  {
    return _val == other._val;
  }
  bool operator!=(const resource& other) const
  {
    return _val != other._val;
  }

  int _val = 0;

  _LIBCUDACXX_TEMPLATE(class Property)
  _LIBCUDACXX_REQUIRES((!cuda::property_with_value<Property>) && _CUDA_VSTD::_One_of<Property, Properties...>) //
  friend void get_property(const resource&, Property) noexcept {}

  _LIBCUDACXX_TEMPLATE(class Property)
  _LIBCUDACXX_REQUIRES(cuda::property_with_value<Property>&& _CUDA_VSTD::_One_of<Property, Properties...>) //
  friend typename Property::value_type get_property(const resource& res, Property) noexcept
  {
    return res._val;
  }
};

namespace constructible
{
using ref =
  cuda::mr::resource_ref<property_with_value<int>, property_with_value<double>, property_without_value<std::size_t>>;

using matching_properties =
  resource<property_with_value<double>, property_without_value<std::size_t>, property_with_value<int>>;

using missing_stateful_property  = resource<property_with_value<int>, property_without_value<std::size_t>>;
using missing_stateless_property = resource<property_with_value<int>, property_with_value<double>>;

using cuda::std::is_constructible;
static_assert(is_constructible<ref, matching_properties&>::value, "");
static_assert(!is_constructible<ref, missing_stateful_property&>::value, "");
static_assert(!is_constructible<ref, missing_stateless_property&>::value, "");

static_assert(is_constructible<ref, matching_properties*>::value, "");
static_assert(!is_constructible<ref, missing_stateful_property*>::value, "");
static_assert(!is_constructible<ref, missing_stateless_property*>::value, "");

static_assert(is_constructible<ref, ref&>::value, "");

// Ensure we require a mutable valid reference and do not bind against rvalues
static_assert(!is_constructible<ref, matching_properties>::value, "");
static_assert(!is_constructible<ref, const matching_properties&>::value, "");
static_assert(!is_constructible<ref, const matching_properties*>::value, "");

static_assert(cuda::std::is_copy_constructible<ref>::value, "");
static_assert(cuda::std::is_move_constructible<ref>::value, "");
} // namespace constructible

namespace assignable
{
using ref =
  cuda::mr::resource_ref<property_with_value<int>, property_with_value<double>, property_without_value<std::size_t>>;

using res = resource<property_with_value<int>, property_with_value<double>, property_without_value<std::size_t>>;

using other_res =
  resource<property_without_value<int>,
           property_with_value<int>,
           property_with_value<double>,
           property_without_value<std::size_t>>;

using cuda::std::is_assignable;
static_assert(cuda::std::is_assignable<ref, res&>::value, "");
static_assert(cuda::std::is_assignable<ref, other_res&>::value, "");

static_assert(cuda::std::is_copy_assignable<ref>::value, "");
static_assert(cuda::std::is_move_assignable<ref>::value, "");
} // namespace assignable

int main(int, char**)
{
  return 0;
}
