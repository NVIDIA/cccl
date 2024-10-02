//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_MEMORY_RESOURCE_RESOURCE_REF_TYPES_H
#define TEST_MEMORY_RESOURCE_RESOURCE_REF_TYPES_H

#include <cuda/memory_resource>

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

  void* allocate_async(std::size_t, std::size_t, cuda::stream_ref)
  {
    return &_val;
  }

  void deallocate_async(void* ptr, std::size_t, std::size_t, cuda::stream_ref)
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
  _LIBCUDACXX_REQUIRES((!cuda::property_with_value<Property>) && _CUDA_VSTD::__is_included_in<Property, Properties...>)
  friend void get_property(const resource&, Property) noexcept {}

  _LIBCUDACXX_TEMPLATE(class Property)
  _LIBCUDACXX_REQUIRES(cuda::property_with_value<Property>&& _CUDA_VSTD::__is_included_in<Property, Properties...>)
  friend typename Property::value_type get_property(const resource& res, Property) noexcept
  {
    return static_cast<typename Property::value_type>(res._val);
  }
};

#endif // TEST_MEMORY_RESOURCE_RESOURCE_REF_TYPES_H
