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

// cuda::mr::resource_ref construction

#include <cuda/memory_resource>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include "types.h"

namespace constructible
{
using ref = cuda::mr::resource_ref<cuda::mr::host_accessible,
                                   property_with_value<int>,
                                   property_with_value<double>,
                                   property_without_value<std::size_t>>;

using matching_properties =
  resource<cuda::mr::host_accessible,
           property_with_value<double>,
           property_without_value<std::size_t>,
           property_with_value<int>>;

using missing_stateful_property =
  resource<cuda::mr::host_accessible, property_with_value<int>, property_without_value<std::size_t>>;
using missing_stateless_property =
  resource<cuda::mr::host_accessible, property_with_value<int>, property_with_value<double>>;

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
using ref = cuda::mr::resource_ref<cuda::mr::host_accessible,
                                   property_with_value<int>,
                                   property_with_value<double>,
                                   property_without_value<std::size_t>>;

using res = resource<cuda::mr::host_accessible,
                     property_with_value<int>,
                     property_with_value<double>,
                     property_without_value<std::size_t>>;

using other_res =
  resource<cuda::mr::host_accessible,
           property_without_value<int>,
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
