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

#include <cuda/memory_resource>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

// Verify that the properties exist
static_assert(cuda::std::is_empty<cuda::mr::host_accessible>::value, "");
static_assert(cuda::std::is_empty<cuda::mr::device_accessible>::value, "");

// Verify that host accessible is the default if nothing is specified
static_assert(!cuda::mr::__is_host_accessible<>, "");
static_assert(cuda::mr::__is_host_accessible<cuda::mr::host_accessible>, "");
static_assert(!cuda::mr::__is_host_accessible<cuda::mr::device_accessible>, "");
static_assert(cuda::mr::__is_host_accessible<cuda::mr::host_accessible, cuda::mr::device_accessible>, "");

// Verify that device accessible needs to be explicitly specified
static_assert(!cuda::mr::__is_device_accessible<>, "");
static_assert(!cuda::mr::__is_device_accessible<cuda::mr::host_accessible>, "");
static_assert(cuda::mr::__is_device_accessible<cuda::mr::device_accessible>, "");
static_assert(cuda::mr::__is_device_accessible<cuda::mr::host_accessible, cuda::mr::device_accessible>, "");

// Verify that host device accessible needs to be explicitly specified
static_assert(!cuda::mr::__is_host_device_accessible<>, "");
static_assert(!cuda::mr::__is_host_device_accessible<cuda::mr::host_accessible>, "");
static_assert(!cuda::mr::__is_host_device_accessible<cuda::mr::device_accessible>, "");
static_assert(cuda::mr::__is_host_device_accessible<cuda::mr::host_accessible, cuda::mr::device_accessible>, "");

int main(int, char**)
{
  return 0;
}
