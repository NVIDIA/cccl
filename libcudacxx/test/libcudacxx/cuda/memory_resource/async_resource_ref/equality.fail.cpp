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

// cuda::mr::async_resource_ref equality

#include <cuda/memory_resource>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/stream_ref>

#include "types.h"

using ref = cuda::mr::async_resource_ref<cuda::mr::host_accessible,
                                         property_with_value<int>,
                                         property_with_value<double>,
                                         property_without_value<std::size_t>>;
using different_properties =
  cuda::mr::async_resource_ref<cuda::mr::host_accessible,
                               property_with_value<short>,
                               property_with_value<int>,
                               property_without_value<std::size_t>>;

using res = async_resource<cuda::mr::host_accessible,
                           property_with_value<int>,
                           property_with_value<double>,
                           property_without_value<std::size_t>>;

void test_equality()
{
  res input{42};
  res with_equal_value{42};
  res with_different_value{1337};

  // Requires matching properties
  assert(ref{input} == different_properties{with_equal_value});
  assert(ref{input} != different_properties{with_different_value});
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_equality();))

  return 0;
}
