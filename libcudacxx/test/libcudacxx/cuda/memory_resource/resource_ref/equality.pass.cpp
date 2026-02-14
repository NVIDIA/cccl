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

// cuda::mr::synchronous_resource_ref equality

#include <cuda/memory_resource>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>

#include "types.h"

using ref = cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible,
                                               property_with_value<int>,
                                               property_with_value<double>,
                                               property_without_value<std::size_t>>;

using pertubed_properties =
  cuda::mr::synchronous_resource_ref<cuda::mr::host_accessible,
                                     property_with_value<double>,
                                     property_with_value<int>,
                                     property_without_value<std::size_t>>;

using res = resource<cuda::mr::host_accessible,
                     property_with_value<int>,
                     property_with_value<double>,
                     property_without_value<std::size_t>>;
using other_res =
  resource<cuda::mr::host_accessible,
           property_with_value<double>,
           property_with_value<int>,
           property_without_value<std::size_t>>;

void test_equality()
{
  res input{42};
  res with_equal_value{42};
  res with_different_value{1337};

  assert(input == with_equal_value);
  assert(input != with_different_value);

  assert(ref{input} == ref{with_equal_value});
  assert(ref{input} != ref{with_different_value});

  // Should ignore pertubed properties
  assert(ref{input} == pertubed_properties{with_equal_value});
  assert(ref{input} != pertubed_properties{with_different_value});

  // Should reject different resources
  other_res other_with_matching_value{42};
  other_res other_with_different_value{1337};
  assert(ref{input} != ref{other_with_matching_value});
  assert(ref{input} != ref{other_with_different_value});

  assert(ref{input} != pertubed_properties{other_with_matching_value});
  assert(ref{input} != pertubed_properties{other_with_matching_value});
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_equality();))

  return 0;
}
