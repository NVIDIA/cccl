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

// Verify that we properly account for host_accessible being implict when nothing is specified
static_assert(cuda::mr::__is_valid_subset_v<cuda::std::__type_set<>, cuda::mr::host_accessible>, "");
static_assert(cuda::mr::__is_valid_subset_v<cuda::std::__type_set<cuda::mr::host_accessible>, int>, "");

// Verify that we properly require device_accessible to be specified
static_assert(!cuda::mr::__is_valid_subset_v<cuda::std::__type_set<>, cuda::mr::device_accessible>, "");
static_assert(!cuda::mr::__is_valid_subset_v<cuda::std::__type_set<cuda::mr::device_accessible>, int>, "");

// Verify that we properly allow subsets of execution space modifiers
static_assert(
  cuda::mr::__is_valid_subset_v<cuda::std::__type_set<cuda::mr::device_accessible>, cuda::mr::device_accessible>, "");
static_assert(cuda::mr::__is_valid_subset_v<cuda::std::__type_set<cuda::mr::device_accessible>,
                                            cuda::mr::host_accessible,
                                            cuda::mr::device_accessible>,
              "");
static_assert(cuda::mr::__is_valid_subset_v<cuda::std::__type_set<cuda::mr::device_accessible>,
                                            cuda::mr::host_accessible,
                                            cuda::mr::device_accessible>,
              "");

int main(int, char**)
{
  return 0;
}
