//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/memory_resource>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/initializer_list>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/container.cuh>

#include "helper.h"
#include "types.h"
#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE("cudax::async_mdarray properties",
                   "[container][async_mdarray]",
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   cuda::std::tuple<cuda::mr::device_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Array                  = typename extract_properties<TestType>::async_mdarray;
  using iterator               = typename extract_properties<TestType>::iterator;
  using const_iterator         = typename extract_properties<TestType>::const_iterator;
  using reverse_iterator       = cuda::std::reverse_iterator<iterator>;
  using const_reverse_iterator = cuda::std::reverse_iterator<const_iterator>;

  // Check the type aliases
  static_assert(cuda::std::is_same<int, typename Array::value_type>::value, "");
  static_assert(cuda::std::is_same<cuda::std::size_t, typename Array::size_type>::value, "");
  static_assert(cuda::std::is_same<cuda::std::ptrdiff_t, typename Array::difference_type>::value, "");
  static_assert(cuda::std::is_same<int*, typename Array::pointer>::value, "");
  static_assert(cuda::std::is_same<const int*, typename Array::const_pointer>::value, "");
  static_assert(cuda::std::is_same<int&, typename Array::reference>::value, "");
  static_assert(cuda::std::is_same<const int&, typename Array::const_reference>::value, "");
  static_assert(cuda::std::is_same<iterator, typename Array::iterator>::value, "");
  static_assert(cuda::std::is_same<const_iterator, typename Array::const_iterator>::value, "");
  static_assert(cuda::std::is_same<cuda::std::reverse_iterator<iterator>, typename Array::reverse_iterator>::value, "");
  static_assert(
    cuda::std::is_same<cuda::std::reverse_iterator<const_iterator>, typename Array::const_reverse_iterator>::value, "");

#if TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)
  static_assert(cuda::std::ranges::contiguous_range<Array>);
#endif // TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)
}
