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

#include <cuda/experimental/vector.cuh>

#include "helper.h"
#include "types.h"
#include <catch2/catch.hpp>

TEMPLATE_TEST_CASE("cudax::vector properties",
                   "[container][vector]",
                   cuda::std::tuple<cuda::mr::host_accessible>,
                   cuda::std::tuple<cuda::mr::device_accessible>,
                   (cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>) )
{
  using Vector                 = typename extract_properties<TestType>::vector;
  using iterator               = typename extract_properties<TestType>::iterator;
  using const_iterator         = typename extract_properties<TestType>::const_iterator;
  using reverse_iterator       = cuda::std::reverse_iterator<iterator>;
  using const_reverse_iterator = cuda::std::reverse_iterator<const_iterator>;

  // Check the type aliases
  static_assert(cuda::std::is_same<int, typename Vector::value_type>::value, "");
  static_assert(cuda::std::is_same<cuda::std::size_t, typename Vector::size_type>::value, "");
  static_assert(cuda::std::is_same<cuda::std::ptrdiff_t, typename Vector::difference_type>::value, "");
  static_assert(cuda::std::is_same<int*, typename Vector::pointer>::value, "");
  static_assert(cuda::std::is_same<const int*, typename Vector::const_pointer>::value, "");
  static_assert(cuda::std::is_same<int&, typename Vector::reference>::value, "");
  static_assert(cuda::std::is_same<const int&, typename Vector::const_reference>::value, "");
  static_assert(cuda::std::is_same<iterator, typename Vector::iterator>::value, "");
  static_assert(cuda::std::is_same<const_iterator, typename Vector::const_iterator>::value, "");
  static_assert(cuda::std::is_same<cuda::std::reverse_iterator<iterator>, typename Vector::reverse_iterator>::value,
                "");
  static_assert(
    cuda::std::is_same<cuda::std::reverse_iterator<const_iterator>, typename Vector::const_reverse_iterator>::value,
    "");

#if TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)
  static_assert(cuda::std::ranges::contiguous_range<Vector>);
#endif // TEST_STD_VER >= 2017 && !defined(TEST_COMPILER_MSVC_2017)
}
