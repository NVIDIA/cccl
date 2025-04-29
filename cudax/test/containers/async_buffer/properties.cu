//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
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

C2H_TEST("cudax::async_buffer properties",
         "[container][async_buffer]",
         c2h::type_list<cuda::std::tuple<cuda::mr::host_accessible>,
                        cuda::std::tuple<cuda::mr::device_accessible>,
                        cuda::std::tuple<cuda::mr::host_accessible, cuda::mr::device_accessible>>)
{
  using TestT                  = c2h::get<0, TestType>;
  using Buffer                 = typename extract_properties<TestT>::async_buffer;
  using iterator               = typename extract_properties<TestT>::iterator;
  using const_iterator         = typename extract_properties<TestT>::const_iterator;
  using reverse_iterator       = cuda::std::reverse_iterator<iterator>;
  using const_reverse_iterator = cuda::std::reverse_iterator<const_iterator>;

  // Check the type aliases
  static_assert(cuda::std::is_same_v<int, typename Buffer::value_type>, "");
  static_assert(cuda::std::is_same_v<cuda::std::size_t, typename Buffer::size_type>, "");
  static_assert(cuda::std::is_same_v<cuda::std::ptrdiff_t, typename Buffer::difference_type>, "");
  static_assert(cuda::std::is_same_v<int*, typename Buffer::pointer>, "");
  static_assert(cuda::std::is_same_v<const int*, typename Buffer::const_pointer>, "");
  static_assert(cuda::std::is_same_v<int&, typename Buffer::reference>, "");
  static_assert(cuda::std::is_same_v<const int&, typename Buffer::const_reference>, "");
  static_assert(cuda::std::is_same_v<iterator, typename Buffer::iterator>, "");
  static_assert(cuda::std::is_same_v<const_iterator, typename Buffer::const_iterator>, "");
  static_assert(cuda::std::is_same_v<cuda::std::reverse_iterator<iterator>, typename Buffer::reverse_iterator>, "");
  static_assert(
    cuda::std::is_same_v<cuda::std::reverse_iterator<const_iterator>, typename Buffer::const_reverse_iterator>, "");
  static_assert(cuda::std::ranges::contiguous_range<Buffer>);
}
