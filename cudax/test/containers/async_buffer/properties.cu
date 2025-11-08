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

#if _CCCL_CTK_AT_LEAST(12, 6)
using test_types = c2h::type_list<cuda::std::tuple<int, cuda::mr::host_accessible>,
                                  cuda::std::tuple<int, cuda::mr::device_accessible>,
                                  cuda::std::tuple<int, cuda::mr::host_accessible, cuda::mr::device_accessible>>;
#else // ^^^ _CCCL_CTK_AT_LEAST(12, 6) ^^^ / vvv _CCCL_CTK_BELOW(12, 6) vvv
using test_types = c2h::type_list<cuda::std::tuple<int, cuda::mr::device_accessible>>;
#endif // ^^^ _CCCL_CTK_BELOW(12, 6) ^^^

C2H_CCCLRT_TEST("cudax::buffer properties", "[container][buffer]", test_types)
{
  using TestT                  = c2h::get<0, TestType>;
  using Buffer                 = typename extract_properties<TestT>::buffer;
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
