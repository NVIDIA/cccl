//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// The forward declarations of std::array and of its get overloads match the debug mode of libstdc++, which provides
// them through the inline namespace std::__debug before GCC 11.

// ADDITIONAL_COMPILE_DEFINITIONS: _GLIBCXX_DEBUG

#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#if _CCCL_HAS_HOST_STD_LIB()
#  include <array>
#endif // _CCCL_HAS_HOST_STD_LIB()

#include "test_macros.h"

#if _CCCL_HAS_HOST_STD_LIB()
static_assert(cuda::std::tuple_size<std::array<int, 3>>::value == 3);

void test()
{
  std::array<int, 3> arr{1, 2, 3};
  const auto& const_arr = arr;
  assert(std::get<0>(arr) == 1);
  assert(std::get<1>(const_arr) == 2);
  assert(cuda::std::get<2>(arr) == 3);
  assert(std::get<0>(cuda::std::move(arr)) == 1);
}
#endif // _CCCL_HAS_HOST_STD_LIB()

int main(int, char**)
{
#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, (test();))
#endif // _CCCL_HAS_HOST_STD_LIB()
  return 0;
}
