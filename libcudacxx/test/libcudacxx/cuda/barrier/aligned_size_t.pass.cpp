//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: pre-sm-70

// <cuda/barrier>

#include <cuda/barrier>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
  using aligned_t = cuda::aligned_size_t<1>;
  static_assert(!cuda::std::is_default_constructible<aligned_t>::value, "");
  static_assert(aligned_t::align == 1, "");
  {
    const aligned_t aligned{42};
    assert(aligned.value == 42);
    assert(static_cast<cuda::std::size_t>(aligned) == 42);
  }
  return true;
}

// test C++11 differently
static_assert(cuda::aligned_size_t<42>{1337}.value == 1337, "");

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2014
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2014
  return 0;
}
