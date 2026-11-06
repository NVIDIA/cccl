//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<class T>
//   const T&
//   clamp(const T& v, const T& lo, const T& hi);

// clang-format off
#include <disable_nvfp_conversions_and_operators.h>
// clang-format on

#include <cuda/std/__type_traits/is_extended_arithmetic.h>
#include <cuda/std/algorithm>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "cuda_fp_types.h"
#include "test_macros.h"

struct Tag
{
  TEST_FUNC constexpr Tag()
      : val(0)
      , tag("Default")
  {}
  TEST_FUNC constexpr Tag(int a, const char* b)
      : val(a)
      , tag(b)
  {}

  int val;
  const char* tag;
};

TEST_FUNC constexpr bool eq(const Tag& rhs, const Tag& lhs)
{
  return rhs.val == lhs.val && rhs.tag == lhs.tag;
}
TEST_FUNC constexpr bool operator<(const Tag& rhs, const Tag& lhs)
{
  return rhs.val < lhs.val;
}

template <class T>
TEST_FUNC constexpr void test(const T& a, const T& lo, const T& hi, const T& x)
{
  if constexpr (cuda::std::is_arithmetic_v<T>)
  {
    assert(cuda::std::clamp(a, lo, hi) == x);
  }
  else if constexpr (cuda::std::__is_extended_arithmetic_v<T>)
  {
    assert(::__heq(cuda::std::clamp(a, lo, hi), x));
  }
  else
  {
    assert(&cuda::std::clamp(a, lo, hi) == &x);
  }
}

TEST_FUNC constexpr bool test()
{
  {
    int x = 0;
    int y = 0;
    int z = 0;
    test(x, y, z, x);
    test(y, x, z, y);
  }
  {
    int x = 0;
    int y = 1;
    int z = 2;
    test(x, y, z, y);
    test(y, x, z, y);
  }
  {
    int x = 1;
    int y = 0;
    int z = 1;
    test(x, y, z, x);
    test(y, x, z, x);
  }

  {
    //  If they're all the same, we should get the value back.
    Tag x{0, "Zero-x"};
    Tag y{0, "Zero-y"};
    Tag z{0, "Zero-z"};
    assert(eq(cuda::std::clamp(x, y, z), x));
    assert(eq(cuda::std::clamp(y, x, z), y));
  }

  {
    //  If it's the same as the lower bound, we get the value back.
    Tag x{0, "Zero-x"};
    Tag y{0, "Zero-y"};
    Tag z{1, "One-z"};
    assert(eq(cuda::std::clamp(x, y, z), x));
    assert(eq(cuda::std::clamp(y, x, z), y));
  }

  {
    //  If it's the same as the upper bound, we get the value back.
    Tag x{1, "One-x"};
    Tag y{0, "Zero-y"};
    Tag z{1, "One-z"};
    assert(eq(cuda::std::clamp(x, y, z), x));
    assert(eq(cuda::std::clamp(z, y, x), z));
  }

  {
    //  If the value is between, we should get the value back
    Tag x{1, "One-x"};
    Tag y{0, "Zero-y"};
    Tag z{2, "Two-z"};
    assert(eq(cuda::std::clamp(x, y, z), x));
    assert(eq(cuda::std::clamp(y, x, z), x));
  }

  {
    //  If the value is less than the 'lo', we should get the lo back.
    Tag x{0, "Zero-x"};
    Tag y{1, "One-y"};
    Tag z{2, "Two-z"};
    assert(eq(cuda::std::clamp(x, y, z), y));
    assert(eq(cuda::std::clamp(y, x, z), y));
  }
  {
    //  If the value is greater than 'hi', we should get hi back.
    Tag x{2, "Two-x"};
    Tag y{0, "Zero-y"};
    Tag z{1, "One-z"};
    assert(eq(cuda::std::clamp(x, y, z), z));
    assert(eq(cuda::std::clamp(y, z, x), z));
  }

  return true;
}

TEST_FUNC void test_extended_arithmetic()
{
#if _CCCL_HAS_NVFP16()
  {
    __half x = ::__float2half(0.0f);
    __half y = ::__float2half(1.0f);
    __half z = ::__float2half(2.0f);
    test(x, y, z, y);
    test(y, x, z, y);
  }
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  {
    __nv_bfloat16 x = ::__float2bfloat16(0.0f);
    __nv_bfloat16 y = ::__float2bfloat16(1.0f);
    __nv_bfloat16 z = ::__float2bfloat16(2.0f);
    test(x, y, z, y);
    test(y, x, z, y);
  }
#endif // _CCCL_HAS_NVBF16()
}

int main(int, char**)
{
  test();
  static_assert(test());

  test_extended_arithmetic();

  return 0;
}
