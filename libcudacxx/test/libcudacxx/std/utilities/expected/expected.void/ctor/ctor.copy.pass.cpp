//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// constexpr expected(const expected& rhs);
//
// Effects: If rhs.has_value() is false, direct-non-list-initializes unex with rhs.error().
//
// Postconditions: rhs.has_value() == this->has_value().
//
// Throws: Any exception thrown by the initialization of unex.
//
// Remarks:
// - This constructor is defined as deleted unless is_copy_constructible_v<E> is true.
// - This constructor is trivial if is_trivially_copy_constructible_v<E> is true.

#include <cuda/std/cassert>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

struct NonCopyable
{
  NonCopyable(const NonCopyable&) = delete;
};

struct CopyableNonTrivial
{
  int i;
  __host__ __device__ constexpr CopyableNonTrivial(int ii)
      : i(ii)
  {}
  __host__ __device__ constexpr CopyableNonTrivial(const CopyableNonTrivial& o)
      : i(o.i)
  {}
#if TEST_STD_VER > 2017
  __host__ __device__ friend constexpr bool operator==(const CopyableNonTrivial&, const CopyableNonTrivial&) = default;
#else
  __host__ __device__ friend constexpr bool
  operator==(const CopyableNonTrivial& lhs, const CopyableNonTrivial& rhs) noexcept
  {
    return lhs.i == rhs.i;
  }
  __host__ __device__ friend constexpr bool
  operator!=(const CopyableNonTrivial& lhs, const CopyableNonTrivial& rhs) noexcept
  {
    return lhs.i != rhs.i;
  }
#endif // TEST_STD_VER > 2017
};

// Test: This constructor is defined as deleted unless is_copy_constructible_v<E> is true.
static_assert(cuda::std::is_copy_constructible_v<cuda::std::expected<void, int>>, "");
static_assert(cuda::std::is_copy_constructible_v<cuda::std::expected<void, CopyableNonTrivial>>, "");
static_assert(!cuda::std::is_copy_constructible_v<cuda::std::expected<void, NonCopyable>>, "");

// Test: This constructor is trivial if is_trivially_copy_constructible_v<E> is true.
static_assert(cuda::std::is_trivially_copy_constructible_v<cuda::std::expected<void, int>>, "");
static_assert(!cuda::std::is_trivially_copy_constructible_v<cuda::std::expected<void, CopyableNonTrivial>>, "");

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  // copy the error non-trivial
  {
    const cuda::std::expected<void, CopyableNonTrivial> e1(cuda::std::unexpect, 5);
    auto e2 = e1;
    assert(!e2.has_value());
    assert(e2.error().i == 5);
  }

  // copy the error trivial
  {
    const cuda::std::expected<void, int> e1(cuda::std::unexpect, 5);
    auto e2 = e1;
    assert(!e2.has_value());
    assert(e2.error() == 5);
  }
  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
void test_exceptions()
{
  struct Except
  {};

  struct Throwing
  {
    Throwing() = default;
    __host__ __device__ Throwing(const Throwing&)
    {
      throw Except{};
    }
  };

  // throw on copying error
  {
    const cuda::std::expected<void, Throwing> e1(cuda::std::unexpect);
    try
    {
      auto e2 = e1;
      unused(e2);
      assert(false);
    }
    catch (Except)
    {}
  }
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS
  return 0;
}
