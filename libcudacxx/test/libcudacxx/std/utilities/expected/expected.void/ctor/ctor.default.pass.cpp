//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// constexpr expected() noexcept;

#include <cuda/std/cassert>
#include <cuda/std/expected>
#include <cuda/std/type_traits>

#include "test_macros.h"

// Test noexcept

struct NoDefaultCtor
{
  __host__ __device__ constexpr NoDefaultCtor() = delete;
};

static_assert(cuda::std::is_nothrow_default_constructible_v<cuda::std::expected<void, int>>, "");
static_assert(cuda::std::is_nothrow_default_constructible_v<cuda::std::expected<void, NoDefaultCtor>>, "");

struct MyInt
{
  int i;
#if TEST_STD_VER > 2017
  __host__ __device__ friend constexpr bool operator==(const MyInt&, const MyInt&) = default;
#else
  __host__ __device__ friend constexpr bool operator==(const MyInt& lhs, const MyInt& rhs) noexcept
  {
    return lhs.i == rhs.i;
  }
  __host__ __device__ friend constexpr bool operator!=(const MyInt& lhs, const MyInt& rhs) noexcept
  {
    return lhs.i != rhs.i;
  }
#endif // TEST_STD_VER > 2017
};

__host__ __device__ constexpr bool test()
{
  // default constructible
  {
    cuda::std::expected<void, int> e;
    assert(e.has_value());
  }

  // non-default constructible
  {
    cuda::std::expected<void, NoDefaultCtor> e;
    assert(e.has_value());
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test(), "");
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  return 0;
}
