//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// constexpr void value() const &;
// constexpr void value() &&;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  // const &
  {
    const cuda::std::expected<void, int> e;
    e.value();
    static_assert(cuda::std::is_same_v<decltype(e.value()), void>, "");
  }

  // &
  {
    cuda::std::expected<void, int> e;
    e.value();
    static_assert(cuda::std::is_same_v<decltype(e.value()), void>, "");
  }

  // &&
  {
    cuda::std::expected<void, int> e;
    cuda::std::move(e).value();
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(e).value()), void>, "");
  }

  // const &&
  {
    const cuda::std::expected<void, int> e;
    cuda::std::move(e).value();
    static_assert(cuda::std::is_same_v<decltype(cuda::std::move(e).value()), void>, "");
  }

  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
void test_exceptions()
{
  // Test const& overload
  try
  {
    const cuda::std::expected<void, int> e(cuda::std::unexpect, 5);
    e.value();
    assert(false);
  }
  catch (const cuda::std::bad_expected_access<int>& ex)
  {
    assert(ex.error() == 5);
  }
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
  test();
#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  static_assert(test(), "");
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS
  return 0;
}
