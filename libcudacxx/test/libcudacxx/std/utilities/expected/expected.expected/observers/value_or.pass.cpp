//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// template<class U> constexpr T value_or(U&& v) const &;
// template<class U> constexpr T value_or(U&& v) &&;

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "MoveOnly.h"
#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  // const &, has_value()
  {
    const cuda::std::expected<int, int> e(5);
    decltype(auto) x = e.value_or(10);
    static_assert(cuda::std::same_as<int, decltype(x)>, "");
    assert(x == 5);
  }

  // const &, !has_value()
  {
    const cuda::std::expected<int, int> e(cuda::std::unexpect, 5);
    decltype(auto) x = e.value_or(10);
    static_assert(cuda::std::same_as<int, decltype(x)>, "");
    assert(x == 10);
  }

  // &&, has_value()
  {
    cuda::std::expected<MoveOnly, int> e(cuda::std::in_place, 5);
    decltype(auto) x = cuda::std::move(e).value_or(10);
    static_assert(cuda::std::same_as<MoveOnly, decltype(x)>, "");
    assert(x == 5);
  }

  // &&, !has_value()
  {
    cuda::std::expected<MoveOnly, int> e(cuda::std::unexpect, 5);
    decltype(auto) x = cuda::std::move(e).value_or(10);
    static_assert(cuda::std::same_as<MoveOnly, decltype(x)>, "");
    assert(x == 10);
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
