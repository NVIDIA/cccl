//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11

// Older Clangs do not support the C++20 feature to constrain destructors

// constexpr void emplace() noexcept;
//
// Effects: If has_value() is false, destroys unex and sets has_val to true.

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../../types.h"
#include "test_macros.h"

template <class T, class = void>
constexpr bool EmplaceNoexcept = false;

template <class T>
constexpr bool EmplaceNoexcept<T, cuda::std::void_t<decltype(cuda::std::declval<T>().emplace())>> =
  noexcept(cuda::std::declval<T>().emplace());

static_assert(!EmplaceNoexcept<int>, "");

static_assert(EmplaceNoexcept<cuda::std::expected<void, int>>, "");

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  // has_value
  {
    cuda::std::expected<void, int> e;
    e.emplace();
    assert(e.has_value());
  }

  // !has_value
  {
    Traced::state state{};
    cuda::std::expected<int, Traced> e(cuda::std::unexpect, state, 5);
    e.emplace();

    assert(state.dtorCalled);
    assert(e.has_value());
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // TEST_STD_VER > 2017 && defined(_CCCL_BUILTIN_ADDRESSOF)
  return 0;
}
