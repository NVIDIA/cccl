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

// template<class... Args>
//   constexpr T& emplace(Args&&... args) noexcept;
// Constraints: is_nothrow_constructible_v<T, Args...> is true.
//
// Effects: Equivalent to:
// if (has_value()) {
//   destroy_at(addressof(val));
// } else {
//   destroy_at(addressof(unex));
//   has_val = true;
// }
// return *construct_at(addressof(val), cuda::std::forward<Args>(args)...);

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/expected>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../../types.h"
#include "test_macros.h"

template <class T, class... Args>
_LIBCUDACXX_CONCEPT_FRAGMENT(CanEmplace_, requires(T t, Args&&... args)((t.emplace(cuda::std::forward<Args>(args)...))));
template <class T, class... Args>
constexpr bool CanEmplace = _LIBCUDACXX_FRAGMENT(CanEmplace_, T, Args...);

static_assert(CanEmplace<cuda::std::expected<int, int>, int>, "");

template <bool Noexcept>
struct CtorFromInt
{
  __host__ __device__ CtorFromInt(int) noexcept(Noexcept);
  __host__ __device__ CtorFromInt(int, int) noexcept(Noexcept);
};

static_assert(CanEmplace<cuda::std::expected<CtorFromInt<true>, int>, int>, "");
static_assert(CanEmplace<cuda::std::expected<CtorFromInt<true>, int>, int, int>, "");
#ifndef TEST_COMPILER_ICC
static_assert(!CanEmplace<cuda::std::expected<CtorFromInt<false>, int>, int>, "");
static_assert(!CanEmplace<cuda::std::expected<CtorFromInt<false>, int>, int, int>, "");
#endif // TEST_COMPILER_ICC

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  // has_value
  {
    BothNoexcept::state oldState{};
    BothNoexcept::state newState{};
    cuda::std::expected<BothNoexcept, int> e(cuda::std::in_place, oldState, 5);
    decltype(auto) x = e.emplace(newState, 10);
    static_assert(cuda::std::same_as<decltype(x), BothNoexcept&>, "");
    assert(&x == &(*e));

    assert(oldState.dtorCalled);
    assert(e.has_value());
    assert(e.value().data_ == 10);
  }

  // !has_value
  {
    BothMayThrow::state oldState{};
    cuda::std::expected<int, BothMayThrow> e(cuda::std::unexpect, oldState, 5);
    decltype(auto) x = e.emplace(10);
    static_assert(cuda::std::same_as<decltype(x), int&>, "");
    assert(&x == &(*e));

    assert(oldState.dtorCalled);
    assert(e.has_value());
    assert(e.value() == 10);
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
