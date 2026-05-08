//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template<class... TTypes, class... UTypes>
//   struct common_type<tuple<TTypes...>, tuple<UTypes...>>;

#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct no_common_type
{};

template <class... Args>
TEST_FUNC constexpr auto has_common_type_imp(int)
  -> decltype(sizeof(typename cuda::std::common_type<Args...>::type), bool{})
{
  return true;
}

template <class... Args>
TEST_FUNC constexpr bool has_common_type_imp(long)
{
  return false;
}

template <class... Args>
static constexpr bool has_common_type = cuda::std::integral_constant<bool, has_common_type_imp<Args...>(0)>::value;

TEST_FUNC constexpr bool test()
{
  {
    using T = cuda::std::tuple<int>;
    static_assert(has_common_type<T, T>);
    static_assert(cuda::std::same_as<cuda::std::common_type_t<T, T>, T>);
  }
  {
    using T        = cuda::std::tuple<const int, short, const long&>;
    using U        = cuda::std::tuple<int, long, volatile long&>;
    using Expected = cuda::std::tuple<int, long, long>;
    static_assert(has_common_type<T, U>);
    static_assert(cuda::std::same_as<cuda::std::common_type_t<T, U>, Expected>);
  }
  {
    using T = cuda::std::tuple<int>;
    using U = cuda::std::tuple<int, int>;
    static_assert(!has_common_type<T, U>);
  }
  {
    using T = cuda::std::tuple<int, no_common_type>;
    using U = cuda::std::tuple<int, int>;
    static_assert(!has_common_type<T, U>);
  }

  return true;
}

int main(int, char**)
{
  static_assert(test());
  return 0;
}
