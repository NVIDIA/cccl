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

// template<class... TTypes, class... UTypes, template<class> class TQual,
//          template<class> class UQual>
//   struct basic_common_reference<tuple<TTypes...>, tuple<UTypes...>, TQual, UQual>;

#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct no_common_reference
{};

template <class T>
struct X
{};

template <class T>
struct Y
{};

struct common_ref
{};

template <template <class> class TQual, template <class> class UQual>
struct cuda::std::basic_common_reference<X<int>, Y<int>, TQual, UQual>
{
  using type = common_ref;
};

template <template <class> class TQual, template <class> class UQual>
struct cuda::std::basic_common_reference<Y<int>, X<int>, TQual, UQual>
{
  using type = common_ref;
};

template <class... Args>
TEST_FUNC constexpr auto has_common_reference_imp(int)
  -> decltype(sizeof(typename cuda::std::common_reference<Args...>::type), bool{})
{
  return true;
}

template <class... Args>
TEST_FUNC constexpr bool has_common_reference_imp(long)
{
  return false;
}

template <class... Args>
inline constexpr bool has_common_reference =
  cuda::std::integral_constant<bool, has_common_reference_imp<Args...>(0)>::value;

TEST_FUNC constexpr bool test()
{
  {
    using T = cuda::std::tuple<int>;
    static_assert(cuda::std::same_as<cuda::std::common_reference_t<T, T>, T>);
  }
  {
    using T        = cuda::std::tuple<int&, const long&>;
    using U        = cuda::std::tuple<const int&, long&>;
    using Expected = cuda::std::tuple<const int&, const long&>;
    static_assert(has_common_reference<T, U>);
    static_assert(cuda::std::same_as<cuda::std::common_reference_t<T, U>, Expected>);
  }
  {
    using T        = cuda::std::tuple<X<int>>;
    using U        = cuda::std::tuple<Y<int>>;
    using Expected = cuda::std::tuple<common_ref>;
    static_assert(has_common_reference<T, U>);
    static_assert(cuda::std::same_as<cuda::std::common_reference_t<T, U>, Expected>);
  }
  {
    using T = cuda::std::tuple<X<double>>;
    using U = cuda::std::tuple<Y<double>>;
    // Only X<int> and Y<int> are specialized
    static_assert(!has_common_reference<T, U>);
  }
  {
    using T = cuda::std::tuple<int>;
    using U = cuda::std::tuple<int, int>;
    static_assert(!has_common_reference<T, U>);
  }
  {
    using T = cuda::std::tuple<int, no_common_reference>;
    using U = cuda::std::tuple<int, int>;
    static_assert(!has_common_reference<T, U>);
  }

  return true;
}

int main(int, char**)
{
  static_assert(test());
  return 0;
}
