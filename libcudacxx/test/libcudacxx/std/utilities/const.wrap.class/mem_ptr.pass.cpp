//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// nvcc + msvc + c++17 combination fails to compile with error:
//   template instantiation resulted in unexpected function type
// XFAIL: nvcc && msvc && c++17

// constant_wrapper

// template<constexpr-param L, constexpr-param R>
//   friend constexpr auto operator->*(L, R) noexcept -> constant_wrapper<L::value->*(R::value)>
//     { return {}; }

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/utility>

#include "helpers.h"
#include "test_macros.h"

struct S
{
  int member = 42;
};

constexpr S s_value{};

template <class L, class R, class = void>
inline constexpr bool HasPtrToMem = false;
template <class L, class R>
inline constexpr bool
  HasPtrToMem<L, R, cuda::std::void_t<decltype(cuda::std::declval<L&>()->*cuda::std::declval<R&>())>> = true;

template <class L, class R, class = void>
inline constexpr bool HasNoexceptPtrToMem = false;
template <class L, class R>
inline constexpr bool
  HasNoexceptPtrToMem<L, R, cuda::std::enable_if_t<noexcept(cuda::std::declval<L&>()->*cuda::std::declval<R&>())>> =
    true;

struct WithOps
{
  int value;
  TEST_FUNC constexpr WithOps(int v)
      : value(v)
  {}

  TEST_FUNC friend constexpr auto operator->*(WithOps w, int WithOps::* pm)
  {
    return w.value + (&w)->*pm;
  }
};

struct OpsReturnNonStructural
{
  int value;
  TEST_FUNC constexpr OpsReturnNonStructural(int v)
      : value(v)
  {}

  TEST_FUNC friend constexpr auto operator->*(OpsReturnNonStructural o, int OpsReturnNonStructural::* pm)
  {
    return NonStructural{o.value + (&o)->*pm};
  }
};

struct NoOps
{};

static_assert(HasPtrToMem<cuda::std::__constant_wrapper<&s_value>, cuda::std::__constant_wrapper<&S::member>>);
static_assert(HasNoexceptPtrToMem<cuda::std::__constant_wrapper<&s_value>, cuda::std::__constant_wrapper<&S::member>>);

static_assert(HasPtrToMem<cuda::std::__constant_wrapper<&s_value>, int S::*>);
static_assert(!HasPtrToMem<cuda::std::__constant_wrapper<&s_value>, int>);

TEST_FUNC constexpr bool test()
{
  {
    // use builtin operator->*
    cuda::std::__constant_wrapper<(&s_value)> cwS;
    cuda::std::__constant_wrapper<&S::member> cwPM;
    decltype(auto) result1 = cwS->*cwPM;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<42>, decltype(result1)>);
    static_assert(result1 == 42);
  }

  {
    // todo(dabayer): Try to make this work with nvcc
#if !TEST_CUDA_COMPILER(NVCC) && !TEST_COMPILER(NVRTC)
    // mix runtime and constant_wrapper parameters, will use built-in operator
    cuda::std::__constant_wrapper<(&s_value)> cwS;
    int S::* pm                                           = &S::member;
    cuda::std::same_as<const int&> decltype(auto) result1 = cwS->*pm;
    assert(result1 == 42);
#endif // !TEST_CUDA_COMPILER(NVCC) && !TEST_COMPILER(NVRTC)
  }

#if TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)
  {
    // custom operator->*
    cuda::std::__constant_wrapper<WithOps{42}> cwWO;
    cuda::std::__constant_wrapper<&WithOps::value> cwPM;
    decltype(auto) result1 = cwWO->*cwPM;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<84>, decltype(result1)>);
    static_assert(result1 == 84);
  }

  {
    // Return non-structural type
    // Will use underlying type's runtime operators
    cuda::std::__constant_wrapper<OpsReturnNonStructural{42}> cwORNS;
    cuda::std::__constant_wrapper<&OpsReturnNonStructural::value> cwPM;
    cuda::std::same_as<NonStructural> decltype(auto) result1 = cwORNS->*cwPM;
    assert(result1.get() == 84);
  }
#endif // TEST_STD_VER >= 2020 && !TEST_COMPILER(NVRTC)

  {
    // integral_constant
    cuda::std::__constant_wrapper<(&s_value)> cwS;
    cuda::std::integral_constant<int S::*, &S::member> icPM;
    decltype(auto) result1 = cwS->*icPM;
    static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<42>, decltype(result1)>);
    static_assert(result1 == 42);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
