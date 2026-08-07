//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// todo(dabayer): nvrtc doesn't support non-trivial types as static data members without -default-device, fails with:
//   A class static data member with non-const type is considered a host variable, and host variables are not allowed in
//   JIT mode. Consider using -default-device flag to process such data members as __device__ variables in JIT mode
// UNSUPPORTED: nvrtc

// This test crashes msvc with message:
//   Internal compiler error. Try simplifying or changing the program near the locations listed above.
// UNSUPPORTED: msvc

// NTTP may not have a class type in C++17.
// REQUIRES: !c++17

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

template <class L, class R>
concept HasPtrToMem = requires(L l, R r) {
  { l->*r };
};

template <class L, class R>
concept HasNoexceptPtrToMem = requires(L l, R r) {
  { l->*r } noexcept;
};

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
    cuda::std::same_as<cuda::std::__constant_wrapper<42>> decltype(auto) result1 = cwS->*cwPM;
    static_assert(result1 == 42);
  }

  {
    // todo(dabayer): Try to make this work with nvcc
#if !_CCCL_CUDA_COMPILER(NVCC)
    // mix runtime and constant_wrapper parameters, will use built-in operator
    cuda::std::__constant_wrapper<(&s_value)> cwS;
    int S::* pm                                           = &S::member;
    cuda::std::same_as<const int&> decltype(auto) result1 = cwS->*pm;
    assert(result1 == 42);
#endif // !_CCCL_CUDA_COMPILER(NVCC)
  }

  {
    // custom operator->*
    cuda::std::__constant_wrapper<WithOps{42}> cwWO;
    cuda::std::__constant_wrapper<&WithOps::value> cwPM;
    cuda::std::same_as<cuda::std::__constant_wrapper<84>> decltype(auto) result1 = cwWO->*cwPM;
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

  {
    // integral_constant
    cuda::std::__constant_wrapper<(&s_value)> cwS;
    cuda::std::integral_constant<int S::*, &S::member> icPM;
    cuda::std::same_as<cuda::std::__constant_wrapper<42>> decltype(auto) result1 = cwS->*icPM;
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
