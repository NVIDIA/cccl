//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/__functional/lazy_call_or.h>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct plus_one
{
  TEST_FUNC constexpr int operator()(int value) const noexcept
  {
    return value + 1;
  }
};

struct throwing_plus_one
{
  TEST_FUNC constexpr int operator()(int value) const noexcept(false)
  {
    return value + 1;
  }
};

struct fallback_long
{
  TEST_FUNC constexpr long operator()() const noexcept
  {
    return 13L;
  }
};

struct throwing_fallback
{
  TEST_FUNC constexpr long operator()() const noexcept(false)
  {
    return 13L;
  }
};

struct fallback_value_category
{
  TEST_FUNC constexpr int operator()() & noexcept
  {
    return 41;
  }

  TEST_FUNC constexpr int operator()() && noexcept
  {
    return 42;
  }
};

struct counting_fallback
{
  int* calls_;

  TEST_FUNC constexpr int operator()() const noexcept
  {
    ++*calls_;
    return 13;
  }
};

struct arg
{
  int value_;
};

struct value_category_fn
{
  TEST_FUNC constexpr int operator()(arg& value) const noexcept
  {
    return value.value_;
  }

  TEST_FUNC constexpr int operator()(arg&& value) const noexcept
  {
    return value.value_ + 1;
  }
};

TEST_FUNC constexpr void test_call_result()
{
  static_assert(cuda::std::is_same_v<decltype(cuda::__lazy_call_or(plus_one{}, fallback_long{}, 41)), int>);
  static_assert(cuda::std::is_same_v<decltype(cuda::__lazy_call_or(cuda::std::ignore, fallback_long{}, 41)), long>);

  static_assert(cuda::std::is_same_v<cuda::__lazy_call_result_or_t<plus_one, fallback_long, int>, int>);
  static_assert(cuda::std::is_same_v<cuda::__lazy_call_result_or_t<cuda::std::__ignore_t, fallback_long, int>, long>);
}

TEST_FUNC constexpr void test_noexcept()
{
  static_assert(noexcept(cuda::__lazy_call_or(plus_one{}, throwing_fallback{}, 41)));
  static_assert(noexcept(cuda::__lazy_call_or(cuda::std::ignore, fallback_long{}, 41)));
  // MSVC 19.39 and GCC 9 and lower compute the wrong answer for these
#if (TEST_COMPILER(MSVC, <=, 19, 39) || TEST_COMPILER(GCC, <=, 9)) && (TEST_STD_VER <= 2017)
#  define WRONG_ANSWER
#endif // (msvc-19.39- || gcc-9-) && cpp17

#ifndef WRONG_ANSWER
  static_assert(!noexcept(cuda::__lazy_call_or(throwing_plus_one{}, fallback_long{}, 41)));
  static_assert(!noexcept(cuda::__lazy_call_or(cuda::std::ignore, throwing_fallback{}, 41)));
#endif // !WRONG_ANSWER
}

TEST_FUNC constexpr void test_constexpr()
{
  static_assert(cuda::__lazy_call_or(plus_one{}, fallback_long{}, 41) == 42);
  static_assert(cuda::__lazy_call_or(cuda::std::ignore, fallback_long{}, 41) == 13L);
  // lvalue arg forwarding
  {
    fallback_value_category fallback{};
    static_assert(cuda::__lazy_call_or(cuda::std::ignore, fallback, 0) == 41);
    arg value{41};
    // must be assert() because value is not constexpr
    assert(cuda::__lazy_call_or(value_category_fn{}, fallback_long{}, value) == 41);
    assert(cuda::__lazy_call_or(value_category_fn{}, fallback_long{}, ::cuda::std::move(value)) == 42);
  }
  static_assert(cuda::__lazy_call_or(value_category_fn{}, fallback_long{}, arg{41}) == 42);
  static_assert(cuda::__lazy_call_or(cuda::std::ignore, fallback_value_category{}, 0) == 42);
}

TEST_FUNC constexpr void test_lazy_evaluation()
{
  int fallback_calls = 0;
  counting_fallback fallback{&fallback_calls};

  static_assert(cuda::__lazy_call_or(plus_one{}, fallback, 41) == 42);
  assert(fallback_calls == 0);

  assert(cuda::__lazy_call_or(cuda::std::ignore, fallback, 41) == 13);
  assert(fallback_calls == 1);
}

TEST_FUNC constexpr bool test()
{
  test_call_result();
  test_noexcept();
  test_constexpr();
  test_lazy_evaluation();
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
