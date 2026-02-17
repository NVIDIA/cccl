//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/optional>

// template<class F> constexpr optional or_else(F&&) &&;
// template<class F> constexpr optional or_else(F&&) const&;

#include <cuda/std/cassert>
#include <cuda/std/optional>

#include "MoveOnly.h"

struct NonMovable
{
  NonMovable()             = default;
  NonMovable(NonMovable&&) = delete;
};

template <class Opt, class F>
_CCCL_CONCEPT has_or_else =
  _CCCL_REQUIRES_EXPR((Opt, F), Opt&& opt, F&& f)((cuda::std::forward<Opt>(opt).or_else(cuda::std::forward<F>(f))));

template <class T>
__host__ __device__ cuda::std::optional<T> return_optional();

static_assert(has_or_else<cuda::std::optional<int>&, decltype(return_optional<int>)>, "");
static_assert(has_or_else<cuda::std::optional<int>&&, decltype(return_optional<int>)>, "");
static_assert(!has_or_else<cuda::std::optional<MoveOnly>&, decltype(return_optional<MoveOnly>)>, "");
static_assert(has_or_else<cuda::std::optional<MoveOnly>&&, decltype(return_optional<MoveOnly>)>, "");
static_assert(!has_or_else<cuda::std::optional<NonMovable>&, decltype(return_optional<NonMovable>)>, "");
static_assert(!has_or_else<cuda::std::optional<NonMovable>&&, decltype(return_optional<NonMovable>)>, "");

__host__ __device__ cuda::std::optional<int> take_int(int);
__host__ __device__ void take_int_return_void(int);

// For some reason, MSVC handles the assertions above correctly in pre-C++20 modes, but... fails to fail these.
// And it's not because the tested expressions are valid; no, they fail to compile, but something in the sfinae
// machinery used by has_or_else just trips this quite special compiler up. Workaround, manually re-spell the same thing
// again. I don't understand why it's just these that fail with has_or_else, not any of the ones above - but MSVC's
// error messages are so monumentally unhelpful, that I decided to stop wasting time on this and just work around the
// cases that were giving it trouble.
#if TEST_COMPILER(MSVC)
template <class T, class F, class = void>
struct has_or_else_war : cuda::std::false_type
{};

template <class T, class F>
struct has_or_else_war<T, F, decltype(cuda::std::declval<T>().or_else(cuda::std::declval<F>()), void())>
    : cuda::std::true_type
{};

static_assert(!has_or_else_war<cuda::std::optional<int>&, decltype(take_int)>::value, "");
static_assert(!has_or_else_war<cuda::std::optional<int>&, decltype(take_int_return_void)>::value, "");
static_assert(!has_or_else_war<cuda::std::optional<int>&, int>::value, "");
#else
static_assert(!has_or_else<cuda::std::optional<int>&, decltype(take_int)>, "");
static_assert(!has_or_else<cuda::std::optional<int>&, decltype(take_int_return_void)>, "");
static_assert(!has_or_else<cuda::std::optional<int>&, int>, "");
#endif

__host__ __device__ constexpr bool test()
{
  {
    cuda::std::optional<int> opt{};
    assert(opt.or_else([] {
      return cuda::std::optional<int>{0};
    }) == 0);
    opt = 1;
    opt.or_else([] {
      assert(false);
      return cuda::std::optional<int>{};
    });
  }

  {
    int val = 42;
    cuda::std::optional<int&> opt{};
    assert(opt.or_else([&val] {
      return cuda::std::optional<int&>{val};
    }) == 42);
    opt = val;
    opt.or_else([] {
      assert(false);
      return cuda::std::optional<int&>{};
    });
  }

  return true;
}

__host__ __device__ constexpr bool test_nontrivial()
{
  {
    cuda::std::optional<MoveOnly> opt{};
    opt = cuda::std::move(opt).or_else([] {
      return cuda::std::optional<MoveOnly>{MoveOnly{}};
    });
    cuda::std::move(opt).or_else([] {
      assert(false);
      return cuda::std::optional<MoveOnly>{};
    });
  }

  return true;
}

int main(int, char**)
{
  test();
  test_nontrivial();

  // GCC <9 incorrectly trips on the assertions in this, so disable it there
#if !TEST_COMPILER(GCC, <, 10)
  static_assert(test(), "");
#endif // !TEST_COMPILER(GCC, <, 11)
#if TEST_STD_VER > 2017
#  if defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test_nontrivial());
#  endif // defined(_CCCL_BUILTIN_ADDRESSOF)
#endif // TEST_STD_VER > 2017

  return 0;
}
