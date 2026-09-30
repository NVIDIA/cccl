//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/functional>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct Nothrow
{
  TEST_FUNC Nothrow() noexcept {}
};

struct NotDefaultable
{
  TEST_FUNC NotDefaultable() = delete;
  TEST_FUNC NotDefaultable(int) noexcept {}
};

struct MaybeThrowingDefault
{
  TEST_FUNC MaybeThrowingDefault() noexcept(false) {}
};

struct MaybeThrowingCopy
{
  TEST_FUNC MaybeThrowingCopy(const MaybeThrowingCopy&) noexcept(false) {}
};

struct MaybeThrowingMove
{
  TEST_FUNC MaybeThrowingMove(MaybeThrowingCopy&&) noexcept(false) {}
};

template <class Fn>
TEST_FUNC constexpr void test()
{
  using zip_function = cuda::zip_function<Fn>;
  static_assert(cuda::std::is_default_constructible_v<zip_function> == cuda::std::is_default_constructible_v<Fn>);
  static_assert(
    cuda::std::is_nothrow_default_constructible_v<zip_function> == cuda::std::is_nothrow_default_constructible_v<Fn>);

  static_assert(cuda::std::is_constructible_v<zip_function, Fn&&>);
  static_assert(cuda::std::is_constructible_v<zip_function, const Fn&>);
  static_assert(
    cuda::std::is_nothrow_constructible_v<zip_function, const Fn&> == cuda::std::is_nothrow_copy_constructible_v<Fn>);
  static_assert(
    cuda::std::is_nothrow_constructible_v<zip_function, Fn&&> == cuda::std::is_nothrow_move_constructible_v<Fn>);

  static_assert(cuda::std::is_copy_constructible_v<zip_function>);
  static_assert(cuda::std::is_move_constructible_v<zip_function>);
  static_assert(cuda::std::is_copy_assignable_v<zip_function>);
  static_assert(cuda::std::is_move_assignable_v<zip_function>);
}

TEST_FUNC constexpr void test()
{
  test<Nothrow>();
  test<NotDefaultable>();
  test<MaybeThrowingDefault>();
  test<MaybeThrowingCopy>();
  test<MaybeThrowingMove>();
}

int main(int, char**)
{
  return 0;
}
