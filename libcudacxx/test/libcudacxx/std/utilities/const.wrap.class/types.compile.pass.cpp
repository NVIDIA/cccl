//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// gcc-10 segfaults with any use of constant_wrapper, gcc-11 fails to evaluate:
//   typename decltype(__cw_fixed_value(_Xp))::type
// UNSUPPORTED: gcc-10 || gcc-11

// todo(dabayer): Find a way to make this work for nvrtc.
// nvrtc doesn't allow accessing the static constexpr const auto& value member.
// UNSUPPORTED: nvrtc

// REQUIRES: !c++17

// constant_wrapper

// static constexpr const auto & value = X.data;
// using type = constant_wrapper;
// using value_type = decltype(X)::type;

#include <cuda/std/algorithm>
#include <cuda/std/concepts>
#include <cuda/std/utility>

static_assert(cuda::std::__constant_wrapper<42>::value == 42);
static_assert(cuda::std::same_as<decltype(cuda::std::__constant_wrapper<42>::value), const int&>);
static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<42>::type, cuda::std::__constant_wrapper<42>>);
static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<42>::value_type, int>);

struct S
{
  int member = 42;
};

static_assert(cuda::std::__constant_wrapper<S{5}>::value.member == 5);
static_assert(cuda::std::same_as<decltype(cuda::std::__constant_wrapper<S{5}>::value), const S&>);
static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<S{5}>::type, cuda::std::__constant_wrapper<S{5}>>);
static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<S{5}>::value_type, S>);

// todo: Find out why this doesn't compile.
// static_assert(cuda::std::ranges::equal(cuda::std::__constant_wrapper<"abcd">::value, "abcd"));
static_assert(cuda::std::same_as<decltype(cuda::std::__constant_wrapper<"abcd">::value), const char (&)[5]>);
static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<"abcd">::type, cuda::std::__constant_wrapper<"abcd">>);
static_assert(cuda::std::same_as<cuda::std::__constant_wrapper<"abcd">::value_type, const char[5]>);

int main(int, char**)
{
  return 0;
}
