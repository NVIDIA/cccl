//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// test forward_like

#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

struct U
{}; // class type so const-qualification is not stripped from a prvalue
using CU = const U;
using T  = int;
using CT = const T;

U u{};
const U& cu = u;

static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<T>(U{})), U&&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<T>(CU{})), CU&&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<T>(u)), U&&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<T>(cu)), CU&&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<T>(cuda::std::move(u))), U&&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<T>(cuda::std::move(cu))), CU&&>);

static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<CT>(U{})), CU&&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<CT>(CU{})), CU&&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<CT>(u)), CU&&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<CT>(cu)), CU&&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<CT>(cuda::std::move(u))), CU&&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<CT>(cuda::std::move(cu))), CU&&>);

static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<T&>(U{})), U&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<T&>(CU{})), CU&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<T&>(u)), U&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<T&>(cu)), CU&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<T&>(cuda::std::move(u))), U&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<T&>(cuda::std::move(cu))), CU&>);

static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<CT&>(U{})), CU&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<CT&>(CU{})), CU&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<CT&>(u)), CU&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<CT&>(cu)), CU&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<CT&>(cuda::std::move(u))), CU&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<CT&>(cuda::std::move(cu))), CU&>);

static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<T&&>(U{})), U&&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<T&&>(CU{})), CU&&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<T&&>(u)), U&&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<T&&>(cu)), CU&&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<T&&>(cuda::std::move(u))), U&&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<T&&>(cuda::std::move(cu))), CU&&>);

static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<CT&&>(U{})), CU&&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<CT&&>(CU{})), CU&&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<CT&&>(u)), CU&&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<CT&&>(cu)), CU&&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<CT&&>(cuda::std::move(u))), CU&&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<CT&&>(cuda::std::move(cu))), CU&&>);

static_assert(noexcept(cuda::std::forward_like<T>(u)));

static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<U&>(u)), U&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<CU&>(cu)), CU&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<U&&>(cuda::std::move(u))), U&&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<CU&&>(cuda::std::move(cu))), CU&&>);

struct NoCtorCopyMove
{
  NoCtorCopyMove()                      = delete;
  NoCtorCopyMove(const NoCtorCopyMove&) = delete;
  NoCtorCopyMove(NoCtorCopyMove&&)      = delete;
};

static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<CT&&>(cuda::std::declval<NoCtorCopyMove>())),
                                   const NoCtorCopyMove&&>);
static_assert(cuda::std::is_same_v<decltype(cuda::std::forward_like<CT&>(cuda::std::declval<NoCtorCopyMove>())),
                                   const NoCtorCopyMove&>);
static_assert(
  cuda::std::is_same_v<decltype(cuda::std::forward_like<T&&>(cuda::std::declval<NoCtorCopyMove>())), NoCtorCopyMove&&>);
static_assert(
  cuda::std::is_same_v<decltype(cuda::std::forward_like<T&>(cuda::std::declval<NoCtorCopyMove>())), NoCtorCopyMove&>);

static_assert(noexcept(cuda::std::forward_like<T>(cuda::std::declval<NoCtorCopyMove>())));

__host__ __device__ constexpr bool test()
{
  {
    int val       = 1729;
    auto&& result = cuda::std::forward_like<const double&>(val);
    static_assert(cuda::std::is_same_v<decltype(result), const int&>);
    assert(&result == &val);
  }
  {
    int val       = 1729;
    auto&& result = cuda::std::forward_like<double&>(val);
    static_assert(cuda::std::is_same_v<decltype(result), int&>);
    assert(&result == &val);
  }
  {
    int val       = 1729;
    auto&& result = cuda::std::forward_like<const double&&>(val);
    static_assert(cuda::std::is_same_v<decltype(result), const int&&>);
    assert(&result == &val);
  }
  {
    int val       = 1729;
    auto&& result = cuda::std::forward_like<double&&>(val);
    static_assert(cuda::std::is_same_v<decltype(result), int&&>);
    assert(&result == &val);
  }
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
