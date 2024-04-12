//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// class cuda::std::ranges::subrange;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "test_iterators.h"
#include "test_macros.h"

#if TEST_STD_VER > 2017
template <size_t I, class S>
concept HasGet = requires { cuda::std::get<I>(cuda::std::declval<S>()); };
#else
template <size_t I, class S, class = void>
constexpr bool HasGet = false;

template <size_t I, class S>
constexpr bool HasGet<I, S, cuda::std::void_t<decltype(cuda::std::get<I>(cuda::std::declval<S>()))>> = true;
#endif

static_assert(HasGet<0, cuda::std::ranges::subrange<int*>>);
static_assert(HasGet<1, cuda::std::ranges::subrange<int*>>);
static_assert(!HasGet<2, cuda::std::ranges::subrange<int*>>);
static_assert(!HasGet<3, cuda::std::ranges::subrange<int*>>);

__host__ __device__ constexpr bool test()
{
  {
    using It   = int*;
    using Sent = sentinel_wrapper<int*>;
    int a[]    = {1, 2, 3};
    using R    = cuda::std::ranges::subrange<It, Sent, cuda::std::ranges::subrange_kind::unsized>;
    R r        = R(It(a), Sent(It(a + 3)));
    ASSERT_SAME_TYPE(decltype(cuda::std::get<0>(r)), It);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<1>(r)), Sent);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<0>(static_cast<R&&>(r))), It);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<1>(static_cast<R&&>(r))), Sent);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<0>(static_cast<const R&>(r))), It);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<1>(static_cast<const R&>(r))), Sent);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<0>(static_cast<const R&&>(r))), It);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<1>(static_cast<const R&&>(r))), Sent);
    assert(base(cuda::std::get<0>(r)) == a); // copy from It
    assert(base(base(cuda::std::get<1>(r))) == a + 3); // copy from Sent
    assert(base(cuda::std::get<0>(cuda::std::move(r))) == a); // copy from It
    assert(base(base(cuda::std::get<1>(cuda::std::move(r)))) == a + 3); // copy from Sent
  }
  {
    using It   = int*;
    using Sent = sentinel_wrapper<int*>;
    int a[]    = {1, 2, 3};
    using R    = cuda::std::ranges::subrange<It, Sent, cuda::std::ranges::subrange_kind::sized>;
    R r        = R(It(a), Sent(It(a + 3)), 3);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<0>(r)), It);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<1>(r)), Sent);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<0>(static_cast<R&&>(r))), It);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<1>(static_cast<R&&>(r))), Sent);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<0>(static_cast<const R&>(r))), It);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<1>(static_cast<const R&>(r))), Sent);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<0>(static_cast<const R&&>(r))), It);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<1>(static_cast<const R&&>(r))), Sent);
    assert(base(cuda::std::get<0>(r)) == a); // copy from It
    assert(base(base(cuda::std::get<1>(r))) == a + 3); // copy from Sent
    assert(base(cuda::std::get<0>(cuda::std::move(r))) == a); // copy from It
    assert(base(base(cuda::std::get<1>(cuda::std::move(r)))) == a + 3); // copy from Sent
  }
  {
    // Test the fix for LWG 3589.
    using It   = cpp20_input_iterator<int*>;
    using Sent = sentinel_wrapper<It>;
    int a[]    = {1, 2, 3};
    using R    = cuda::std::ranges::subrange<It, Sent>;
    R r        = R(It(a), Sent(It(a + 3)));
    static_assert(!HasGet<0, R&>);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<1>(r)), Sent);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<0>(static_cast<R&&>(r))), It);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<1>(static_cast<R&&>(r))), Sent);
    static_assert(!HasGet<0, const R&>);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<1>(static_cast<const R&>(r))), Sent);
    static_assert(!HasGet<0, const R&&>);
    ASSERT_SAME_TYPE(decltype(cuda::std::get<1>(static_cast<const R&&>(r))), Sent);
    assert(base(base(cuda::std::get<1>(r))) == a + 3); // copy from Sent
    assert(base(cuda::std::get<0>(cuda::std::move(r))) == a); // move from It
    assert(base(base(cuda::std::get<1>(cuda::std::move(r)))) == a + 3); // copy from Sent
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
