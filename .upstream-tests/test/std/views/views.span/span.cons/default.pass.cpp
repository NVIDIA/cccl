//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11

// <span>

// constexpr span() noexcept;

#include <cuda/std/span>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__
void checkCV()
{
//  Types the same (dynamic sized)
    {
    cuda::std::span<               int> s1;
    cuda::std::span<const          int> s2;
    cuda::std::span<      volatile int> s3;
    cuda::std::span<const volatile int> s4;
    assert(s1.size() + s2.size() + s3.size() + s4.size() == 0);
    }

//  Types the same (static sized)
    {
    cuda::std::span<               int,0> s1;
    cuda::std::span<const          int,0> s2;
    cuda::std::span<      volatile int,0> s3;
    cuda::std::span<const volatile int,0> s4;
    assert(s1.size() + s2.size() + s3.size() + s4.size() == 0);
    }
}


template <typename T>
__host__ __device__
constexpr bool testConstexprSpan()
{
    cuda::std::span<const T>    s1;
    cuda::std::span<const T, 0> s2;
    return
        s1.data() == nullptr && s1.size() == 0
    &&  s2.data() == nullptr && s2.size() == 0;
}


template <typename T>
__host__ __device__
void testRuntimeSpan()
{
    ASSERT_NOEXCEPT(T{});
    cuda::std::span<const T>    s1;
    cuda::std::span<const T, 0> s2;
    assert(s1.data() == nullptr && s1.size() == 0);
    assert(s2.data() == nullptr && s2.size() == 0);
}


struct A{};

int main(int, char**)
{
    STATIC_ASSERT_CXX14(testConstexprSpan<int>());
    STATIC_ASSERT_CXX14(testConstexprSpan<long>());
    STATIC_ASSERT_CXX14(testConstexprSpan<double>());
    STATIC_ASSERT_CXX14(testConstexprSpan<A>());

    testRuntimeSpan<int>();
    testRuntimeSpan<long>();
    testRuntimeSpan<double>();
    testRuntimeSpan<A>();

    checkCV();

    static_assert( cuda::std::is_default_constructible<cuda::std::span<int, cuda::std::dynamic_extent>>::value, "");
    static_assert( cuda::std::is_default_constructible<cuda::std::span<int, 0>>::value, "");
    static_assert(!cuda::std::is_default_constructible<cuda::std::span<int, 2>>::value, "");

    return 0;
}
