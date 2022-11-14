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

// constexpr pointer data() const noexcept;
//


#include <cuda/std/span>
#include <cuda/std/cassert>

#include "test_macros.h"

template <typename Span>
__host__ __device__
constexpr bool testConstexprSpan(Span sp, typename Span::pointer ptr)
{
    ASSERT_NOEXCEPT(sp.data());
    return sp.data() == ptr;
}


template <typename Span>
__host__ __device__
void testRuntimeSpan(Span sp, typename Span::pointer ptr)
{
    ASSERT_NOEXCEPT(sp.data());
    assert(sp.data() == ptr);
}

struct A{};
__device__ constexpr int iArr1[] = { 0,  1,  2,  3,  4,  5,  6,  7,  8,  9};
__device__           int iArr2[] = {10, 11, 12, 13, 14, 15, 16, 17, 18, 19};

int main(int, char**)
{

//  dynamic size
    static_assert(testConstexprSpan(cuda::std::span<int>(), nullptr),         "");
    static_assert(testConstexprSpan(cuda::std::span<long>(), nullptr),        "");
    static_assert(testConstexprSpan(cuda::std::span<double>(), nullptr),      "");
    static_assert(testConstexprSpan(cuda::std::span<A>(), nullptr),           "");

    static_assert(testConstexprSpan(cuda::std::span<const int>(iArr1, 1), iArr1), "");
    static_assert(testConstexprSpan(cuda::std::span<const int>(iArr1, 2), iArr1), "");
    static_assert(testConstexprSpan(cuda::std::span<const int>(iArr1, 3), iArr1), "");
    static_assert(testConstexprSpan(cuda::std::span<const int>(iArr1, 4), iArr1), "");

    static_assert(testConstexprSpan(cuda::std::span<const int>(iArr1 + 1, 1), iArr1 + 1), "");
    static_assert(testConstexprSpan(cuda::std::span<const int>(iArr1 + 2, 2), iArr1 + 2), "");
    static_assert(testConstexprSpan(cuda::std::span<const int>(iArr1 + 3, 3), iArr1 + 3), "");
    static_assert(testConstexprSpan(cuda::std::span<const int>(iArr1 + 4, 4), iArr1 + 4), "");

//  static size
    static_assert(testConstexprSpan(cuda::std::span<int, 0>(), nullptr),         "");
    static_assert(testConstexprSpan(cuda::std::span<long, 0>(), nullptr),        "");
    static_assert(testConstexprSpan(cuda::std::span<double, 0>(), nullptr),      "");
    static_assert(testConstexprSpan(cuda::std::span<A, 0>(), nullptr),           "");

    static_assert(testConstexprSpan(cuda::std::span<const int, 1>(iArr1, 1), iArr1), "");
    static_assert(testConstexprSpan(cuda::std::span<const int, 2>(iArr1, 2), iArr1), "");
    static_assert(testConstexprSpan(cuda::std::span<const int, 3>(iArr1, 3), iArr1), "");
    static_assert(testConstexprSpan(cuda::std::span<const int, 4>(iArr1, 4), iArr1), "");

    static_assert(testConstexprSpan(cuda::std::span<const int, 1>(iArr1 + 1, 1), iArr1 + 1), "");
    static_assert(testConstexprSpan(cuda::std::span<const int, 2>(iArr1 + 2, 2), iArr1 + 2), "");
    static_assert(testConstexprSpan(cuda::std::span<const int, 3>(iArr1 + 3, 3), iArr1 + 3), "");
    static_assert(testConstexprSpan(cuda::std::span<const int, 4>(iArr1 + 4, 4), iArr1 + 4), "");


//  dynamic size
    testRuntimeSpan(cuda::std::span<int>(), nullptr);
    testRuntimeSpan(cuda::std::span<long>(), nullptr);
    testRuntimeSpan(cuda::std::span<double>(), nullptr);
    testRuntimeSpan(cuda::std::span<A>(), nullptr);

    testRuntimeSpan(cuda::std::span<int>(iArr2, 1), iArr2);
    testRuntimeSpan(cuda::std::span<int>(iArr2, 2), iArr2);
    testRuntimeSpan(cuda::std::span<int>(iArr2, 3), iArr2);
    testRuntimeSpan(cuda::std::span<int>(iArr2, 4), iArr2);

    testRuntimeSpan(cuda::std::span<int>(iArr2 + 1, 1), iArr2 + 1);
    testRuntimeSpan(cuda::std::span<int>(iArr2 + 2, 2), iArr2 + 2);
    testRuntimeSpan(cuda::std::span<int>(iArr2 + 3, 3), iArr2 + 3);
    testRuntimeSpan(cuda::std::span<int>(iArr2 + 4, 4), iArr2 + 4);

//  static size
    testRuntimeSpan(cuda::std::span<int, 0>(), nullptr);
    testRuntimeSpan(cuda::std::span<long, 0>(), nullptr);
    testRuntimeSpan(cuda::std::span<double, 0>(), nullptr);
    testRuntimeSpan(cuda::std::span<A, 0>(), nullptr);

    testRuntimeSpan(cuda::std::span<int, 1>(iArr2, 1), iArr2);
    testRuntimeSpan(cuda::std::span<int, 2>(iArr2, 2), iArr2);
    testRuntimeSpan(cuda::std::span<int, 3>(iArr2, 3), iArr2);
    testRuntimeSpan(cuda::std::span<int, 4>(iArr2, 4), iArr2);

    testRuntimeSpan(cuda::std::span<int, 1>(iArr2 + 1, 1), iArr2 + 1);
    testRuntimeSpan(cuda::std::span<int, 2>(iArr2 + 2, 2), iArr2 + 2);
    testRuntimeSpan(cuda::std::span<int, 3>(iArr2 + 3, 3), iArr2 + 3);
    testRuntimeSpan(cuda::std::span<int, 4>(iArr2 + 4, 4), iArr2 + 4);

  return 0;
}
