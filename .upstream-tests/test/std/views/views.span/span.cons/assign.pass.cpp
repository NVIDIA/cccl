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

//  constexpr span& operator=(const span& other) noexcept = default;

#include <cuda/std/span>
#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/utility>

#include "test_macros.h"

using cuda::std::span;

template <typename T>
__host__ __device__
constexpr bool doAssign(T lhs, T rhs)
{
    ASSERT_NOEXCEPT(cuda::std::declval<T&>() = rhs);
    lhs = rhs;
    return lhs.data() == rhs.data()
     &&    lhs.size() == rhs.size();
}

struct A{};

__device__ constexpr int carr1[] = {1,2,3,4};
__device__ constexpr int carr2[] = {3,4,5};
__device__ constexpr int carr3[] = {7,8};
__device__           int   arr[] = {5,6,7,9};

int main(int, char**)
{

//  constexpr dynamically sized assignment
    {
//  On systems where 'ptrdiff_t' is a synonym for 'int',
//  the call span(ptr, 0) selects the (pointer, size_type) constructor.
//  On systems where 'ptrdiff_t' is NOT a synonym for 'int',
//  it is ambiguous, because of 0 also being convertible to a null pointer
//  and so the compiler can't choose between:
//      span(pointer, size_type)
//  and span(pointer, pointer)
//  We cast zero to std::ptrdiff_t to remove that ambiguity.
//  Example:
//      On darwin x86_64, ptrdiff_t is the same as long int.
//      On darwin i386, ptrdiff_t is the same as int.
        constexpr cuda::std::span<const int> spans[] = {
            {},
            {carr1, static_cast<cuda::std::size_t>(0)},
            {carr1,     1U},
            {carr1,     2U},
            {carr1,     3U},
            {carr1,     4U},
            {carr2, static_cast<cuda::std::size_t>(0)},
            {carr2,     1U},
            {carr2,     2U},
            {carr2,     3U},
            {carr3, static_cast<cuda::std::size_t>(0)},
            {carr3,     1U},
            {carr3,     2U}
            };
        //static_assert(cuda::std::size(spans) == 13 );

//  No for loops in constexpr land :-(
        STATIC_ASSERT_CXX14(doAssign(spans[0], spans[0]));
        STATIC_ASSERT_CXX14(doAssign(spans[0], spans[1]));
        STATIC_ASSERT_CXX14(doAssign(spans[0], spans[2]));
        STATIC_ASSERT_CXX14(doAssign(spans[0], spans[3]));
        STATIC_ASSERT_CXX14(doAssign(spans[0], spans[4]));
        STATIC_ASSERT_CXX14(doAssign(spans[0], spans[5]));
        STATIC_ASSERT_CXX14(doAssign(spans[0], spans[6]));
        STATIC_ASSERT_CXX14(doAssign(spans[0], spans[7]));
        STATIC_ASSERT_CXX14(doAssign(spans[0], spans[8]));
        STATIC_ASSERT_CXX14(doAssign(spans[0], spans[9]));
        STATIC_ASSERT_CXX14(doAssign(spans[0], spans[10]));
        STATIC_ASSERT_CXX14(doAssign(spans[0], spans[11]));
        STATIC_ASSERT_CXX14(doAssign(spans[0], spans[12]));

        STATIC_ASSERT_CXX14(doAssign(spans[1], spans[1]));
        STATIC_ASSERT_CXX14(doAssign(spans[1], spans[2]));
        STATIC_ASSERT_CXX14(doAssign(spans[1], spans[3]));
        STATIC_ASSERT_CXX14(doAssign(spans[1], spans[4]));
        STATIC_ASSERT_CXX14(doAssign(spans[1], spans[5]));
        STATIC_ASSERT_CXX14(doAssign(spans[1], spans[6]));
        STATIC_ASSERT_CXX14(doAssign(spans[1], spans[7]));
        STATIC_ASSERT_CXX14(doAssign(spans[1], spans[8]));
        STATIC_ASSERT_CXX14(doAssign(spans[1], spans[9]));
        STATIC_ASSERT_CXX14(doAssign(spans[1], spans[10]));
        STATIC_ASSERT_CXX14(doAssign(spans[1], spans[11]));
        STATIC_ASSERT_CXX14(doAssign(spans[1], spans[12]));

        STATIC_ASSERT_CXX14(doAssign(spans[2], spans[2]));
        STATIC_ASSERT_CXX14(doAssign(spans[2], spans[3]));
        STATIC_ASSERT_CXX14(doAssign(spans[2], spans[4]));
        STATIC_ASSERT_CXX14(doAssign(spans[2], spans[5]));
        STATIC_ASSERT_CXX14(doAssign(spans[2], spans[6]));
        STATIC_ASSERT_CXX14(doAssign(spans[2], spans[7]));
        STATIC_ASSERT_CXX14(doAssign(spans[2], spans[8]));
        STATIC_ASSERT_CXX14(doAssign(spans[2], spans[9]));
        STATIC_ASSERT_CXX14(doAssign(spans[2], spans[10]));
        STATIC_ASSERT_CXX14(doAssign(spans[2], spans[11]));
        STATIC_ASSERT_CXX14(doAssign(spans[2], spans[12]));

        STATIC_ASSERT_CXX14(doAssign(spans[3], spans[3]));
        STATIC_ASSERT_CXX14(doAssign(spans[3], spans[4]));
        STATIC_ASSERT_CXX14(doAssign(spans[3], spans[4]));
        STATIC_ASSERT_CXX14(doAssign(spans[3], spans[4]));
        STATIC_ASSERT_CXX14(doAssign(spans[3], spans[4]));
        STATIC_ASSERT_CXX14(doAssign(spans[3], spans[4]));
        STATIC_ASSERT_CXX14(doAssign(spans[3], spans[4]));
        STATIC_ASSERT_CXX14(doAssign(spans[3], spans[4]));
        STATIC_ASSERT_CXX14(doAssign(spans[3], spans[4]));
        STATIC_ASSERT_CXX14(doAssign(spans[3], spans[10]));
        STATIC_ASSERT_CXX14(doAssign(spans[3], spans[11]));
        STATIC_ASSERT_CXX14(doAssign(spans[3], spans[12]));

        STATIC_ASSERT_CXX14(doAssign(spans[4], spans[4]));
        STATIC_ASSERT_CXX14(doAssign(spans[4], spans[5]));
        STATIC_ASSERT_CXX14(doAssign(spans[4], spans[6]));
        STATIC_ASSERT_CXX14(doAssign(spans[4], spans[7]));
        STATIC_ASSERT_CXX14(doAssign(spans[4], spans[8]));
        STATIC_ASSERT_CXX14(doAssign(spans[4], spans[9]));
        STATIC_ASSERT_CXX14(doAssign(spans[4], spans[10]));
        STATIC_ASSERT_CXX14(doAssign(spans[4], spans[11]));
        STATIC_ASSERT_CXX14(doAssign(spans[4], spans[12]));

        STATIC_ASSERT_CXX14(doAssign(spans[5], spans[5]));
        STATIC_ASSERT_CXX14(doAssign(spans[5], spans[6]));
        STATIC_ASSERT_CXX14(doAssign(spans[5], spans[7]));
        STATIC_ASSERT_CXX14(doAssign(spans[5], spans[8]));
        STATIC_ASSERT_CXX14(doAssign(spans[5], spans[9]));
        STATIC_ASSERT_CXX14(doAssign(spans[5], spans[10]));
        STATIC_ASSERT_CXX14(doAssign(spans[5], spans[11]));
        STATIC_ASSERT_CXX14(doAssign(spans[5], spans[12]));

        STATIC_ASSERT_CXX14(doAssign(spans[6], spans[6]));
        STATIC_ASSERT_CXX14(doAssign(spans[6], spans[7]));
        STATIC_ASSERT_CXX14(doAssign(spans[6], spans[8]));
        STATIC_ASSERT_CXX14(doAssign(spans[6], spans[9]));
        STATIC_ASSERT_CXX14(doAssign(spans[6], spans[10]));
        STATIC_ASSERT_CXX14(doAssign(spans[6], spans[11]));
        STATIC_ASSERT_CXX14(doAssign(spans[6], spans[12]));

        STATIC_ASSERT_CXX14(doAssign(spans[7], spans[7]));
        STATIC_ASSERT_CXX14(doAssign(spans[7], spans[8]));
        STATIC_ASSERT_CXX14(doAssign(spans[7], spans[9]));
        STATIC_ASSERT_CXX14(doAssign(spans[7], spans[10]));
        STATIC_ASSERT_CXX14(doAssign(spans[7], spans[11]));
        STATIC_ASSERT_CXX14(doAssign(spans[7], spans[12]));

        STATIC_ASSERT_CXX14(doAssign(spans[8], spans[8]));
        STATIC_ASSERT_CXX14(doAssign(spans[8], spans[9]));
        STATIC_ASSERT_CXX14(doAssign(spans[8], spans[10]));
        STATIC_ASSERT_CXX14(doAssign(spans[8], spans[11]));
        STATIC_ASSERT_CXX14(doAssign(spans[8], spans[12]));

        STATIC_ASSERT_CXX14(doAssign(spans[9], spans[9]));
        STATIC_ASSERT_CXX14(doAssign(spans[9], spans[10]));
        STATIC_ASSERT_CXX14(doAssign(spans[9], spans[11]));
        STATIC_ASSERT_CXX14(doAssign(spans[9], spans[12]));

        STATIC_ASSERT_CXX14(doAssign(spans[10], spans[10]));
        STATIC_ASSERT_CXX14(doAssign(spans[10], spans[11]));
        STATIC_ASSERT_CXX14(doAssign(spans[10], spans[12]));

        STATIC_ASSERT_CXX14(doAssign(spans[11], spans[11]));
        STATIC_ASSERT_CXX14(doAssign(spans[11], spans[12]));

        STATIC_ASSERT_CXX14(doAssign(spans[12], spans[12]));

//      for (size_t i = 0; i < cuda::std::size(spans); ++i)
//          for (size_t j = i; j < cuda::std::size(spans); ++j)
//              static_assert(doAssign(spans[i], spans[j]));
    }

//  constexpr statically sized assignment
    {
        using spanType = cuda::std::span<const int,2>;
        
        constexpr spanType spans[] = {
            spanType{carr1, 2},
            spanType{carr1 + 1, 2},
            spanType{carr1 + 2, 2},
            spanType{carr2, 2},
            spanType{carr2 + 1, 2},
            spanType{carr3, 2}
            };
        STATIC_ASSERT_CXX14(cuda::std::size(spans) == 6 );

        //  No for loops in constexpr land :-(
        STATIC_ASSERT_CXX14(doAssign(spans[0], spans[0]));
        STATIC_ASSERT_CXX14(doAssign(spans[0], spans[1]));
        STATIC_ASSERT_CXX14(doAssign(spans[0], spans[2]));
        STATIC_ASSERT_CXX14(doAssign(spans[0], spans[3]));
        STATIC_ASSERT_CXX14(doAssign(spans[0], spans[4]));
        STATIC_ASSERT_CXX14(doAssign(spans[0], spans[5]));

        STATIC_ASSERT_CXX14(doAssign(spans[1], spans[1]));
        STATIC_ASSERT_CXX14(doAssign(spans[1], spans[2]));
        STATIC_ASSERT_CXX14(doAssign(spans[1], spans[3]));
        STATIC_ASSERT_CXX14(doAssign(spans[1], spans[4]));
        STATIC_ASSERT_CXX14(doAssign(spans[1], spans[5]));

        STATIC_ASSERT_CXX14(doAssign(spans[2], spans[2]));
        STATIC_ASSERT_CXX14(doAssign(spans[2], spans[3]));
        STATIC_ASSERT_CXX14(doAssign(spans[2], spans[4]));
        STATIC_ASSERT_CXX14(doAssign(spans[2], spans[5]));

        STATIC_ASSERT_CXX14(doAssign(spans[3], spans[3]));
        STATIC_ASSERT_CXX14(doAssign(spans[3], spans[4]));
        STATIC_ASSERT_CXX14(doAssign(spans[3], spans[5]));

        STATIC_ASSERT_CXX14(doAssign(spans[4], spans[4]));
        STATIC_ASSERT_CXX14(doAssign(spans[4], spans[5]));

        STATIC_ASSERT_CXX14(doAssign(spans[5], spans[5]));

//      for (size_t i = 0; i < cuda::std::size(spans); ++i)
//          for (size_t j = i; j < cuda::std::size(spans); ++j)
//              static_assert(doAssign(spans[i], spans[j]));
    }

//  dynamically sized assignment
    {
        cuda::std::span<int> spans[] = {
            {},
            {arr,     arr + 1},
            {arr,     arr + 2},
            {arr,     arr + 3},
            {arr + 1, arr + 3} // same size as s2
            };

        for (size_t i = 0; i < 5; ++i)
            for (size_t j = i; j < 5; ++j)
                assert((doAssign(spans[i], spans[j])));
    }

//  statically sized assignment
    {
        using spanType = cuda::std::span<int,2>;
        spanType spans[] = {
            spanType{arr,     arr + 2},
            spanType{arr + 1, arr + 3},
            spanType{arr + 2, arr + 4}
            };

        for (size_t i = 0; i < 3; ++i)
            for (size_t j = i; j < 3; ++j)
                assert((doAssign(spans[i], spans[j])));
    }

  return 0;
}
