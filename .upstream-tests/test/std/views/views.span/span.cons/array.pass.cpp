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

// template<size_t N>
//     constexpr span(element_type (&arr)[N]) noexcept;
//
// Remarks: These constructors shall not participate in overload resolution unless:
//   — extent == dynamic_extent || N == extent is true, and
//   — remove_pointer_t<decltype(data(arr))>(*)[] is convertible to ElementType(*)[].
//


#include <cuda/std/span>
#include <cuda/std/cassert>

#include "test_macros.h"

__device__                int   arr[] = {1,2,3};
__device__ const          int  carr[] = {4,5,6};
__device__       volatile int  varr[] = {7,8,9};
__device__ const volatile int cvarr[] = {1,3,5};

__host__ __device__ void checkCV()
{
//  Types the same (dynamic sized)
    {
    cuda::std::span<               int> s1{  arr};    // a cuda::std::span<               int> pointing at int.
    cuda::std::span<const          int> s2{ carr};    // a cuda::std::span<const          int> pointing at const int.
    cuda::std::span<      volatile int> s3{ varr};    // a cuda::std::span<      volatile int> pointing at volatile int.
    cuda::std::span<const volatile int> s4{cvarr};    // a cuda::std::span<const volatile int> pointing at const volatile int.
    assert(s1.size() + s2.size() + s3.size() + s4.size() == 12);
    }

//  Types the same (static sized)
    {
    cuda::std::span<               int,3> s1{  arr};  // a cuda::std::span<               int> pointing at int.
    cuda::std::span<const          int,3> s2{ carr};  // a cuda::std::span<const          int> pointing at const int.
    cuda::std::span<      volatile int,3> s3{ varr};  // a cuda::std::span<      volatile int> pointing at volatile int.
    cuda::std::span<const volatile int,3> s4{cvarr};  // a cuda::std::span<const volatile int> pointing at const volatile int.
    assert(s1.size() + s2.size() + s3.size() + s4.size() == 12);
    }


//  types different (dynamic sized)
    {
    cuda::std::span<const          int> s1{ arr};     // a cuda::std::span<const          int> pointing at int.
    cuda::std::span<      volatile int> s2{ arr};     // a cuda::std::span<      volatile int> pointing at int.
    cuda::std::span<      volatile int> s3{ arr};     // a cuda::std::span<      volatile int> pointing at const int.
    cuda::std::span<const volatile int> s4{ arr};     // a cuda::std::span<const volatile int> pointing at int.
    cuda::std::span<const volatile int> s5{carr};     // a cuda::std::span<const volatile int> pointing at const int.
    cuda::std::span<const volatile int> s6{varr};     // a cuda::std::span<const volatile int> pointing at volatile int.
    assert(s1.size() + s2.size() + s3.size() + s4.size() + s5.size() + s6.size() == 18);
    }

//  types different (static sized)
    {
    cuda::std::span<const          int,3> s1{ arr};   // a cuda::std::span<const          int> pointing at int.
    cuda::std::span<      volatile int,3> s2{ arr};   // a cuda::std::span<      volatile int> pointing at int.
    cuda::std::span<      volatile int,3> s3{ arr};   // a cuda::std::span<      volatile int> pointing at const int.
    cuda::std::span<const volatile int,3> s4{ arr};   // a cuda::std::span<const volatile int> pointing at int.
    cuda::std::span<const volatile int,3> s5{carr};   // a cuda::std::span<const volatile int> pointing at const int.
    cuda::std::span<const volatile int,3> s6{varr};   // a cuda::std::span<const volatile int> pointing at volatile int.
    assert(s1.size() + s2.size() + s3.size() + s4.size() + s5.size() + s6.size() == 18);
    }
}

template<class T>
__host__ __device__ constexpr bool testSpan()
{
    T val[2] = {};

    ASSERT_NOEXCEPT(cuda::std::span<T>{val});
    ASSERT_NOEXCEPT(cuda::std::span<T, 2>{val});
    ASSERT_NOEXCEPT(cuda::std::span<const T>{val});
    ASSERT_NOEXCEPT(cuda::std::span<const T, 2>{val});

    cuda::std::span<T> s1 = val;
    cuda::std::span<T, 2> s2 = val;
    cuda::std::span<const T> s3 = val;
    cuda::std::span<const T, 2> s4 = val;
    assert(s1.data() == val && s1.size() == 2);
    assert(s2.data() == val && s2.size() == 2);
    assert(s3.data() == val && s3.size() == 2);
    assert(s4.data() == val && s4.size() == 2);

    cuda::std::span<const int> s5 = {{1,2}};
    cuda::std::span<const int, 2> s6 = {{1,2}};
    assert(s5.size() == 2);  // and it dangles
    assert(s6.size() == 2);  // and it dangles

    return true;
}


struct A {};

int main(int, char**)
{
    testSpan<int>();
    testSpan<double>();
    testSpan<A>();

    static_assert(testSpan<int>(), "");
    static_assert(testSpan<double>(), "");
    static_assert(testSpan<A>(), "");

    checkCV();

    return 0;
}
