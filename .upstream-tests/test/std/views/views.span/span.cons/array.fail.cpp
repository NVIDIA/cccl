//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11
// UNSUPPORTED: nvrtc

// <span>

// template<size_t N>
//     constexpr span(element_type (&arr)[N]) noexcept;
// template<size_t N>
//     constexpr span(array<value_type, N>& arr) noexcept;
// template<size_t N>
//     constexpr span(const array<value_type, N>& arr) noexcept;
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

int main(int, char**)
{
//  Size wrong
    {
    cuda::std::span<int, 2>   s1(arr); // expected-error {{no matching constructor for initialization of 'cuda::std::span<int, 2>'}}
    }

//  Type wrong
    {
    cuda::std::span<float>    s1(arr);   // expected-error {{no matching constructor for initialization of 'cuda::std::span<float>'}}
    cuda::std::span<float, 3> s2(arr);   // expected-error {{no matching constructor for initialization of 'cuda::std::span<float, 3>'}}
    }

//  CV wrong (dynamically sized)
    {
    cuda::std::span<               int> s1{ carr};    // expected-error {{no matching constructor for initialization of 'cuda::std::span<int>'}}
    cuda::std::span<               int> s2{ varr};    // expected-error {{no matching constructor for initialization of 'cuda::std::span<int>'}}
    cuda::std::span<               int> s3{cvarr};    // expected-error {{no matching constructor for initialization of 'cuda::std::span<int>'}}
    cuda::std::span<const          int> s4{ varr};    // expected-error {{no matching constructor for initialization of 'cuda::std::span<const int>'}}
    cuda::std::span<const          int> s5{cvarr};    // expected-error {{no matching constructor for initialization of 'cuda::std::span<const int>'}}
    cuda::std::span<      volatile int> s6{ carr};    // expected-error {{no matching constructor for initialization of 'cuda::std::span<volatile int>'}}
    cuda::std::span<      volatile int> s7{cvarr};    // expected-error {{no matching constructor for initialization of 'cuda::std::span<volatile int>'}}
    }

//  CV wrong (statically sized)
    {
    cuda::std::span<               int,3> s1{ carr};  // expected-error {{no matching constructor for initialization of 'cuda::std::span<int, 3>'}}
    cuda::std::span<               int,3> s2{ varr};  // expected-error {{no matching constructor for initialization of 'cuda::std::span<int, 3>'}}
    cuda::std::span<               int,3> s3{cvarr};  // expected-error {{no matching constructor for initialization of 'cuda::std::span<int, 3>'}}
    cuda::std::span<const          int,3> s4{ varr};  // expected-error {{no matching constructor for initialization of 'cuda::std::span<const int, 3>'}}
    cuda::std::span<const          int,3> s5{cvarr};  // expected-error {{no matching constructor for initialization of 'cuda::std::span<const int, 3>'}}
    cuda::std::span<      volatile int,3> s6{ carr};  // expected-error {{no matching constructor for initialization of 'cuda::std::span<volatile int, 3>'}}
    cuda::std::span<      volatile int,3> s7{cvarr};  // expected-error {{no matching constructor for initialization of 'cuda::std::span<volatile int, 3>'}}
    }

  return 0;
}
