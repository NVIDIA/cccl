//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11

// <span>

// template<class OtherElementType, size_t OtherExtent>
//    constexpr span(const span<OtherElementType, OtherExtent>& s) noexcept;
//
//  Remarks: This constructor shall not participate in overload resolution unless:
//      Extent == dynamic_extent || Extent == OtherExtent is true, and
//      OtherElementType(*)[] is convertible to ElementType(*)[].

#include <cuda/std/cassert>
#include <cuda/std/span>

#include "test_macros.h"

__host__ __device__ void checkCV()
{
  cuda::std::span<int> sp;
  //  cuda::std::span<const          int>  csp;
  cuda::std::span<volatile int> vsp;
  //  cuda::std::span<const volatile int> cvsp;

  cuda::std::span<int, 0> sp0;
  //  cuda::std::span<const          int, 0>  csp0;
  cuda::std::span<volatile int, 0> vsp0;
  //  cuda::std::span<const volatile int, 0> cvsp0;

  //  dynamic -> dynamic
  {
    cuda::std::span<const int> s1{sp}; // a cuda::std::span<const          int> pointing at int.
    cuda::std::span<volatile int> s2{sp}; // a cuda::std::span<      volatile int> pointing at int.
    cuda::std::span<const volatile int> s3{sp}; // a cuda::std::span<const volatile int> pointing at int.
    cuda::std::span<const volatile int> s4{vsp}; // a cuda::std::span<const volatile int> pointing at volatile int.
    assert(s1.size() + s2.size() + s3.size() + s4.size() == 0);
  }

  //  static -> static
  {
    cuda::std::span<const int, 0> s1{sp0}; // a cuda::std::span<const          int> pointing at int.
    cuda::std::span<volatile int, 0> s2{sp0}; // a cuda::std::span<      volatile int> pointing at int.
    cuda::std::span<const volatile int, 0> s3{sp0}; // a cuda::std::span<const volatile int> pointing at int.
    cuda::std::span<const volatile int, 0> s4{vsp0}; // a cuda::std::span<const volatile int> pointing at volatile int.
    assert(s1.size() + s2.size() + s3.size() + s4.size() == 0);
  }

  //  static -> dynamic
  {
    cuda::std::span<const int> s1{sp0}; // a cuda::std::span<const          int> pointing at int.
    cuda::std::span<volatile int> s2{sp0}; // a cuda::std::span<      volatile int> pointing at int.
    cuda::std::span<const volatile int> s3{sp0}; // a cuda::std::span<const volatile int> pointing at int.
    cuda::std::span<const volatile int> s4{vsp0}; // a cuda::std::span<const volatile int> pointing at volatile int.
    assert(s1.size() + s2.size() + s3.size() + s4.size() == 0);
  }

  //  dynamic -> static (not allowed)
}

template <typename T>
__host__ __device__ constexpr bool testConstexprSpan()
{
  cuda::std::span<T> s0{};
  cuda::std::span<T, 0> s1{};
  cuda::std::span<T> s2(s1); // static -> dynamic
  ASSERT_NOEXCEPT(cuda::std::span<T>{s0});
  ASSERT_NOEXCEPT(cuda::std::span<T, 0>{s1});
  ASSERT_NOEXCEPT(cuda::std::span<T>{s1});

  return s0.data() == nullptr && s0.size() == 0 && s1.data() == nullptr && s1.size() == 0 && s2.data() == nullptr
      && s2.size() == 0;
}

template <typename T>
__host__ __device__ void testRuntimeSpan()
{
  cuda::std::span<T> s0{};
  cuda::std::span<T, 0> s1{};
  cuda::std::span<T> s2(s1); // static -> dynamic
  ASSERT_NOEXCEPT(cuda::std::span<T>{s0});
  ASSERT_NOEXCEPT(cuda::std::span<T, 0>{s1});
  ASSERT_NOEXCEPT(cuda::std::span<T>{s1});

  assert(s0.data() == nullptr && s0.size() == 0);
  assert(s1.data() == nullptr && s1.size() == 0);
  assert(s2.data() == nullptr && s2.size() == 0);
}

struct A
{};

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

  return 0;
}
