//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//
// UNSUPPORTED: c++03, c++11

// <cuda/std/span>

//  template<class Container>
//    constexpr explicit(Extent != dynamic_extent) span(Container&);
//  template<class Container>
//    constexpr explicit(Extent != dynamic_extent) span(Container const&);

// This test checks for libc++'s non-conforming temporary extension to cuda::std::span
// to support construction from containers that look like contiguous ranges.
//
// This extension is only supported when we don't ship <ranges>, and we can
// remove it once we get rid of _LIBCUDACXX_HAS_NO_INCOMPLETE_RANGES.

#include <cuda/std/cassert>
#include <cuda/std/span>

#include "test_macros.h"

#if !defined(TEST_COMPILER_NVRTC)
#  include <vector>
#endif // !TEST_COMPILER_NVRTC

//  Look ma - I'm a container!
template <typename T>
struct IsAContainer
{
  __host__ __device__ constexpr IsAContainer()
      : v_{}
  {}
  __host__ __device__ constexpr size_t size() const
  {
    return 1;
  }
  __host__ __device__ constexpr T* data()
  {
    return &v_;
  }
  __host__ __device__ constexpr const T* data() const
  {
    return &v_;
  }
  __host__ __device__ constexpr T* begin()
  {
    return &v_;
  }
  __host__ __device__ constexpr const T* begin() const
  {
    return &v_;
  }
  __host__ __device__ constexpr T* end()
  {
    return &v_ + 1;
  }
  __host__ __device__ constexpr const T* end() const
  {
    return &v_ + 1;
  }

  __host__ __device__ constexpr T const* getV() const
  {
    return &v_;
  } // for checking
  T v_;
};

__host__ __device__ void checkCV()
{
  IsAContainer<int> v{};

  //  Types the same
  {
    cuda::std::span<int> s1{v}; // a span<               int> pointing at int.
    unused(s1);
  }

  //  types different
  {
    cuda::std::span<const int> s1{v}; // a span<const          int> pointing at int.
    cuda::std::span<volatile int> s2{v}; // a span<      volatile int> pointing at int.
    cuda::std::span<volatile int> s3{v}; // a span<      volatile int> pointing at const int.
    cuda::std::span<const volatile int> s4{v}; // a span<const volatile int> pointing at int.
    unused(s1);
    unused(s2);
    unused(s3);
    unused(s4);
  }

  //  Constructing a const view from a temporary
  {
    cuda::std::span<const int> s1{IsAContainer<int>()};
    unused(s1);
  }
}

template <typename T>
__host__ __device__ constexpr bool testConstexprSpan()
{
  constexpr IsAContainer<const T> val{};
  cuda::std::span<const T> s1{val};
  return s1.data() == val.getV() && s1.size() == 1;
}

template <typename T>
__host__ __device__ constexpr bool testConstexprSpanStatic()
{
  constexpr IsAContainer<const T> val{};
  cuda::std::span<const T, 1> s1{val};
  return s1.data() == val.getV() && s1.size() == 1;
}

template <typename T>
__host__ __device__ void testRuntimeSpan()
{
  IsAContainer<T> val{};
  const IsAContainer<T> cVal;
  cuda::std::span<T> s1{val};
  cuda::std::span<const T> s2{cVal};
  assert(s1.data() == val.getV() && s1.size() == 1);
  assert(s2.data() == cVal.getV() && s2.size() == 1);
}

template <typename T>
__host__ __device__ void testRuntimeSpanStatic()
{
  IsAContainer<T> val{};
  const IsAContainer<T> cVal;
  cuda::std::span<T, 1> s1{val};
  cuda::std::span<const T, 1> s2{cVal};
  assert(s1.data() == val.getV() && s1.size() == 1);
  assert(s2.data() == cVal.getV() && s2.size() == 1);
}

#if !defined(TEST_COMPILER_NVRTC)
template <typename T>
void testContainers()
{
  ::std::vector<T> val(1);
  const ::std::vector<T> cVal(1);
  cuda::std::span<T> s1{val};
  cuda::std::span<const T> s2{cVal};
  assert(s1.data() == val.data() && s1.size() == 1);
  assert(s2.data() == cVal.data() && s2.size() == 1);
}
#endif // !TEST_COMPILER_NVRTC

struct A
{};

int main(int, char**)
{
  static_assert(testConstexprSpan<int>(), "");
  static_assert(testConstexprSpan<long>(), "");
  static_assert(testConstexprSpan<double>(), "");
  static_assert(testConstexprSpan<A>(), "");

  static_assert(testConstexprSpanStatic<int>(), "");
  static_assert(testConstexprSpanStatic<long>(), "");
  static_assert(testConstexprSpanStatic<double>(), "");
  static_assert(testConstexprSpanStatic<A>(), "");

  testRuntimeSpan<int>();
  testRuntimeSpan<long>();
  testRuntimeSpan<double>();
  testRuntimeSpan<A>();

  testRuntimeSpanStatic<int>();
  testRuntimeSpanStatic<long>();
  testRuntimeSpanStatic<double>();
  testRuntimeSpanStatic<A>();

  checkCV();

#if !defined(TEST_COMPILER_NVRTC)
  NV_IF_TARGET(NV_IS_HOST, (testContainers<int>(); testContainers<A>();))
#endif // !TEST_COMPILER_NVRTC

  return 0;
}
