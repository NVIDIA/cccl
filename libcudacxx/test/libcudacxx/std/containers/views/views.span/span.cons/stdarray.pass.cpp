//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <span>

// template<size_t N>
//     constexpr span(array<value_type, N>& arr) noexcept;
// template<size_t N>
//     constexpr span(const array<value_type, N>& arr) noexcept;
//
// Remarks: These constructors shall not participate in overload resolution unless:
//   — extent == dynamic_extent || N == extent is true, and
//   — remove_pointer_t<decltype(data(arr))>(*)[] is convertible to ElementType(*)[].
//

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/span>

#if !_CCCL_COMPILER(NVRTC)
#  include <array>
#endif // !_CCCL_COMPILER(NVRTC)

#include "test_macros.h"

struct A
{};

template <template <class, size_t> class array>
__host__ __device__ constexpr void checkCV()
{
  array<int, 3> arr = {1, 2, 3};
  //  STL says these are not cromulent
  //  std::array<const int,3> carr = {4,5,6};
  //  std::array<volatile int, 3> varr = {7,8,9};
  //  std::array<const volatile int, 3> cvarr = {1,3,5};

  //  Types the same (dynamic sized)
  {
    cuda::std::span<int> s1{arr}; // a cuda::std::span<               int> pointing at int.
  }

  //  Types the same (static sized)
  {
    cuda::std::span<int, 3> s1{arr}; // a cuda::std::span<               int> pointing at int.
  }

  //  types different (dynamic sized)
  {
    cuda::std::span<const int> s1{arr}; // a cuda::std::span<const          int> pointing at int.
    cuda::std::span<volatile int> s2{arr}; // a cuda::std::span<      volatile int> pointing at int.
    cuda::std::span<volatile int> s3{arr}; // a cuda::std::span<      volatile int> pointing at const int.
    cuda::std::span<const volatile int> s4{arr}; // a cuda::std::span<const volatile int> pointing at int.
  }

  //  types different (static sized)
  {
    cuda::std::span<const int, 3> s1{arr}; // a cuda::std::span<const          int> pointing at int.
    cuda::std::span<volatile int, 3> s2{arr}; // a cuda::std::span<      volatile int> pointing at int.
    cuda::std::span<volatile int, 3> s3{arr}; // a cuda::std::span<      volatile int> pointing at const int.
    cuda::std::span<const volatile int, 3> s4{arr}; // a cuda::std::span<const volatile int> pointing at int.
  }
}

template <template <class, size_t> class array, typename T, typename U>
__host__ __device__ constexpr void test()
{
  {
    array<U, 2> val = {U(), U()};
    static_assert(noexcept(cuda::std::span<T>{val}));
    static_assert(noexcept(cuda::std::span<T, 2>{val}));
    cuda::std::span<T> s1{val};
    cuda::std::span<T, 2> s2{val};
    assert(s1.data() == &val[0]);
    assert(s1.size() == 2);
    assert(s2.data() == &val[0]);
    assert(s2.size() == 2);
  }

  {
    const array<U, 2> val = {U(), U()};
    static_assert(noexcept(cuda::std::span<const T>{val}));
    static_assert(noexcept(cuda::std::span<const T, 2>{val}));
    cuda::std::span<const T> s1{val};
    cuda::std::span<const T, 2> s2{val};
    assert(s1.data() == &val[0]);
    assert(s1.size() == 2);
    assert(s2.data() == &val[0]);
    assert(s2.size() == 2);
  }
}

template <template <class, size_t> class array, typename T>
__host__ __device__ constexpr void test()
{
  test<array, T, T>();
  test<array, const T, const T>();
  test<array, const T, T>();
}

template <template <class, size_t> class array>
__host__ __device__ constexpr void test()
{
  test<array, int>();
  test<array, long>();
  test<array, double>();
  test<array, A>();

  test<array, int*>();
  test<array, const int*>();

  checkCV<array>();
}

__host__ __device__ constexpr bool test()
{
  test<cuda::std::array>();
#if !TEST_COMPILER(NVRTC)
  NV_IF_TARGET(NV_IS_HOST, (test<std::array>();));
#endif // !TEST_COMPILER(NVRTC)

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
