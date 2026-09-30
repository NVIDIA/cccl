//===---------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===---------------------------------------------------------------------===//

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

#if !TEST_COMPILER(NVRTC)
#  if defined(__cpp_lib_span)
#    include <span>
#  endif //__cpp_lib_span
#  include <vector>
#endif // !TEST_COMPILER(NVRTC)

struct A
{};

//  Look ma - I'm a container!
template <typename T>
struct IsAContainer
{
  TEST_FUNC constexpr IsAContainer()
      : v_{}
  {}
  TEST_FUNC constexpr size_t size() const
  {
    return 1;
  }
  TEST_FUNC constexpr T* data()
  {
    return &v_;
  }
  TEST_FUNC constexpr const T* data() const
  {
    return &v_;
  }
  TEST_FUNC constexpr T* begin()
  {
    return &v_;
  }
  TEST_FUNC constexpr const T* begin() const
  {
    return &v_;
  }
  TEST_FUNC constexpr T* end()
  {
    return &v_ + 1;
  }
  TEST_FUNC constexpr const T* end() const
  {
    return &v_ + 1;
  }

  TEST_FUNC constexpr T const* getV() const
  {
    return &v_;
  } // for checking
  T v_;
};

TEST_FUNC void checkCV()
{
  IsAContainer<int> v{};

  //  Types the same
  {
    [[maybe_unused]] cuda::std::span<int> s1{v}; // a span<int> pointing at int.
  }

  //  types different
  {
    [[maybe_unused]] cuda::std::span<const int> s1{v}; // a span<const int> pointing at int.
    [[maybe_unused]] cuda::std::span<volatile int> s2{v}; // a span<volatile int> pointing at int.
    [[maybe_unused]] cuda::std::span<volatile int> s3{v}; // a span<volatile int> pointing at const int.
    [[maybe_unused]] cuda::std::span<const volatile int> s4{v}; // a span<const volatile int> pointing at int.
  }

  //  Constructing a const view from a temporary
  {
    [[maybe_unused]] cuda::std::span<const int> s1{IsAContainer<int>()};
  }
}

template <typename T>
TEST_FUNC constexpr void test()
{
  {
    IsAContainer<T> val{};
    const IsAContainer<T> cVal;
    cuda::std::span<T> s1{val};
    cuda::std::span<const T> s2{cVal};
    assert(s1.data() == val.getV() && s1.size() == 1);
    assert(s2.data() == cVal.getV() && s2.size() == 1);
  }

  {
    IsAContainer<T> val{};
    const IsAContainer<T> cVal;
    cuda::std::span<T, 1> s1{val};
    cuda::std::span<const T, 1> s2{cVal};
    assert(s1.data() == val.getV() && s1.size() == 1);
    assert(s2.data() == cVal.getV() && s2.size() == 1);
  }
}

#if !TEST_COMPILER(NVRTC)
template <typename T>
void test_std()
{
  ::std::vector<T> val(1);
  const ::std::vector<T> cVal(1);
  { // from container
    cuda::std::span<T> s1{val};
    cuda::std::span<const T> s2{cVal};
    assert(s1.data() == val.data() && s1.size() == 1);
    assert(s2.data() == cVal.data() && s2.size() == 1);
  }
#  if defined(__cpp_lib_span)
  { // from std::span
    // we are requiring `cuda::std::enable_borrowed_range` but only get `std::enable_borrowed_range`
    // cuda::std::span<T> s1{std::span{val}};
    cuda::std::span<const T> s2{std::span{cVal}};
    // assert(s1.data() == val.data() && s1.size() == 1);
    assert(s2.data() == cVal.data() && s2.size() == 1);
  }
#  endif // __cpp_lib_span
}
#endif // !TEST_COMPILER(NVRTC)

TEST_FUNC constexpr bool test()
{
  test<int>();
  test<long>();
  test<double>();
  test<A>();

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  checkCV();

#if !TEST_COMPILER(NVRTC)
  NV_IF_TARGET(NV_IS_HOST, (test_std<int>(); test_std<A>();))
#endif // !TEST_COMPILER(NVRTC)

  return 0;
}
