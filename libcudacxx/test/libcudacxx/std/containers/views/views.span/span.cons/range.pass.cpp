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
__host__ __device__ constexpr void test()
{
  { // dynamic span
    {
      IsAContainer<T> val{};
      cuda::std::span<T> s1{val};
      assert(s1.data() == val.getV());
      assert(s1.size() == 1);
    }

    {
      IsAContainer<const T> val{};
      cuda::std::span<const T> s1{val};
      assert(s1.data() == val.getV());
      assert(s1.size() == 1);
    }

    {
      const IsAContainer<T> val{};
      cuda::std::span<const T> s1{val};
      assert(s1.data() == val.getV());
      assert(s1.size() == 1);
    }
  }

  { // static span
    {
      IsAContainer<T> val{};
      cuda::std::span<T, 1> s1{val};
      assert(s1.data() == val.getV());
      assert(s1.size() == 1);
    }

    {
      IsAContainer<const T> val{};
      cuda::std::span<const T, 1> s1{val};
      assert(s1.data() == val.getV());
      assert(s1.size() == 1);
    }

    {
      const IsAContainer<T> val{};
      cuda::std::span<const T, 1> s1{val};
      assert(s1.data() == val.getV());
      assert(s1.size() == 1);
    }
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

__host__ __device__ constexpr void test()
{
  test<int>();
  test<long>();
  test<double>();
  test<A>();
}

int main(int, char**)
{
  test();
  checkCV();

#if !TEST_COMPILER(NVRTC)
  NV_IF_TARGET(NV_IS_HOST, (test_std<int>(); test_std<A>();))
#endif // !TEST_COMPILER(NVRTC)

  return 0;
}
