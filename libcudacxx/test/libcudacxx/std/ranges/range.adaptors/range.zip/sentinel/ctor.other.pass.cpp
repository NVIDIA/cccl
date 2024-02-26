//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// constexpr sentinel(sentinel<!Const> s);

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "../types.h"

template <class T>
struct convertible_sentinel_wrapper
{
  explicit convertible_sentinel_wrapper() = default;
  __host__ __device__ constexpr convertible_sentinel_wrapper(const T& it)
      : it_(it)
  {}

  template <class U, cuda::std::enable_if_t<cuda::std::convertible_to<const U&, T>, int> = 0>
  __host__ __device__ constexpr convertible_sentinel_wrapper(const convertible_sentinel_wrapper<U>& other)
      : it_(other.it_)
  {}

  __host__ __device__ constexpr friend bool operator==(convertible_sentinel_wrapper const& self, const T& other)
  {
    return self.it_ == other;
  }
#if TEST_STD_VER <= 2017
  __host__ __device__ constexpr friend bool operator==(const T& other, convertible_sentinel_wrapper const& self)
  {
    return self.it_ == other;
  }
  __host__ __device__ constexpr friend bool operator!=(convertible_sentinel_wrapper const& self, const T& other)
  {
    return self.it_ != other;
  }
  __host__ __device__ constexpr friend bool operator!=(const T& other, convertible_sentinel_wrapper const& self)
  {
    return self.it_ != other;
  }
#endif // TEST_STD_VER <= 2017

  T it_;
};

struct NonSimpleNonCommonConvertibleView : IntBufferView
{
#if defined(TEST_COMPILER_NVRTC) // nvbug 3961621
  template <cuda::std::size_t N>
  __host__ __device__ constexpr NonSimpleNonCommonConvertibleView(int (&b)[N])
      : IntBufferView(b)
  {}
#else
  using IntBufferView::IntBufferView;
#endif

  __host__ __device__ constexpr int* begin()
  {
    return buffer_;
  }
  __host__ __device__ constexpr const int* begin() const
  {
    return buffer_;
  }
  __host__ __device__ constexpr convertible_sentinel_wrapper<int*> end()
  {
    return convertible_sentinel_wrapper<int*>(buffer_ + size_);
  }
  __host__ __device__ constexpr convertible_sentinel_wrapper<const int*> end() const
  {
    return convertible_sentinel_wrapper<const int*>(buffer_ + size_);
  }
};

static_assert(!cuda::std::ranges::common_range<NonSimpleNonCommonConvertibleView>);
static_assert(cuda::std::ranges::random_access_range<NonSimpleNonCommonConvertibleView>);
static_assert(!cuda::std::ranges::sized_range<NonSimpleNonCommonConvertibleView>);
static_assert(cuda::std::convertible_to<cuda::std::ranges::sentinel_t<NonSimpleNonCommonConvertibleView>,
                                        cuda::std::ranges::sentinel_t<NonSimpleNonCommonConvertibleView const>>);
static_assert(!simple_view<NonSimpleNonCommonConvertibleView>);

__host__ __device__ constexpr bool test()
{
  int buffer1[4] = {1, 2, 3, 4};
  int buffer2[5] = {1, 2, 3, 4, 5};
  cuda::std::ranges::zip_view v{NonSimpleNonCommonConvertibleView(buffer1), NonSimpleNonCommonConvertibleView(buffer2)};
  static_assert(!cuda::std::ranges::common_range<decltype(v)>);
  auto sent1                                             = v.end();
  cuda::std::ranges::sentinel_t<const decltype(v)> sent2 = sent1;
  static_assert(!cuda::std::is_same_v<decltype(sent1), decltype(sent2)>);

#if !defined(TEST_COMPILER_NVRTC)
  assert(v.begin() != sent2);
  assert(v.begin() + 4 == sent2);
#else
  assert(v.begin() != sent1);
  assert(v.begin() + 4 == sent1);
#endif // TEST_COMPILER_NVRTC
  assert(cuda::std::as_const(v).begin() != sent2);
  assert(cuda::std::as_const(v).begin() + 4 == sent2);

  // Cannot create a non-const iterator from a const iterator.
  static_assert(!cuda::std::constructible_from<decltype(sent1), decltype(sent2)>);
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
