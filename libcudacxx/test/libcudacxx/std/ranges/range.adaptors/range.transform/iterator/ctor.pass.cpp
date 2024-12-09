//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// transform_view::<iterator>::transform_view::<iterator>();

#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

struct NoDefaultInit
{
  typedef cuda::std::random_access_iterator_tag iterator_category;
  typedef int value_type;
  typedef cuda::std::ptrdiff_t difference_type;
  typedef int* pointer;
  typedef int& reference;
  typedef NoDefaultInit self;

  __host__ __device__ NoDefaultInit(int*);

  __host__ __device__ reference operator*() const;
  __host__ __device__ pointer operator->() const;
#ifndef TEST_HAS_NO_SPACESHIP_OPERATOR
  __host__ __device__ auto operator<=>(const self&) const = default;
#else // ^^^ !TEST_HAS_NO_SPACESHIP_OPERATOR ^^^ / vvv TEST_HAS_NO_SPACESHIP_OPERATOR vvv
  __host__ __device__ bool operator<(const self&) const;
  __host__ __device__ bool operator<=(const self&) const;
  __host__ __device__ bool operator>(const self&) const;
  __host__ __device__ bool operator>=(const self&) const;
#endif // TEST_HAS_NO_SPACESHIP_OPERATOR

  __host__ __device__ friend bool operator==(const self&, int*);
#if TEST_STD_VER <= 2017
  __host__ __device__ friend bool operator==(int*, const self&);
  __host__ __device__ friend bool operator!=(const self&, int*);
  __host__ __device__ friend bool operator!=(int*, const self&);
#endif // TEST_STD_VER <= 2017

  __host__ __device__ self& operator++();
  __host__ __device__ self operator++(int);

  __host__ __device__ self& operator--();
  __host__ __device__ self operator--(int);

  __host__ __device__ self& operator+=(difference_type n);
  __host__ __device__ self operator+(difference_type n) const;
  __host__ __device__ friend self operator+(difference_type n, self x);

  __host__ __device__ self& operator-=(difference_type n);
  __host__ __device__ self operator-(difference_type n) const;
  __host__ __device__ difference_type operator-(const self&) const;

  __host__ __device__ reference operator[](difference_type n) const;
};

struct IterNoDefaultInitView : cuda::std::ranges::view_base
{
  __host__ __device__ NoDefaultInit begin() const;
  __host__ __device__ int* end() const;
  __host__ __device__ NoDefaultInit begin();
  __host__ __device__ int* end();
};

__host__ __device__ constexpr bool test()
{
  cuda::std::ranges::transform_view<MoveOnlyView, PlusOne> transformView{};
  auto iter = cuda::std::move(transformView).begin();
  cuda::std::ranges::iterator_t<cuda::std::ranges::transform_view<MoveOnlyView, PlusOne>> i2(iter);
  unused(i2);
  cuda::std::ranges::iterator_t<const cuda::std::ranges::transform_view<MoveOnlyView, PlusOne>> constIter(iter);
  unused(constIter);

  static_assert(cuda::std::default_initializable<
                cuda::std::ranges::iterator_t<cuda::std::ranges::transform_view<MoveOnlyView, PlusOne>>>);
  static_assert(!cuda::std::default_initializable<
                cuda::std::ranges::iterator_t<cuda::std::ranges::transform_view<IterNoDefaultInitView, PlusOne>>>);

  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // _LIBCUDACXX_ADDRESSOF

  return 0;
}
