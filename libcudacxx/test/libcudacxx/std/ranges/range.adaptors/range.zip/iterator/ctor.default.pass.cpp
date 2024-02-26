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

// iterator() = default;

#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "../types.h"
#include "test_macros.h"

struct PODIter
{
  int i; // deliberately uninitialised

  using iterator_category = cuda::std::random_access_iterator_tag;
  using value_type        = int;
  using difference_type   = intptr_t;

  __host__ __device__ constexpr int operator*() const
  {
    return i;
  }

  __host__ __device__ constexpr PODIter& operator++()
  {
    return *this;
  }
  __host__ __device__ constexpr void operator++(int) {}

#if TEST_STD_VER >= 2020
  __host__ __device__ friend constexpr bool operator==(const PODIter&, const PODIter&) = default;
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
  __host__ __device__ friend constexpr bool operator==(const PODIter& lhs, const PODIter& rhs)
  {
    return lhs.i == rhs.i;
  }
  __host__ __device__ friend constexpr bool operator!=(const PODIter& lhs, const PODIter& rhs)
  {
    return lhs.i != rhs.i;
  }
#endif // TEST_STD_VER <=2017
};

struct IterDefaultCtrView : cuda::std::ranges::view_base
{
  __host__ __device__ PODIter begin() const
  {
    return PODIter{};
  }
  __host__ __device__ PODIter end() const
  {
    return PODIter{};
  }
};

struct IterNoDefaultCtrView : cuda::std::ranges::view_base
{
  __host__ __device__ cpp20_input_iterator<int*> begin() const
  {
    return cpp20_input_iterator<int*>{nullptr};
  }
  __host__ __device__ sentinel_wrapper<cpp20_input_iterator<int*>> end() const
  {
    return sentinel_wrapper<cpp20_input_iterator<int*>>{};
  }
};

template <class... Views>
using zip_iter = cuda::std::ranges::iterator_t<cuda::std::ranges::zip_view<Views...>>;

static_assert(!cuda::std::default_initializable<zip_iter<IterNoDefaultCtrView>>);
static_assert(!cuda::std::default_initializable<zip_iter<IterNoDefaultCtrView, IterDefaultCtrView>>);
static_assert(!cuda::std::default_initializable<zip_iter<IterNoDefaultCtrView, IterNoDefaultCtrView>>);
static_assert(cuda::std::default_initializable<zip_iter<IterDefaultCtrView>>);
static_assert(cuda::std::default_initializable<zip_iter<IterDefaultCtrView, IterDefaultCtrView>>);

__host__ __device__ constexpr bool test()
{
  using ZipIter = zip_iter<IterDefaultCtrView>;
  {
    ZipIter iter;
    auto [x] = *iter;
    assert(x == 0); // PODIter has to be initialised to have value 0
  }

  {
    ZipIter iter = {};
    auto [x]     = *iter;
    assert(x == 0); // PODIter has to be initialised to have value 0
  }
  return true;
}

int main(int, char**)
{
  test();
#if defined(_LIBCUDACXX_ADDRESSOF)
  static_assert(test(), "");
#endif // defined(_LIBCUDACXX_ADDRESSOF)

  return 0;
}
