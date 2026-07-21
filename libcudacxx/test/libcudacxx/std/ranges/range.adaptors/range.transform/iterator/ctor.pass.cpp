//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// transform_view::<iterator>::transform_view::<iterator>();

#include <cuda/std/ranges>

#include "../types.h"
#include "test_macros.h"

struct NoDefaultInit
{
  using iterator_category = cuda::std::random_access_iterator_tag;
  using value_type        = int;
  using difference_type   = cuda::std::ptrdiff_t;
  using pointer           = int*;
  using reference         = int&;
  using self              = NoDefaultInit;

  TEST_FUNC NoDefaultInit(int*);

  TEST_FUNC reference operator*() const;
  TEST_FUNC pointer operator->() const;
#if TEST_HAS_SPACESHIP()
  TEST_FUNC auto operator<=>(const self&) const = default;
#else // ^^^ TEST_HAS_SPACESHIP() ^^^ / vvv !TEST_HAS_SPACESHIP() vvv
  TEST_FUNC bool operator<(const self&) const;
  TEST_FUNC bool operator<=(const self&) const;
  TEST_FUNC bool operator>(const self&) const;
  TEST_FUNC bool operator>=(const self&) const;
#endif // !TEST_HAS_SPACESHIP()

  TEST_FUNC friend bool operator==(const self&, int*);
#if TEST_STD_VER <= 2017
  TEST_FUNC friend bool operator==(int*, const self&);
  TEST_FUNC friend bool operator!=(const self&, int*);
  TEST_FUNC friend bool operator!=(int*, const self&);
#endif // TEST_STD_VER <= 2017

  TEST_FUNC self& operator++();
  TEST_FUNC self operator++(int);

  TEST_FUNC self& operator--();
  TEST_FUNC self operator--(int);

  TEST_FUNC self& operator+=(difference_type n);
  TEST_FUNC self operator+(difference_type n) const;
  TEST_FUNC friend self operator+(difference_type n, self x);

  TEST_FUNC self& operator-=(difference_type n);
  TEST_FUNC self operator-(difference_type n) const;
  TEST_FUNC difference_type operator-(const self&) const;

  TEST_FUNC reference operator[](difference_type n) const;
};

struct IterNoDefaultInitView : cuda::std::ranges::view_base
{
  TEST_FUNC NoDefaultInit begin() const;
  TEST_FUNC int* end() const;
  TEST_FUNC NoDefaultInit begin();
  TEST_FUNC int* end();
};

TEST_FUNC constexpr bool test()
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
#if defined(_CCCL_BUILTIN_ADDRESSOF)
  static_assert(test());
#endif // _CCCL_BUILTIN_ADDRESSOF

  return 0;
}
