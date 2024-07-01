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

// constexpr iterator(iterator<!Const> i)
//   requires Const && convertible_to<iterator_t<V>, iterator_t<Base>>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "../types.h"

template <bool Const>
struct ConvertibleIter : IterBase<ConvertibleIter<Const>>
{
  using iterator_category = cuda::std::random_access_iterator_tag;
  using value_type        = cuda::std::tuple<int>;
  using difference_type   = intptr_t;

  bool movedFromOtherConst = false;
  int i                    = 0;

  constexpr ConvertibleIter() = default;
  __host__ __device__ constexpr ConvertibleIter(int ii)
      : i(ii)
  {}
  template <bool otherConst, cuda::std::enable_if_t<Const != otherConst, int> = 0>
  __host__ __device__ constexpr ConvertibleIter(ConvertibleIter<otherConst> it)
      : movedFromOtherConst(true)
      , i(it.i)
  {}
};

template <class Iter, class ConstIter>
struct BasicView : cuda::std::ranges::view_base
{
  __host__ __device__ Iter begin()
  {
    return Iter{};
  }
  __host__ __device__ Iter end()
  {
    return Iter{};
  }

  __host__ __device__ ConstIter begin() const
  {
    return ConstIter{};
  }
  __host__ __device__ ConstIter end() const
  {
    return ConstIter{};
  }
};

template <class View>
using ElemIter = cuda::std::ranges::iterator_t<cuda::std::ranges::elements_view<View, 0>>;

template <class View>
using ConstElemIter = cuda::std::ranges::iterator_t<const cuda::std::ranges::elements_view<View, 0>>;

using ConvertibleView = BasicView<ConvertibleIter<false>, ConvertibleIter<true>>;
using NonConvertibleView =
  BasicView<forward_iterator<cuda::std::tuple<int>*>, bidirectional_iterator<cuda::std::tuple<int>*>>;

static_assert(cuda::std::is_constructible_v<ConstElemIter<ConvertibleView>, ElemIter<ConvertibleView>>);
static_assert(!cuda::std::is_constructible_v<ElemIter<ConvertibleView>, ConstElemIter<ConvertibleView>>);
static_assert(!cuda::std::is_constructible_v<ConstElemIter<NonConvertibleView>, ElemIter<NonConvertibleView>>);
static_assert(!cuda::std::is_constructible_v<ElemIter<NonConvertibleView>, ConstElemIter<NonConvertibleView>>);

__host__ __device__ constexpr bool test()
{
  ElemIter<ConvertibleView> iter{ConvertibleIter<false>{5}};
  ConstElemIter<ConvertibleView> constIter = iter; // implicit
  assert(constIter.base().movedFromOtherConst);
  assert(constIter.base().i == 5);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
