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
//       requires Const && (convertible_to<iterator_t<Views>,
//                                         iterator_t<maybe-const<Const, Views>>> && ...);

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/tuple>

#include "../types.h"

using ConstIterIncompatibleView =
  BasicView<forward_iterator<int*>,
            forward_iterator<int*>,
            random_access_iterator<const int*>,
            random_access_iterator<const int*>>;
static_assert(!cuda::std::convertible_to<cuda::std::ranges::iterator_t<ConstIterIncompatibleView>,
                                         cuda::std::ranges::iterator_t<const ConstIterIncompatibleView>>);

__host__ __device__ constexpr bool test()
{
  int buffer[3] = {1, 2, 3};

  {
    cuda::std::ranges::zip_view v(NonSimpleCommon{buffer});
    auto iter1                                             = v.begin();
    cuda::std::ranges::iterator_t<const decltype(v)> iter2 = iter1;
    assert(iter1 == iter2);

    static_assert(!cuda::std::is_same_v<decltype(iter1), decltype(iter2)>);

    // We cannot create a non-const iterator from a const iterator.
    static_assert(!cuda::std::constructible_from<decltype(iter1), decltype(iter2)>);
  }

  {
    // underlying non-const to const not convertible
    cuda::std::ranges::zip_view v(ConstIterIncompatibleView{buffer});
    auto iter1 = v.begin();
    auto iter2 = cuda::std::as_const(v).begin();

    static_assert(!cuda::std::is_same_v<decltype(iter1), decltype(iter2)>);

    static_assert(!cuda::std::constructible_from<decltype(iter1), decltype(iter2)>);
    static_assert(!cuda::std::constructible_from<decltype(iter2), decltype(iter1)>);
    unused(iter1);
    unused(iter2);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
