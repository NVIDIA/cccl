//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<class F, class... Rs>
// zip_transform_view(F, Rs&&...) -> zip_transform_view<F, views::all_t<Rs>...>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>

#include "types.h"

struct Container
{
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};

struct Fn
{
  template <class... T>
  TEST_FUNC int operator()(T&&...) const
  {
    return 5;
  }
};

TEST_FUNC void testCTAD()
{
  static_assert(
    cuda::std::is_same_v<decltype(cuda::std::ranges::zip_transform_view(Fn{}, Container{})),
                         cuda::std::ranges::zip_transform_view<Fn, cuda::std::ranges::owning_view<Container>>>);

  static_assert(
    cuda::std::is_same_v<decltype(cuda::std::ranges::zip_transform_view(Fn{}, Container{}, IntView{})),
                         cuda::std::ranges::zip_transform_view<Fn, cuda::std::ranges::owning_view<Container>, IntView>>);

  Container c{};
  static_assert(
    cuda::std::is_same_v<decltype(cuda::std::ranges::zip_transform_view(Fn{}, Container{}, IntView{}, c)),
                         cuda::std::ranges::zip_transform_view<Fn,
                                                               cuda::std::ranges::owning_view<Container>,
                                                               IntView,
                                                               cuda::std::ranges::ref_view<Container>>>);
}

int main(int, char**)
{
  testCTAD();
}
