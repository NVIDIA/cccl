//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// template <class... Rs>
// zip_view(Rs&&...) -> zip_view<views::all_t<Rs>...>;

#include <cuda/std/cassert>
#include <cuda/std/ranges>
#include <cuda/std/utility>

#include "test_macros.h"

struct Container
{
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};

struct View : cuda::std::ranges::view_base
{
  TEST_FUNC int* begin() const;
  TEST_FUNC int* end() const;
};

// GCC really does not like those inside the function
using result_zip_owning_container      = cuda::std::ranges::zip_view<cuda::std::ranges::owning_view<Container>>;
using result_zip_owning_container_view = cuda::std::ranges::zip_view<cuda::std::ranges::owning_view<Container>, View>;
using result_zip_ref_container_view =
  cuda::std::ranges::zip_view<cuda::std::ranges::owning_view<Container>, View, cuda::std::ranges::ref_view<Container>>;

TEST_FUNC void testCTAD()
{
  static_assert(cuda::std::is_same_v<decltype(cuda::std::ranges::zip_view(Container{})), result_zip_owning_container>);

  static_assert(
    cuda::std::is_same_v<decltype(cuda::std::ranges::zip_view(Container{}, View{})), result_zip_owning_container_view>);

  Container c{};
  static_assert(
    cuda::std::is_same_v<decltype(cuda::std::ranges::zip_view(Container{}, View{}, c)), result_zip_ref_container_view>);
  unused(c);
}

int main(int, char**)
{
  return 0;
}
