//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

// libcu++ forward declares the std::get overloads for std::array, std::pair and std::tuple. The host headers are
// deliberately included after the libcu++ ones, so their definitions of std::get must be compatible with those
// declarations. libc++ marks std::get with an ABI tag, which a redeclaration must not add.

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <array>
#include <tuple>
#include <utility>

#include "test_macros.h"

void test_array()
{
  std::array<int, 3> arr{1, 2, 3};
  const std::array<int, 3>& carr = arr;

  static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(arr)), int&>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(carr)), const int&>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(cuda::std::move(arr))), int&&>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(cuda::std::move(carr))), const int&&>);

  assert(&cuda::std::get<0>(arr) == &arr[0]);
  assert(&cuda::std::get<1>(carr) == &arr[1]);
  assert(cuda::std::get<2>(cuda::std::move(arr)) == 3);
  assert(cuda::std::get<2>(cuda::std::move(carr)) == 3);

  assert(&std::get<0>(arr) == &arr[0]);
  assert(std::get<2>(carr) == 3);
}

void test_pair()
{
  std::pair<int, long> pr{1, 2};
  const std::pair<int, long>& cpr = pr;

  static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(pr)), int&>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::get<1>(cpr)), const long&>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(cuda::std::move(pr))), int&&>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::get<1>(cuda::std::move(cpr))), const long&&>);

  assert(&cuda::std::get<0>(pr) == &pr.first);
  assert(&cuda::std::get<1>(cpr) == &pr.second);
  assert(cuda::std::get<0>(cuda::std::move(pr)) == 1);
  assert(cuda::std::get<1>(cuda::std::move(cpr)) == 2);

  assert(&std::get<0>(pr) == &pr.first);
  assert(std::get<1>(cpr) == 2);
}

void test_tuple()
{
  std::tuple<int, long, short> tup{1, 2, 3};
  const std::tuple<int, long, short>& ctup = tup;

  static_assert(cuda::std::is_same_v<decltype(cuda::std::get<0>(tup)), int&>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::get<1>(ctup)), const long&>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::get<2>(cuda::std::move(tup))), short&&>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::get<2>(cuda::std::move(ctup))), const short&&>);

  assert(&cuda::std::get<0>(tup) == &std::get<0>(tup));
  assert(cuda::std::get<1>(ctup) == 2);
  assert(cuda::std::get<2>(cuda::std::move(tup)) == 3);
  assert(cuda::std::get<2>(cuda::std::move(ctup)) == 3);

  assert(std::get<1>(ctup) == 2);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (test_array(); test_pair(); test_tuple();))
  return 0;
}
