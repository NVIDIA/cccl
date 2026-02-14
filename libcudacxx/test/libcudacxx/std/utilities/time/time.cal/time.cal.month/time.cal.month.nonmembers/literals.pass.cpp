//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <chrono>

// inline constexpr month January{1};
// inline constexpr month February{2};
// inline constexpr month March{3};
// inline constexpr month April{4};
// inline constexpr month May{5};
// inline constexpr month June{6};
// inline constexpr month July{7};
// inline constexpr month August{8};
// inline constexpr month September{9};
// inline constexpr month October{10};
// inline constexpr month November{11};
// inline constexpr month December{12};

#include <cuda/std/cassert>
#include <cuda/std/chrono>
#include <cuda/std/type_traits>

#include "test_macros.h"

__host__ __device__ constexpr bool test()
{
  static_assert(cuda::std::is_same_v<const cuda::std::chrono::month, decltype(cuda::std::chrono::January)>);
  static_assert(cuda::std::is_same_v<const cuda::std::chrono::month, decltype(cuda::std::chrono::February)>);
  static_assert(cuda::std::is_same_v<const cuda::std::chrono::month, decltype(cuda::std::chrono::March)>);
  static_assert(cuda::std::is_same_v<const cuda::std::chrono::month, decltype(cuda::std::chrono::April)>);
  static_assert(cuda::std::is_same_v<const cuda::std::chrono::month, decltype(cuda::std::chrono::May)>);
  static_assert(cuda::std::is_same_v<const cuda::std::chrono::month, decltype(cuda::std::chrono::June)>);
  static_assert(cuda::std::is_same_v<const cuda::std::chrono::month, decltype(cuda::std::chrono::July)>);
  static_assert(cuda::std::is_same_v<const cuda::std::chrono::month, decltype(cuda::std::chrono::August)>);
  static_assert(cuda::std::is_same_v<const cuda::std::chrono::month, decltype(cuda::std::chrono::September)>);
  static_assert(cuda::std::is_same_v<const cuda::std::chrono::month, decltype(cuda::std::chrono::October)>);
  static_assert(cuda::std::is_same_v<const cuda::std::chrono::month, decltype(cuda::std::chrono::November)>);
  static_assert(cuda::std::is_same_v<const cuda::std::chrono::month, decltype(cuda::std::chrono::December)>);

  assert(cuda::std::chrono::January == cuda::std::chrono::month(1));
  assert(cuda::std::chrono::February == cuda::std::chrono::month(2));
  assert(cuda::std::chrono::March == cuda::std::chrono::month(3));
  assert(cuda::std::chrono::April == cuda::std::chrono::month(4));
  assert(cuda::std::chrono::May == cuda::std::chrono::month(5));
  assert(cuda::std::chrono::June == cuda::std::chrono::month(6));
  assert(cuda::std::chrono::July == cuda::std::chrono::month(7));
  assert(cuda::std::chrono::August == cuda::std::chrono::month(8));
  assert(cuda::std::chrono::September == cuda::std::chrono::month(9));
  assert(cuda::std::chrono::October == cuda::std::chrono::month(10));
  assert(cuda::std::chrono::November == cuda::std::chrono::month(11));
  assert(cuda::std::chrono::December == cuda::std::chrono::month(12));

  assert(static_cast<unsigned>(cuda::std::chrono::January) == 1);
  assert(static_cast<unsigned>(cuda::std::chrono::February) == 2);
  assert(static_cast<unsigned>(cuda::std::chrono::March) == 3);
  assert(static_cast<unsigned>(cuda::std::chrono::April) == 4);
  assert(static_cast<unsigned>(cuda::std::chrono::May) == 5);
  assert(static_cast<unsigned>(cuda::std::chrono::June) == 6);
  assert(static_cast<unsigned>(cuda::std::chrono::July) == 7);
  assert(static_cast<unsigned>(cuda::std::chrono::August) == 8);
  assert(static_cast<unsigned>(cuda::std::chrono::September) == 9);
  assert(static_cast<unsigned>(cuda::std::chrono::October) == 10);
  assert(static_cast<unsigned>(cuda::std::chrono::November) == 11);
  assert(static_cast<unsigned>(cuda::std::chrono::December) == 12);

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
