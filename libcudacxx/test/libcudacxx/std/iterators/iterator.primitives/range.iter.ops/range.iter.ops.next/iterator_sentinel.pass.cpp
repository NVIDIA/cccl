//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// ranges::next(it, bound)

#include <cuda/std/cassert>
#include <cuda/std/concepts>
#include <cuda/std/iterator>
#include <cuda/std/utility>

#include "../types.h"
#include "test_iterators.h"

template <typename It>
__host__ __device__ constexpr void check_assignable(int* first, int* last, int* expected)
{
  It it(first);
  auto sent = assignable_sentinel(It(last));
  It result = cuda::std::ranges::next(cuda::std::move(it), sent);
  assert(base(result) == expected);
}

template <typename It>
__host__ __device__ constexpr void check_sized_sentinel(int* first, int* last, int* expected)
{
  auto size = (last - first);

  It it(first);
  auto sent             = distance_apriori_sentinel(size);
  decltype(auto) result = cuda::std::ranges::next(cuda::std::move(it), sent);
  static_assert(cuda::std::same_as<decltype(result), It>);
  assert(base(result) == expected);
}

template <typename It>
__host__ __device__ constexpr void check_sentinel(int* first, int* last, int* expected)
{
  It it(first);
  auto sent             = sentinel_wrapper(It(last));
  decltype(auto) result = cuda::std::ranges::next(cuda::std::move(it), sent);
  static_assert(cuda::std::same_as<decltype(result), It>);
  assert(base(result) == expected);
}

__host__ __device__ constexpr bool test()
{
  int range[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};

  for (int n = 0; n != 10; ++n)
  {
    check_assignable<cpp17_input_iterator<int*>>(range, range + n, range + n);
    check_assignable<cpp20_input_iterator<int*>>(range, range + n, range + n);
    check_assignable<forward_iterator<int*>>(range, range + n, range + n);
    check_assignable<bidirectional_iterator<int*>>(range, range + n, range + n);
    check_assignable<random_access_iterator<int*>>(range, range + n, range + n);
    check_assignable<contiguous_iterator<int*>>(range, range + n, range + n);
    check_assignable<int*>(range, range + n, range + n);

    check_sized_sentinel<cpp17_input_iterator<int*>>(range, range + n, range + n);
    check_sized_sentinel<cpp20_input_iterator<int*>>(range, range + n, range + n);
    check_sized_sentinel<forward_iterator<int*>>(range, range + n, range + n);
    check_sized_sentinel<bidirectional_iterator<int*>>(range, range + n, range + n);
    check_sized_sentinel<random_access_iterator<int*>>(range, range + n, range + n);
    check_sized_sentinel<contiguous_iterator<int*>>(range, range + n, range + n);
    check_sized_sentinel<int*>(range, range + n, range + n);

    check_sentinel<cpp17_input_iterator<int*>>(range, range + n, range + n);
    check_sentinel<cpp20_input_iterator<int*>>(range, range + n, range + n);
    check_sentinel<forward_iterator<int*>>(range, range + n, range + n);
    check_sentinel<bidirectional_iterator<int*>>(range, range + n, range + n);
    check_sentinel<random_access_iterator<int*>>(range, range + n, range + n);
    check_sentinel<contiguous_iterator<int*>>(range, range + n, range + n);
    check_sentinel<int*>(range, range + n, range + n);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
