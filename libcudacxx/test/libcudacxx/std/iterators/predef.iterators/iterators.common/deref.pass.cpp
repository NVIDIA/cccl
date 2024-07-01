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

// decltype(auto) operator*();
// decltype(auto) operator*() const
//   requires dereferenceable<const I>;

#include <cuda/std/cassert>
#include <cuda/std/iterator>

#include "test_macros.h"
#include "types.h"

__host__ __device__ constexpr bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    auto iter1       = simple_iterator<int*>(buffer);
    auto commonIter1 = cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 =
      cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    const auto iter2       = simple_iterator<int*>(buffer);
    const auto commonIter2 = cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);

    assert(*iter1 == 1);
    assert(*commonIter1 == 1);

    assert(*iter2 == 1);
    assert(*commonIter2 == 1);

    assert(*(commonIter1++) == 1);
    assert(*commonIter1 == 2);
    assert(*(++commonIter1) == 3);
    assert(*commonIter1 == 3);

    for (auto i = 3; commonIter1 != commonSent1; ++i)
    {
      assert(*(commonIter1++) == i);
    }
  }
  {
    auto iter1       = value_iterator<int*>(buffer);
    auto commonIter1 = cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 =
      cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    const auto iter2       = value_iterator<int*>(buffer);
    const auto commonIter2 = cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);

    assert(*iter1 == 1);
    assert(*commonIter1 == 1);

    assert(*iter2 == 1);
    assert(*commonIter2 == 1);

    assert(*(commonIter1++) == 1);
    assert(*commonIter1 == 2);
    assert(*(++commonIter1) == 3);
    assert(*commonIter1 == 3);

    for (auto i = 3; commonIter1 != commonSent1; ++i)
    {
      assert(*(commonIter1++) == i);
    }
  }
  {
    auto iter1       = cpp17_input_iterator<int*>(buffer);
    auto commonIter1 = cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 =
      cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    const auto iter2       = cpp17_input_iterator<int*>(buffer);
    const auto commonIter2 = cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);

    assert(*iter1 == 1);
    assert(*commonIter1 == 1);

    assert(*iter2 == 1);
    assert(*commonIter2 == 1);

    assert(*(commonIter1++) == 1);
    assert(*commonIter1 == 2);
    assert(*(++commonIter1) == 3);
    assert(*commonIter1 == 3);

    for (auto i = 3; commonIter1 != commonSent1; ++i)
    {
      assert(*(commonIter1++) == i);
    }
  }
  {
    auto iter1       = forward_iterator<int*>(buffer);
    auto commonIter1 = cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 =
      cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    const auto iter2       = forward_iterator<int*>(buffer);
    const auto commonIter2 = cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);

    assert(*iter1 == 1);
    assert(*commonIter1 == 1);

    assert(*iter2 == 1);
    assert(*commonIter2 == 1);

    assert(*(commonIter1++) == 1);
    assert(*commonIter1 == 2);
    assert(*(++commonIter1) == 3);
    assert(*commonIter1 == 3);

    for (auto i = 3; commonIter1 != commonSent1; ++i)
    {
      assert(*(commonIter1++) == i);
    }
  }
  {
    auto iter1       = random_access_iterator<int*>(buffer);
    auto commonIter1 = cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 =
      cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    const auto iter2       = random_access_iterator<int*>(buffer);
    const auto commonIter2 = cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);

    assert(*iter1 == 1);
    assert(*commonIter1 == 1);

    assert(*iter2 == 1);
    assert(*commonIter2 == 1);

    assert(*(commonIter1++) == 1);
    assert(*commonIter1 == 2);
    assert(*(++commonIter1) == 3);
    assert(*commonIter1 == 3);

    for (auto i = 3; commonIter1 != commonSent1; ++i)
    {
      assert(*(commonIter1++) == i);
    }
  }

  return true;
}

int main(int, char**)
{
  test();
#if TEST_STD_VER >= 2020
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2020

  return 0;
}
