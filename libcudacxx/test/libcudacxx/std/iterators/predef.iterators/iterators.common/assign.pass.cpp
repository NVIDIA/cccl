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

// template<class I2, class S2>
//   requires convertible_to<const I2&, I> && convertible_to<const S2&, S> &&
//            assignable_from<I&, const I2&> && assignable_from<S&, const S2&>
//     common_iterator& operator=(const common_iterator<I2, S2>& x);

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/ranges>

#include "test_macros.h"
#include "types.h"

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  {
    auto iter1       = cpp17_input_iterator<int*>(buffer);
    auto commonIter1 = cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonIter2 =
      cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(cpp17_input_iterator<int*>(buffer + 1));

    assert(*commonIter1 == 1);
    assert(*commonIter2 == 2);
    assert(commonIter1 != commonIter2);

    commonIter1 = commonIter2;

    assert(*commonIter1 == 2);
    assert(*commonIter2 == 2);
    assert(commonIter1 == commonIter2);
  }
  {
    auto iter1       = forward_iterator<int*>(buffer);
    auto commonIter1 = cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonIter2 =
      cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(forward_iterator<int*>(buffer + 1));

    assert(*commonIter1 == 1);
    assert(*commonIter2 == 2);
    assert(commonIter1 != commonIter2);

    commonIter1 = commonIter2;

    assert(*commonIter1 == 2);
    assert(*commonIter2 == 2);
    assert(commonIter1 == commonIter2);
  }
  {
    auto iter1       = random_access_iterator<int*>(buffer);
    auto commonIter1 = cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 =
      cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    auto commonIter2 = cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1 + 1);
    auto commonSent2 =
      cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 7});

    assert(*commonIter1 == 1);
    assert(*commonIter2 == 2);
    assert(commonIter1 != commonIter2);

    commonIter1 = commonIter2;

    assert(*commonIter1 == 2);
    assert(*commonIter2 == 2);
    assert(commonIter1 == commonIter2);

    assert(cuda::std::next(commonIter1, 6) != commonSent1);
    assert(cuda::std::next(commonIter1, 6) == commonSent2);

    commonSent1 = commonSent2;

    assert(cuda::std::next(commonIter1, 6) == commonSent1);
    assert(cuda::std::next(commonIter1, 6) == commonSent2);
  }
  {
    auto iter1       = assignable_iterator<int*>(buffer);
    auto iter2       = forward_iterator<int*>(buffer + 1);
    auto commonIter1 = cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent1 =
      cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(sentinel_type<int*>{buffer + 8});

    auto commonIter2 = cuda::std::common_iterator<decltype(iter2), sentinel_type<int*>>(iter2);
    auto commonSent2 =
      cuda::std::common_iterator<decltype(iter2), sentinel_type<int*>>(sentinel_type<int*>{buffer + 7});

    assert(*commonIter1 == 1);
    assert(*commonIter2 == 2);

    commonIter1 = commonIter2;

    assert(*commonIter1 == 2);
    assert(*commonIter2 == 2);
    assert(commonIter1 == commonIter2);

    assert(cuda::std::next(commonIter1, 6) != commonSent1);
    assert(cuda::std::next(commonIter1, 6) == commonSent2);

    commonSent1 = commonSent2;

    assert(cuda::std::next(commonIter1, 6) == commonSent1);
    assert(cuda::std::next(commonIter1, 6) == commonSent2);

    commonIter1 = commonSent1;

    assert(commonIter1 == commonSent2);

    commonIter1 = commonSent2;

    assert(commonIter1 == commonSent2);
  }

  return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
void test_exceptions()
{
  int buffer[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  {
    auto iter1       = maybe_valueless_iterator<int*>(buffer);
    auto iter2       = forward_iterator<int*>(buffer);
    auto commonIter1 = cuda::std::common_iterator<decltype(iter1), sentinel_type<int*>>(iter1);
    auto commonSent2 = cuda::std::common_iterator<decltype(iter1), sentinel_throws_on_convert<int*>>(
      sentinel_throws_on_convert<int*>{buffer + 8});
    auto commonIter2 = cuda::std::common_iterator<decltype(iter2), sentinel_type<int*>>(iter2);

    try
    {
      commonIter1 = commonSent2;
      assert(false);
    }
    catch (int x)
    {
      assert(x == 42);
      commonIter1 = commonIter2;
    }

    assert(*commonIter1 == 1);
  }
}
#endif // !TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
  test();
#ifndef TEST_HAS_NO_EXCEPTIONS
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // !TEST_HAS_NO_EXCEPTIONS

#if TEST_STD_VER >= 2020
  static_assert(test(), "");
#endif // TEST_STD_VER >= 2020

  return 0;
}
